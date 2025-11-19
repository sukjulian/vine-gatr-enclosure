# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
import random
import re
import traceback
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import filterfalse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
from uuid import uuid4

import numpy as np
import pyvista
import torch
import torch_geometric as pyg
import wandb
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm


class BaseExperiment(ABC):
    def __init__(
        self,
        config: DictConfig,
        checkpoint_dir: Optional[str] = None,
        id: Optional[str] = None,
        should_resume: bool = False,
    ):
        self.config = self._parse_config(config)

        ##############################################################
        # these are needed to ensure recovery from preemption
        if id is not None:
            self.config.id = id

        self._checkpoints_dir = (
            os.path.join("checkpoints", self._id)
            if checkpoint_dir is None
            else os.path.join(checkpoint_dir, self._id)
        )

        self._should_resume = should_resume

        if hasattr(config.training, "train_seed"):
            torch.manual_seed(config.training.train_seed)
            np.random.seed(config.training.train_seed)
            random.seed(config.training.train_seed)
        ##############################################################

        self._init_wandb()

        self._model = self.get_model(self.config)
        self._dataset = self.get_dataset(self.config)
        self._dataset_splits_idcs = self.get_dataset_splits_idcs(self.config)

        self._test_device: Union[None, torch.device] = None

        self._callback = None

        self._epoch = None

    def set_callback(self, callback: Any) -> None:
        self._callback = callback

    # @property
    # def _checkpoints_dir(self) -> str:
    #     return os.path.join("checkpoints", self._id)

    @staticmethod
    def _parse_config(config: DictConfig) -> DictConfig:
        config_ = cast(
            dict, OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        )

        # Minimal required configuration
        config_["training"].setdefault("batch_size", 1)
        config_["training"].setdefault("grad_accumulation_num_steps", 1)
        config_["training"].setdefault("lr", 3e-4)
        config_["training"].setdefault("lr_decay_factor", 1.0)
        config_["training"].setdefault("lr_decay_factor_plateau", 0.2)
        config_["training"].setdefault("lr_scheduler", "exponential")
        config_["training"].setdefault("var_lr", config_["training"]["lr"])

        return OmegaConf.create(config_)

    def _init_wandb(self) -> None:
        if "wandb" in self.config and self.config.wandb is True:
            config = cast(dict, OmegaConf.to_container(self.config))
            if "wandb_project" in self.config:
                project = self.config.wandb_project
                project = f"gatr-enclosure|{project}"
            else:
                project = "gatr-enclosure"

            print("using wadnb with project", project)
            wandb.init(config=config, project=project, name=self._id)

            self.config = OmegaConf.create(dict(wandb.config.items()))

    @cached_property
    def _id(self) -> str:
        uid = uuid4().hex if "id" not in self.config else self.config.id

        class_id = "-".join(
            re.findall(r"[A-Z][a-z]*|\d+", self.__class__.__name__)
        ).lower()
        instance_id = os.path.join(class_id, uid)

        return instance_id

    @abstractmethod
    def get_model(self, config: DictConfig) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loss_fn(
        y: torch.Tensor, data: pyg.data.Data, config: DictConfig
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def metric_fn(
        y: torch.Tensor,
        y_data: torch.Tensor,
        scatter_idcs: torch.Tensor,
        config: DictConfig,
    ) -> torch.Tensor:
        raise NotImplementedError

    def train_model(self, device: torch.device) -> None:

        optimiser = self._get_optimiser()

        if (
            "automatic_mixed_precision" in self.config.training
            and self.config.training.automatic_mixed_precision is True
        ):
            grad_scaler = torch.amp.GradScaler()

        if self.config.training.lr_scheduler == "exponential":
            lr_scheduler = ExponentialLR(
                optimiser, gamma=self.config.training.lr_decay_factor
            )
        elif self.config.training.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(
                optimiser,
                mode="min",
                patience=4,
                threshold=2e-2,
                factor=self.config.training.lr_decay_factor_plateau,
                min_lr=1e-5,
            )
        else:
            raise ValueError()

        best_metric = 1e8

        self._model.to(device)
        starting_epoch = self._load_checkpoints(self._model, optimiser, lr_scheduler)
        starting_epoch += 1
        if starting_epoch > 0:
            print(f"Resuming training starting from epoch {starting_epoch}")

            if wandb.run is not None:
                best_metric = wandb.run.summary["best_metric"]

        pbar_epochs = tqdm(
            range(starting_epoch, self.config.training.num_epochs),
            desc="Epochs",
            position=0,
            leave=True,
        )

        if (
            "wandb" in self.config and self.config.wandb is True
        ) or wandb.run is not None:
            print("Watching model gradients!")
            wandb.watch(
                models=self._model,
                log="gradients",
                log_freq=len(self._get_data_loader("training")),
            )

        # Training loop
        for epoch in pbar_epochs:

            self._epoch = epoch

            loss_cache: Dict[str, Dict[str, List[float]]] = {
                mode: {term: [] for term in ["total_loss", "loss"]}
                for mode in ["train", "valid"]
                # for mode in ["training", "validation"]
            }

            self._model.train()  # training mode

            pbar_training = tqdm(
                self._get_data_loader("training"),
                desc="Training split",
                position=1,
                leave=False,
            )

            for step_count, data in enumerate(pbar_training, start=1):
                data.to(device)

                if "grad_scaler" in locals():
                    total_loss, other_metrics = (
                        self._optimiser_step_automatic_mixed_precision(
                            optimiser, data, grad_scaler, step_count
                        )
                    )
                else:
                    total_loss, other_metrics = self._optimiser_step(
                        optimiser, data, step_count
                    )

                description = ""
                loss_cache["train"]["total_loss"].append(total_loss.item())
                for metric_name, metric_value in other_metrics.items():
                    if metric_name not in loss_cache["train"]:
                        loss_cache["train"][metric_name] = []
                    loss_cache["train"][metric_name].append(metric_value)

                    description += f"{metric_name}: {metric_value:.3f} |"

                # pbar_training.set_description(f"Loss: {total_loss:.3f}")
                pbar_training.set_description(description)

            model_log_dict_train = (
                {f"train/{k}": v for k, v in self._model.get_log_dict().items()}
                if hasattr(self._model, "get_log_dict")
                else {}
            )  # computed for last batch only

            # self._save_checkpoints(epoch, self._model, optimiser, lr_scheduler)

            self._model.eval()  # inference mode

            # Validation loop
            y = []
            y_data = []
            scatter_idcs = []

            with torch.no_grad():

                validation_loader = self._get_data_loader("validation")

                if validation_loader is not None:
                    for idx, data in enumerate(
                        tqdm(
                            validation_loader,
                            desc="Validation split",
                            position=1,
                            leave=False,
                        )
                    ):
                        data.to(device)

                        total_loss, other_metrics, y_ = self._compute_loss(data)
                        loss_cache["valid"]["total_loss"].append(total_loss.item())
                        for metric_name, metric_value in other_metrics.items():
                            if metric_name not in loss_cache["valid"]:
                                loss_cache["valid"][metric_name] = []
                            loss_cache["valid"][metric_name].append(metric_value)

                        y.append(y_)
                        y_data.append(data.y)
                        scatter_idcs.append(
                            torch.tensor(idx, device=device).expand(y_.size(0))
                        )

                    metric = self.metric_fn(
                        torch.cat(y),
                        torch.cat(y_data),
                        torch.cat(scatter_idcs),
                        config=self.config,
                    ).tolist()

                    model_log_dict = (
                        self._model.get_log_dict()
                        if hasattr(self._model, "get_log_dict")
                        else {}
                    )  # computed for last batch only
                else:
                    metric = []
                    model_log_dict = dict()

            if isinstance(lr_scheduler, ExponentialLR):
                lr_scheduler.step()
            elif isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(mean(loss_cache["valid"]["loss"]))
            else:
                raise ValueError()

            logs = {
                **{
                    # f"{term}_{mode}": mean(value)
                    f"{mode}/{term}": mean(value)
                    for mode, cacheddicts in loss_cache.items()
                    for term, value in cacheddicts.items()
                    if len(value) > 0
                },
                "epoch": epoch,
                "lr": optimiser.param_groups[0]["lr"],
                **model_log_dict,
                **model_log_dict_train,
            }

            if len(metric) > 0:
                logs["metric"] = mean(metric)
                pbar_epochs.set_description(f"Metric: {logs['metric']:.3f}")

                if logs["metric"] < best_metric:
                    best_metric = logs["metric"]
                    if wandb.run is not None:
                        wandb.run.summary["best_metric"] = logs["metric"]

                    self._save_best_model(self._model)

            self._log_to_wandb(logs, epoch)

            self._save_checkpoints(epoch, self._model, optimiser, lr_scheduler)

            if self._callback is not None:
                self._callback(logs, epoch=epoch)

    def test_model(self, device: torch.device) -> None:

        self._load_best_model(self._model)

        self._model.to(device)
        self._model.eval()  # inference mode

        y = []
        y_data = []
        scatter_idcs = []

        with torch.no_grad():

            if hasattr(self._model, "stopwatch"):
                timings = self._model.stopwatch.restart()

            for idx, data in enumerate(
                tqdm(
                    self._get_data_loader("test"),
                    desc="Test split",
                    position=0,
                    leave=False,
                )
            ):
                data.to(device)

                y.append(self._model(data))
                y_data.append(data.y)
                scatter_idcs.append(
                    torch.tensor(idx, device=device).expand(data.y.size(0))
                )

            test_metric = self.metric_fn(
                torch.cat(y),
                torch.cat(y_data),
                torch.cat(scatter_idcs),
                config=self.config,
            )
            self._print_statistics(test_metric)

            self._test_device = device

            if hasattr(self._model, "stopwatch"):
                timings = self._model.stopwatch.mean_duration_splits
                print(timings)
                if wandb.run is not None:
                    for k, v in timings.items():
                        wandb.run.summary[f"timing/{k}"] = v

            if wandb.run is not None:
                wandb.run.summary["test/metric"] = (test_metric).mean().item()
                wandb.run.summary["test/metric_std"] = (test_metric).std().item()

            try:
                self._save_random_visualisation()
            except:
                print(traceback.format_exc())

    def _get_optimiser(self) -> torch.optim.Optimizer:

        def filter_fun(param: torch.nn.Parameter) -> bool:
            return hasattr(param, "var_lr_flag") and param.var_lr_flag is True

        iterator = zip(
            (filter, filterfalse),
            (self.config.training.var_lr, self.config.training.lr),
        )

        optimiser = Adam(
            [
                {"params": fun(filter_fun, self._model.parameters()), "lr": lr}
                for fun, lr in iterator
            ]
        )

        return optimiser

    def _get_data_loader(
        self, split_id: Literal["training", "validation", "test"]
    ) -> pyg.loader.DataLoader:

        assert split_id in ["training", "validation", "test"]
        iseval = split_id != "training"

        if len(self._dataset_splits_idcs[split_id]) == 0:
            return None

        dataset = self._dataset[self._dataset_splits_idcs[split_id]]

        # batch_size = 1 if split_id == "test" else self.config.training.batch_size
        # return pyg.loader.DataLoader(dataset, batch_size, shuffle=True)
        batch_size = 1 if iseval else self.config.training.batch_size
        return pyg.loader.DataLoader(dataset, batch_size, shuffle=not iseval)

    def _optimiser_step(
        self, optimiser: torch.optim.Optimizer, data: pyg.data.Data, step_count: int
    ) -> torch.Tensor:

        total_loss, other_metrics, _ = self._compute_loss(data)
        total_loss.backward()

        if (
            "grad_clipping" in self.config.training
            and self.config.training.grad_clipping is True
        ):
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )

        grad_accumulation_is_complete = (
            step_count % self.config.training.grad_accumulation_num_steps == 0
        )
        data_loader_is_depleted = step_count == len(self._get_data_loader("training"))

        if grad_accumulation_is_complete or data_loader_is_depleted:

            optimiser.step()
            optimiser.zero_grad()

        return total_loss, other_metrics

    def _optimiser_step_automatic_mixed_precision(
        self,
        optimiser: torch.optim.Optimizer,
        data: pyg.data.Data,
        grad_scaler: torch.amp.GradScaler,
        step_count: int,
    ) -> torch.Tensor:

        with torch.amp.autocast("cuda"):
            total_loss, other_metrics, _ = self._compute_loss(data)
            total_loss /= self.config.training.grad_accumulation_num_steps
            # other_metrics = {
            #     key: val / self.config.training.grad_accumulation_num_steps
            #     for key, val in other_metrics.items()
            # }

        grad_scaler.scale(total_loss).backward()

        grad_accumulation_is_complete = (
            step_count % self.config.training.grad_accumulation_num_steps == 0
        )
        data_loader_is_depleted = step_count == len(self._get_data_loader("training"))

        if grad_accumulation_is_complete or data_loader_is_depleted:

            if (
                "grad_clipping" in self.config.training
                and self.config.training.grad_clipping is True
            ):
                grad_scaler.unscale_(optimiser)

                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )

            grad_scaler.step(optimiser)
            grad_scaler.update()

            optimiser.zero_grad()

        return total_loss, other_metrics

    def _compute_loss(
        self, data: pyg.data.Data
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:

        y = self._model(data, batch=data.batch)  # forward pass

        loss = self.loss_fn(y, data, config=self.config)

        metrics = {"loss": loss.item()}

        if hasattr(self._model, "regularization_fn"):
            regularization: dict = self._model.regularization_fn(data, batch=data.batch)

            total_loss = loss

            for key, regularization_val in regularization.items():

                metrics[key] = regularization_val.item()

                if hasattr(self.config.training, f"warmup_{key}"):
                    warmup_epoch = getattr(self.config.training, f"warmup_{key}")

                    if self._epoch > warmup_epoch:
                        continue

                if hasattr(self.config.training, f"lambda_{key}"):
                    lambda_reg = getattr(self.config.training, f"lambda_{key}")

                    if lambda_reg != 0.0:
                        total_loss = total_loss + lambda_reg * regularization_val

        else:
            total_loss = loss

        return total_loss, metrics, y

    def _save_best_model(self, model: torch.nn.Module) -> None:
        self._create_checkpoints_dir()
        torch.save(
            model.state_dict(),
            os.path.join(self._checkpoints_dir, "best_model_checkpoint.pt"),
        )

    def _load_best_model(self, model: torch.nn.Module) -> None:
        checkpoint_path = os.path.join(
            self._checkpoints_dir, "best_model_checkpoint.pt"
        )
        if os.path.isfile(checkpoint_path):
            print("Loading best checkpoint.")
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(
                "Checkpoint file not found, cannot load best checkpoint. Keeping current weights"
            )
            # raise FileNotFoundError

    def _save_checkpoints(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:

        self._create_checkpoints_dir()

        torch.save(
            model.state_dict(),
            os.path.join(self._checkpoints_dir, "model_checkpoint.pt"),
        )
        torch.save(
            optimiser.state_dict(),
            os.path.join(self._checkpoints_dir, "optimiser_checkpoint.pt"),
        )

        checkpoint_path = os.path.join(self._checkpoints_dir, "current_checkpoint.h5")
        torch.save(
            {
                "epoch": epoch,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state(),
                "np_rng": np.random.get_state(),
                "py_rng": random.getstate(),
            },
            checkpoint_path,
        )
        if wandb.run is not None:
            wandb.run.summary["current_checkpoint"] = checkpoint_path

    def _load_checkpoints(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> int:

        path_model_checkpoint = os.path.join(
            self._checkpoints_dir, "model_checkpoint.pt"
        )
        path_optimiser_checkpoint = os.path.join(
            self._checkpoints_dir, "optimiser_checkpoint.pt"
        )
        checkpoint_path = os.path.join(self._checkpoints_dir, "current_checkpoint.h5")

        # # we don't do the asserts since it could have failed before the end of the first epoch and therefore never stored any checkpoint
        # if self._should_resume:
        #     assert os.path.isfile(path_model_checkpoint), path_model_checkpoint
        #     assert os.path.isfile(path_optimiser_checkpoint), path_optimiser_checkpoint
        #     assert os.path.isfile(checkpoint_path), checkpoint_path

        if os.path.exists(path_model_checkpoint):
            model.load_state_dict(torch.load(path_model_checkpoint))
            print("Resuming from previous model checkpoint.")

        if os.path.exists(path_optimiser_checkpoint):
            optimiser.load_state_dict(torch.load(path_optimiser_checkpoint))
            print("Resuming from previous optimiser checkpoint.")

        # try:
        #     checkpoint_path = wandb.run.summary["current_checkpoint"]
        # except KeyError:
        #     print('Warning: no checkpoint found so starting training from beginning')
        #     # set last epoch to -1, so the training starts at 0
        #     return -1

        if not os.path.isfile(checkpoint_path):
            print("Warning: no checkpoint found so starting training from beginning")
            # set last epoch to -1, so the training starts at 0
            return -1

        checkpoint = torch.load(checkpoint_path)

        if scheduler is not None:
            assert "scheduler" in checkpoint
            assert checkpoint["scheduler"] is not None
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("Resuming from previous scheduler checkpoint.")

        torch.set_rng_state(checkpoint["torch_cpu_rng"])
        torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])
        np.random.set_state(checkpoint["np_rng"])
        random.setstate(checkpoint["py_rng"])
        print("Resuming from previous rng state checkpoint.")

        return checkpoint["epoch"]

    def _create_checkpoints_dir(self) -> None:
        Path(self._checkpoints_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _log_to_wandb(loss_cache: Dict[str, Any], epoch: int) -> None:

        if wandb.run is not None:
            wandb.log(loss_cache, step=epoch)

    @staticmethod
    def _print_statistics(metric: torch.Tensor) -> None:

        table = PrettyTable(("", "Mean", "Standard deviation"))
        table.add_row(["Metric", f"{metric.mean():.2%}", f"{metric.std():.2%}"])

        print(table)

    def _save_random_visualisation(self) -> None:
        dataset = self._dataset[self._dataset_splits_idcs["test"]]
        idx = random.randint(0, len(dataset) - 1)

        data = dataset[idx]
        data.to(self._test_device)

        y = self._model(data).cpu()

        data.cpu()
        pos = data.pop("pos")

        if (
            hasattr(self._model, "decoder_id_query_idcs")
            and self._model.decoder_id_query_idcs is not None
        ):
            decoder_query_idcs = data.pop(f"{self._model.decoder_id_query_idcs}_index")

        point_cloud = pyvista.PolyData(pos.numpy())

        for key, value in {"y_model": y, **data.to_dict()}.items():

            if not isinstance(value, torch.Tensor):
                continue

            if value.size(0) == pos.size(0):
                point_cloud[key] = value.numpy()

            elif (
                "decoder_query_idcs" in locals()
                and value.size(0) == decoder_query_idcs.numel()
            ):
                dummy = torch.full(
                    (pos.size(0), *value.shape[1:]), value.min(), dtype=value.dtype
                )
                dummy[decoder_query_idcs] = value

                point_cloud[key] = dummy.numpy()

        if "face" in data:
            point_cloud.regular_faces = data.face.T.numpy()
            point_cloud = (
                point_cloud.compute_normals()
            )  # enables smooth rendering in ParaView

        self._create_checkpoints_dir()
        point_cloud.save(os.path.join(self._checkpoints_dir, f"{idx}.vtk"))

        if hasattr(self, "get_custom_pos_visualisation"):
            _pos_ = self.get_custom_pos_visualisation(data)
            if len(_pos_) > 0:
                pos = _pos_[0]
                frame_id = _pos_[1] if len(_pos_) > 1 else None

                point_cloud = pyvista.PolyData(pos.numpy())
                if frame_id is not None:
                    point_cloud["field"] = frame_id.numpy()

                point_cloud.save(
                    os.path.join(self._checkpoints_dir, f"{idx}_custom.vtk")
                )

    def __call__(self, device: torch.device) -> None:
        self.train_model(device)
        self.test_model(device)

    def time_and_test(self, device: torch.device) -> None:
        self.test_model(device)
        self.test_model(device)
