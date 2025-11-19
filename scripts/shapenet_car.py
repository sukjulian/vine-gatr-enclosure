# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import hydra
import torch
from omegaconf import DictConfig

from gatr_enclosure.experiments import ShapenetCarExperiment


@hydra.main(version_base=None, config_path="../config", config_name="shapenet_car")
def main(config: DictConfig) -> None:

    experiment = ShapenetCarExperiment(config)
    experiment(device=torch.device("cuda"))
    # experiment.time_and_test(device=torch.device("cuda"))


if __name__ == "__main__":
    main()
