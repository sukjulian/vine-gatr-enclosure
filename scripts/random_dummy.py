# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import hydra
import torch
from omegaconf import DictConfig

from gatr_enclosure.experiments import RandomDummyExperiment


@hydra.main(version_base=None, config_path="../config", config_name="random_dummy")
def main(config: DictConfig) -> None:

    experiment = RandomDummyExperiment(config)
    experiment(device=torch.device("cuda"))


if __name__ == "__main__":
    main()
