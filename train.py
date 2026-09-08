from re import S
import logging
import sys
import os
import hydra
from omegaconf import OmegaConf
import torch
from .training.SPTrainer import SPTrainer
from .utils.handler import training_signal

trainer_map = {
    "SPTrainer": SPTrainer,
}

@hydra.main(config_path="./configs", config_name="train", version_base=None)
def main(cfg):

    default_config = cfg["default_config"]

    trainer = trainer_map[cfg["trainer"].trainer](
        base_seed = 42,
        default_config = default_config,
        dataset_config = cfg["dataset"],
        optimizer_config = cfg["optimizer"],
        scheduler_config = cfg["scheduler"],
        training_config = cfg["trainer"],
        checkpoint_dir = default_config.checkpoint_dir,
        cfg=cfg,
    )
    
    trainer.run(training_signal)
    
if __name__ == "__main__":
    main()