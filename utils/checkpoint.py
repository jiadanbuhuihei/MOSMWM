import os
import torch
import random
import numpy as np

class CheckpointManager:
    def __init__(self, checkpoint_dir, max_keep=5):
        self.dir = checkpoint_dir
        self.max_keep = max_keep
        
        os.makedirs(self.dir, exist_ok=True)
        
    def save(self, model, optimizer, scheduler, train_loader, step, config):
        checkpoint = {
            "model": model.state_dict()if model is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step if step is not None else None,
            "config": config if config is not None else None,
            "train_loader_state": train_loader.batch_sampler.state_dict() if train_loader is not None else None,
            "rng":{
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
            }
        }
        torch.save(checkpoint, os.path.join(self.dir, f"checkpoint_{step}.pt"))
        
    def load(self, path, model, optimizer=None, scheduler=None, train_loader=None):
        path = os.path.join(self.dir, path)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if optimizer is not None and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if train_loader is not None and checkpoint["train_loader_state"] is not None:
            train_loader.batch_sampler.load_state_dict(checkpoint["train_loader_state"])
        
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
        
        return checkpoint["step"]
