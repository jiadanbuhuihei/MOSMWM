import hashlib
import numpy as np
import torch
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class Batch:
    obs: np.ndarray
    act: np.ndarray
    viewmats: np.ndarray
    K: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return{
            "obs": self.obs,
            "act": self.act,
            "viewmats": self.viewmats,
            "K": self.K,
        }
        
    def to_tensor(self, device):
        return {
            "obs": torch.from_numpy(self.obs)
                    .to(device),

            "act": torch.from_numpy(self.act)
                    .to(device),

            "viewmats": torch.from_numpy(self.viewmats)
                    .to(device),

            "K": torch.from_numpy(self.K)
                    .to(device),
        }
        

def anything_to_seed(*args):
    serialized_args = []
    for arg in args:
        if isinstance(arg, int):
            type_code = "int"
            value_repr = str(arg)
        elif isinstance(arg, float):
            type_code = "float"
            value_repr = repr(arg)
        elif isinstance(arg, bool):
            type_code = "bool"
            value_repr = str(arg)
        elif isinstance(arg, str):
            type_code = "str"
            value_repr = repr(arg)
        else:
            raise TypeError(f"Unsupported type: {type(arg).__name__}")
        serialized_arg = f"{type_code}:{value_repr}"
        serialized_args.append(serialized_arg)

    serialized_str = "|".join(serialized_args)
    serialized_bytes = serialized_str.encode("utf-8")
    hash_bytes = hashlib.sha256(serialized_bytes).digest()
    seed_int = int.from_bytes(hash_bytes, "big")
    return seed_int % (1 << 64)


def collate_segments_to_batch(segments):
    obs = np.stack([s.obs for s in segments])
    act = np.stack([s.act for s in segments])
    viewmats = np.stack([s.viewmats for s in segments])
    K = segments[0].K
    
    return Batch(obs, act, viewmats, K)