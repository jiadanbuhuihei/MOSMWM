# 该脚本定义dataloader生成函数， 根据指定的dataset和sampler来生成对应的dataloader
import os
import torch
import functools

from .batch_sampler import BatchSampler
from .SPDataset import SPDataset
from .utils import anything_to_seed, collate_segments_to_batch

def build_data_loader(
    dataset,
    batch_size,
    num_workers,
    num_frames,
    seed_data,
    eval,
    rank,
    world_size,
    seed_offset=None,
):
    
    if eval:
        # 还为实现
        pass
    
    else:
        if isinstance(dataset, SPDataset):
            sampler = BatchSampler(
                dataset,
                rank = rank,
                world_size=world_size,
                batch_size=batch_size,
                num_frames=num_frames,
                seed=(
                    [anything_to_seed("sampler"), seed_data] + [seed_offset]
                    if seed_offset is not None
                    else []
                )
            )
            
        else:
            pass
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_segments_to_batch,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        return loader
            
            