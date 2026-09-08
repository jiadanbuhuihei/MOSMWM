import torch
import numpy as np
import os
import math
import json
import logging

from .segment import SegmentId

class BatchSampler(torch.utils.data.Sampler):
    
    def __init__(
        self,
        dataset,
        rank,
        world_size,
        batch_size,
        num_frames,
        seed=[0],
    ):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()
        
    def __len__(self):
        raise NotImplementedError("BatchSampler does not have a fixed length. Use __iter__ instead.")
    
    def __iter__(self):
        while True:
            yield self.sample()
            
    def reset_rng(self):
        self.rng = np.random.default_rng(self._seed)
        
    def sample(self):
        num_episodes = self.dataset.num_episodes()
        
        # 取得该rank的episodes_partition
        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        
        # 筛掉太短的episodes
        short_episode_ids = np.where(self.dataset._lengths < self.num_frames)[0]
        episodes_partition = episodes_partition[
            ~np.isin(episodes_partition, short_episode_ids)
        ]
        
        # 随机选取 batch_size 个 episodes
        episode_ids = self.rng.choice(
            episodes_partition, size = self.batch_size, replace=True
        )
        starts = self.rng.integers(
            low=0,
            high=self.dataset._lengths[episode_ids] - self.num_frames + 1,
        )
        ends = starts + self.num_frames
        
        return [SegmentId(*x) for x in zip(episode_ids, starts, ends)]
    
    def state_dict(self):
        return {
            "rng_state": self.rng.bit_generator.state,
        }
    
    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng_state"]
    
if __name__ == "__main__":
    from .SPDataset import SPDataset
    dataset = SPDataset("world_model/dataset/train",(1280,704))
    sampler = BatchSampler(dataset, 0, 1, 1, 20)
    for i in range(10):
        print(sampler.sample())