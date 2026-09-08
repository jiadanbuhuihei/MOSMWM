import os
import torch
import torch.distributed as dist


def init_distributed():
    
    if "LOCAL_RANK" not in os.environ:
        # 单卡模式
        local_rank = 0
        rank = 0
        world_size = 1

        return local_rank, rank, world_size
    
    # DDP模式   
    local_rank = int(
        os.environ["LOCAL_RANK"],
        init_method="env://"
    )

    torch.cuda.set_device(local_rank)


    dist.init_process_group(
        backend="nccl"
    )


    rank = dist.get_rank()

    world_size = dist.get_world_size()


    return local_rank, rank, world_size