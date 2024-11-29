# ddpbasics.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(
        backend='gloo',  # Use 'nccl' if using GPUs and supported
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

def cleanup():
    dist.destroy_process_group()

def main():
    setup()

    rank = dist.get_rank()
    print(f"Process {rank} initialized successfully.")

    # Test communication
    tensor = torch.ones(1) * rank
    dist.all_reduce(tensor)
    print(f"Process {rank} sees tensor value: {tensor[0]}")

    cleanup()

if __name__ == "__main__":
    # Environment variables are set by torchrun
    # Print to verify environment variables
    print(f"MASTER_ADDR = {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT = {os.environ.get('MASTER_PORT')}")
    print(f"RANK = {os.environ.get('RANK')}")
    print(f"WORLD_SIZE = {os.environ.get('WORLD_SIZE')}")

    main()
