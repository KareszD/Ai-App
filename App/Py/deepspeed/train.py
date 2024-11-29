import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader, TensorDataset

# DeepSpeed requires initializing the distributed process group
def init_distributed():
    import os
    import torch.distributed as dist

    if dist.is_initialized():
        return

    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.seq(x)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)

    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from distributed launcher')
    args = parser.parse_args()

    init_distributed()

    # Create a simple model
    model = SimpleModel(input_size=10, hidden_size=10, output_size=1)

    # Dummy dataset
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)

    # DeepSpeed configuration
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Training loop
    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(model_engine.local_rank)
            batch_y = batch_y.to(model_engine.local_rank)

            outputs = model_engine(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)

            model_engine.backward(loss)
            model_engine.step()

        if model_engine.global_rank == 0:
            print(f"Epoch {epoch+1} completed.")

if __name__ == '__main__':
    main()
