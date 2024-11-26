import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Model definition
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Simple linear model

    def forward(self, x):
        return self.fc(x)

# Main training function
def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend='gloo',  # Use gloo backend for Windows
        init_method='env://',  # Unique address for communication
        rank=rank,
        world_size=world_size,
    )

    # Set the device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)

    # Create dummy dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Model, loss, and optimizer
    model = DummyModel().to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model with DDP
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):  # Train for 5 epochs
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:  # Log only from rank 0
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Number of GPUs available (set to 1 for this example)
    world_size = 1

    # Windows-specific launcher for multiprocessing
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ['USE_LIBUV'] = '0'


    # Spawn processes
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
