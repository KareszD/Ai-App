import os
import torch
import horovod.torch as hvd

def setup():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

def main():
    setup()
    rank = hvd.rank()
    size = hvd.size()
    local_rank = hvd.local_rank()
    
    print(f"Process {rank} (Local rank {local_rank}) of {size} initialized successfully on {torch.cuda.get_device_name()}.")

    # Test communication
    tensor = torch.ones(1).cuda() * rank
    tensor = hvd.allreduce(tensor, name="test_allreduce")
    print(f"Process {rank} sees tensor value: {tensor[0].item()}.")

if __name__ == "__main__":
    
    os.environ[ "HOROVOD_HOSTS"]="pc1:1,pc2:1"
    os.environ[ "HOROVOD_CONTROLLER"]="gloo"

    main()
