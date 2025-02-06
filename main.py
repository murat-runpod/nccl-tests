import os
import torch
import torch.distributed as dist
from torch.distributed import all_reduce
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import timedelta
import signal
import sys

def handle_signal(signum, frame):
    print(f"Received signal {signum}. Cleaning up...")
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)

def ddp_setup():
    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the device before initializing process group
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with longer timeout
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=60)
    )

def print_gpu_info():
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    print(f"Rank {rank} (local_rank {local_rank}):")
    print(f"  Device: {torch.cuda.get_device_name(device)}")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"  Memory Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        ddp_setup()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.cuda.current_device()
        
        print(f"global rank = {rank}")
        print(f"local rank = {local_rank}")
        print(f"world size = {world_size}")
        print(f"device = {torch.cuda.get_device_name(device)}")
        
        # Start with smaller tensor for testing
        tensor_size = 100 * 1024 * 1024  # 100M elements
        print(f"\nTesting with tensor size: {tensor_size:,} elements")
        
        # First test with small tensor
        tensor = torch.ones(tensor_size, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Print GPU info after first operation
        print_gpu_info()
        
        # If small tensor works, try medium size
        if rank == 0:
            print("\nFirst test successful, trying larger tensor...")
        
        tensor_size = 500 * 1024 * 1024  # 500M elements
        print(f"\nTesting with tensor size: {tensor_size:,} elements")
        
        with torch.profiler.profile() as prof:
            tensor = torch.ones(tensor_size, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Print memory usage
        print(f"Size in memory (GB): {tensor.element_size() * tensor.nelement() / 1e9:.2f}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
    except Exception as e:
        print(f"Rank {rank} encountered error: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()