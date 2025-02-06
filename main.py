import os
import torch
import torch.distributed as dist
import time
from datetime import timedelta

def format_size(size_bytes):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def print_gpu_info():
    """Print GPU information for current process"""
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    
    print(f"[Rank {rank} | Local Rank {local_rank}]")
    print(f"  Device: {torch.cuda.get_device_name(device)}")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"  Memory Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

def benchmark_allreduce(size_mb=100, warmup=2, iters=5):
    """
    Benchmark all-reduce operation
    Args:
        size_mb: Size of tensor in MB
        warmup: Number of warmup iterations
        iters: Number of timed iterations
    """
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    
    # Calculate tensor size and create tensor
    size_bytes = size_mb * 1024 * 1024
    n_elements = size_bytes // 4  # 4 bytes per float32
    device = torch.device(f"cuda:{local_rank}")
    
    # Warmup iterations
    tensor = torch.ones(n_elements, dtype=torch.float32, device=device)
    for _ in range(warmup):
        dist.all_reduce(tensor)
    
    torch.cuda.synchronize()
    
    # Time iterations
    times = []
    tensor = torch.ones(n_elements, dtype=torch.float32, device=device)
    
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        
        dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        
        # Only print from rank 0
        if rank == 0:
            tensor_size = tensor.element_size() * tensor.nelement()
            # For all_reduce: 2*(n-1)/n multiplier for n GPUs
            effective_size = 2 * (world_size - 1) / world_size * tensor_size
            bandwidth = effective_size / times[-1]
            
            print(f"\nIteration {i+1}:")
            print(f"  Tensor size: {format_size(tensor_size)}")
            print(f"  Time: {times[-1]*1000:.2f} ms")
            print(f"  Bandwidth: {bandwidth/1e9:.2f} GB/s")
    
    if rank == 0:
        avg_time = sum(times) / len(times)
        tensor_size = tensor.element_size() * tensor.nelement()
        effective_size = 2 * (world_size - 1) / world_size * tensor_size
        avg_bandwidth = effective_size / avg_time
        
        print(f"\nAverage over {iters} iterations:")
        print(f"  Time: {avg_time*1000:.2f} ms")
        print(f"  Bandwidth: {avg_bandwidth/1e9:.2f} GB/s")

def main():
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=60)
    )
    
    # Set device for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Print basic info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"\nStarting benchmark with {world_size} GPUs")
    
    # Print GPU info
    print_gpu_info()
    
    # Barrier to ensure clean output
    dist.barrier()
    
    # Test different sizes
    sizes = [10, 100, 1000]  # MB
    
    for size in sizes:
        if rank == 0:
            print(f"\nTesting size: {size} MB")
        dist.barrier()  # Synchronize before each test
        benchmark_allreduce(size_mb=size)
        dist.barrier()  # Synchronize after each test
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()