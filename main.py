import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.distributed import all_reduce
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='DDP Communication Benchmark')
    parser.add_argument('--size', type=int, default=5*10**9,
                      help='tensor size for benchmarking')
    parser.add_argument('--iterations', type=int, default=10,
                      help='number of iterations for averaging')
    parser.add_argument('--warmup', type=int, default=3,
                      help='number of warmup iterations')
    parser.add_argument('--profile', action='store_true',
                      help='enable detailed profiling')
    return parser.parse_args()

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

def ddp_cleanup():
    dist.destroy_process_group()

def benchmark_communication(tensor_size, iterations, warmup, enable_profiling=False):
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    
    # Print system info
    if rank == 0:
        print(f"\nBenchmarking Configuration:")
        print(f"- World Size: {world_size}")
        print(f"- Tensor Size: {tensor_size:,} elements")
        print(f"- Total Memory: {tensor_size * 4 / (1024**3):.2f} GB")
        print(f"- Iterations: {iterations}")
        print(f"- Warmup iterations: {warmup}")
        print("\nDevice Information:")
        print(f"- CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"- CUDA Version: {torch.version.cuda}")
        print(f"- NCCL Version: {torch.cuda.nccl.version()}\n")

    # Create tensor for benchmarking
    tensor = torch.ones(tensor_size, device=device)
    
    # Warmup iterations
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark iterations
    latencies = []
    bandwidths = []
    
    if enable_profiling:
        prof = torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=iterations,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/pytorch_prof'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    
    for i in range(iterations):
        start = time.perf_counter()
        
        with record_function("all_reduce"):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        latency = (end - start) * 1000  # convert to ms
        size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
        bandwidth = size_gb / (end - start)  # GB/s
        
        latencies.append(latency)
        bandwidths.append(bandwidth)
        
        if enable_profiling:
            prof.step()
    
    if enable_profiling:
        prof.stop()
        if rank == 0:
            print("\nProfiler Summary:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Calculate statistics
    if rank == 0:
        latencies = np.array(latencies)
        bandwidths = np.array(bandwidths)
        
        print("\nPerformance Statistics:")
		print(f"Size (GB): {size_gb:.2f}")
        print(f"\nLatency (ms):")
        print(f"- Mean: {np.mean(latencies):.2f}")
        print(f"- Std Dev: {np.std(latencies):.2f}")
        print(f"- Min: {np.min(latencies):.2f}")
        print(f"- Max: {np.max(latencies):.2f}")
        print(f"\nBandwidth (GB/s):")
        print(f"- Mean: {np.mean(bandwidths):.2f}")
        print(f"- Std Dev: {np.std(bandwidths):.2f}")
        print(f"- Min: {np.min(bandwidths):.2f}")
        print(f"- Max: {np.max(bandwidths):.2f}")

if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    
    try:
        benchmark_communication(
            tensor_size=args.size,
            iterations=args.iterations,
            warmup=args.warmup,
            enable_profiling=args.profile
        )
    finally:
        ddp_cleanup()