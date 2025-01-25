import os
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
    return parser.parse_args()

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

def ddp_cleanup():
    dist.destroy_process_group()

def benchmark_communication(tensor_size, iterations, warmup):
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    
    # Print system info
    if rank == 0:
        print(f"\nBenchmarking Configuration:")
        print(f"- World Size: {world_size}")
        print(f"- Tensor Size: {tensor_size:,} elements")
        print(f"- Total Memory: {tensor_size * 4 / (1024**3):.2f} GB")
        print(f"- Total Size (GB): {tensor_size * 4 / (1024**3):.2f} GB")
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
    
    # Synchronize before profiling
    torch.cuda.synchronize()
    dist.barrier()
    
    # Profile iterations
    with profile(
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
    ) as prof:
        
        for i in range(iterations):
            with record_function("all_reduce"):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
            prof.step()
    
    # Print profiling results
    if rank == 0:
        print("\nProfiler Summary:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Extract timing information from profiler
        all_reduce_events = [event for event in prof.key_averages() if "all_reduce" in event.key]
        if not all_reduce_events:
            print("Warning: No all_reduce events found in profiler output")
            return
            
        # Verify we have the required attributes
        if not hasattr(all_reduce_events[0], 'cuda_time'):
            print("Available event attributes:", dir(all_reduce_events[0]))
            raise AttributeError("Event object doesn't have cuda_time attribute")
            
        if all_reduce_events:
            cuda_times = [event.cuda_time / 1000 for event in all_reduce_events]  # Convert to ms
            size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
            bandwidths = [size_gb / (t / 1000) for t in cuda_times]  # Convert time to seconds for GB/s
            
            print("\nPerformance Statistics:")
            print(f"Latency (ms):")
            print(f"- Mean: {np.mean(cuda_times):.2f}")
            print(f"- Std Dev: {np.std(cuda_times):.2f}")
            print(f"- Min: {np.min(cuda_times):.2f}")
            print(f"- Max: {np.max(cuda_times):.2f}")
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
            warmup=args.warmup
        )
    finally:
        ddp_cleanup()