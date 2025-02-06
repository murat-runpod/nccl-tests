import os
import torch
import torch.distributed as dist
from torch.distributed import all_reduce
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # Set device before init_process_group
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(seconds=10))

def ddp_cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    ddp_setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")  # Use local_rank for device assignment
    
    print(f"global rank = {rank}")
    print(f"local rank = {local_rank}")
    print(f"device = {torch.cuda.get_device_name(device)}")

    tensor = torch.ones(1 * 10**9, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    with torch.profiler.profile() as prof:
        tensor = torch.ones(5 * 10**9, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Size in memory (Mb): {tensor.element_size() * tensor.nelement() / 10**6}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    ddp_cleanup()