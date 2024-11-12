import os
import torch
import torch.distributed as dist
from torch.distributed import all_reduce
from torch.profiler import profile, record_function, ProfilerActivity

from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_setup():
	torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
	dist.init_process_group(backend="nccl")

def ddp_cleanup():
	dist.destroy_process_group()


if __name__ == "__main__":
	ddp_setup()
	rank = int(os.environ["RANK"])
	device = torch.device("cuda:0")
	print("global rank = " + str(rank))
	print("device = " + torch.cuda.get_device_name(device))

	with torch.profiler.profile() as prof:
		tensor = torch.ones(5 * 10**9, device=device)
		dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
	
	print("Size in memory (Mb): " + str(tensor.element_size() * tensor.nelement() / 10**6))

	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

	#tensor = torch.arange(1000, device=device) + 1 + 2 * rank
	#print(tensor)
	#print(tensor)
	ddp_cleanup()
