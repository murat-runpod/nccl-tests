import os
import torch
import torch.distributed as dist
from torch.distributed import all_reduce

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
	tensor = torch.arange(2, device=device) + 1 + 2 * rank
	print(tensor)
	dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
	print(tensor)
	ddp_cleanup()
