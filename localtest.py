import torch 
from torch.profiler import profile, record_function, ProfilerActivity


def vecSum(device):
    with torch.profiler.profile() as prof:
        a = torch.ones(10**9, device=device)
        b = torch.ones(10**9, device=device)
        c = a + b
    print("Vector size (Mb): " + str(a.element_size() * a.nelement() / 10**6))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    a = None
    b = None



if __name__ == "__main__":

    dev0 = torch.device("cuda:0")
    print("device = " + torch.cuda.get_device_name(dev0))
    vecSum(dev0)

    dev1 = torch.device("cuda:1")
    print("device = " + torch.cuda.get_device_name(dev1))
    devSum(dev1)

