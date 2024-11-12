import torch 
from torch.profiler import profile, record_function, ProfilerActivity


def vecSum(device):
    a = torch.ones(10**9, device=device)
    b = torch.ones(10**9, device=device)
    print("Vector size (Mb): " + str(a.element_size() * a.nelement() / 10**6))
    c = a + b
    a = None
    b = None
    c = None

def devProfile(device):
    print("device = " + torch.cuda.get_device_name(device))
    vecSum(device) #warmup
    with torch.profiler.profile() as prof:
        vecSum(device)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if __name__ == "__main__":

    dev0 = torch.device("cuda:0")
    devProfile(dev0)

    dev1 = torch.device("cuda:1")
    devProfile(dev1)

