#!/bin/bash
export NCCL_DEBUG=INFO
#get default interface from 'ip route'
export NCCL_SOCKET_IFNAME=bond0.2636
export NCCL_NSOCS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=32
./.torch/bin/torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=687 \
    --rdzv_backend=static \
    --rdzv_endpoint=213.173.111.94:55000 \
   main.py