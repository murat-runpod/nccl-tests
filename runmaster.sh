#!/bin/bash
export NCC_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0.2411
./torch/bin/torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=687 \
    --rdzv_backend=static \
    --rdzv_endpoint=213.173.102.185:55000 \
   main.py