#!/bin/bash
export NCCL_DEBUG=INFO
#get default interface from 'ip route'
export NCCL_SOCKET_IFNAME=eth1
export NCCL_NSOCS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4
export NCCL_MIN_NCHANNELS=2
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=687 \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR:55000 \
   main.py