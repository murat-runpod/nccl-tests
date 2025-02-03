#!/bin/bash
torchrun \
    --nproc_per_node=$NUM_TRAINERS \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=55000 \
   main.py