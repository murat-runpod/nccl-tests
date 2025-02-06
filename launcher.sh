#!/bin/bash
torchrun \
    --nproc_per_node=$NUM_TRAINERS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
   main.py

   