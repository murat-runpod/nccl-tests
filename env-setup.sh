#!/bin/bash

apt install python3 -y
apt install python3-venv -y
apt install python3-pip -y

python3 -m venv .torch
source ./.torch/bin/activate

