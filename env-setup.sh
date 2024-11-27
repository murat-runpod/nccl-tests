#!/bin/bash

apt install python3 -y
apt install python3-venv -y
apt install python3-pip -y

python3 -m venv .torch
./.torch/bin/activate \
    python3 -m pip install torch torchvision torchaudio

