#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Huansu. Some rights reserved.
# Author: Huansu
# Filename: TestLoad.py
# Note: Test the inference speed of the model in the GPU, \
#            please make sure the net use FuseBranchBackbone.py as its Backbone

import torch
import numpy as np
from nets.ImprovedNeckYolo import YoloBody

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_path = "FuseBranch.pth"

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 1

net = YoloBody(anchors_mask, num_classes).to(device)
net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
net.eval()

x = torch.randn(1, 3, 416, 416).to(device)


starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

t = []
for iteration in range(10000):
    with torch.no_grad():
        starter.record()
        out = net(x)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
    t.append(curr_time)
del t

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

t = []
for iteration in range(100):
    with torch.no_grad():
        starter.record()
        out = net(x)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
    t.append(curr_time)

print('timings mean:%s ms' % np.mean(t))

#

