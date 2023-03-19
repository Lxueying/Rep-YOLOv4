#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Lxueying. Some rights reserved.
# Author: Lxueying
# Filename: PaddingKernel.py
# Note:  To pad the value of kernel. Be ready to create the identity branch. If you want to run this code, you would/
#            be better to run fuse_BN.py firstly.


import torch
import torch.nn as nn


def find_module(model):
    children = list(model.named_children())
    if len(children) != 0:
        for name, child in children:
            if isinstance(child, nn.Conv2d) and child.stride == (2,2) and child.weight.shape[2] == 1:
                child.weight.data = nn.Parameter(torch.nn.functional.pad(child.weight.data, [1, 1, 1, 1]))
                child.kernel_size = (3, 3)
            elif isinstance(child, nn.Conv2d) and child.stride == (1,1) and child.weight.shape[2] == 1 and child.in_channels != 64:
                child.weight.data = nn.Parameter(torch.nn.functional.pad(child.weight.data, [2, 2, 2, 2]))
                child.kernel_size = (5, 5)
            elif isinstance(child, nn.Conv2d) and child.stride == (1,1) and child.weight.shape[2] == 3 and child.in_channels != 64:
                child.weight.data = nn.Parameter(torch.nn.functional.pad(child.weight.data, [1, 1, 1, 1]))
                child.kernel_size = (5, 5)
            elif isinstance(child, nn.Conv2d) and child.stride == (1,1) and child.weight.shape[2] == 1 and child.in_channels == 64:
                child.weight.data = nn.Parameter(torch.nn.functional.pad(child.weight.data, [1, 1, 1, 1]))
                child.kernel_size = (3, 3)
            elif isinstance(child, nn.Module):
                find_module(child)

if __name__ == "__main__":

    model_path = "logs/FuseBN.pth"

    model = torch.load(model_path)

    for name, net in model.named_children():
        if name == "backbone":
            find_module(net)
            print(net)

    torch.save(model, "logs/PaddingKernel.pth")
    print("padding with kernel have done!\nPlease check the net!")
