#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Huansu. Some rights reserved.
# Author: Huansu
# Filename: CreateBranchForIdentity.py
# Note: To create the identity branch and fuse this layer with BN layer.\
#           Please run this code after PaddingKernel.py.


import torch
import torch.nn as nn
import numpy as np

def find_module(model):
    global in_channels, groups, out_channels
    children = list(model.named_children())
    if len(children) != 0:
        for name, child in children:
            if isinstance(child, nn.Conv2d):
                in_channels = child.in_channels
                groups = child.groups
                out_channels = child.out_channels
            if isinstance(child, nn.BatchNorm2d):
                input_dim = in_channels // groups
                kernel_value = np.zeros((in_channels, input_dim, 5, 5), dtype=np.float32)
                for i in range(in_channels):
                    kernel_value[i, i % input_dim, 2, 2] = 1
                tensor = torch.from_numpy(kernel_value).to(child.weight.device)
                kernel = tensor
                running_mean = child.running_mean
                running_var = child.running_var
                gamma = child.weight
                beta = child.bias
                eps = child.eps
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                weight, bias = kernel * t, beta - running_mean * gamma / std
                print(weight.shape, bias.shape)

                module = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=groups)
                module.weight.data = nn.Parameter(weight)
                module.bias.data = nn.Parameter(bias)
                # 添加新的Branch，但是默认该模块最后，暂时没找到指定插入顺序的办法
                model.add_module('IdConv5x5', nn.Sequential(module))
                del model._modules[name]

                # 不明白顺序影不影响推理，所以现矫正顺序
                del model._modules['activation']
                act = nn.LeakyReLU()
                model.add_module("activation", act)

            elif isinstance(child, nn.Module):
                find_module(child)

if __name__ == "__main__":

    model_path = "logs/PaddingKernel.pth"
    # print("start to run the code")

    model = torch.load(model_path)

    for name, net in model.named_children():
        if name == "backbone":
            find_module(net)
            print(net)

    torch.save(model, "logs/CreateIdentity.pth")
    print("done!\nPlease check the net!")



