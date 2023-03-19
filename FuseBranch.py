#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Lxueying. Some rights reserved.
# Author: Lxueying
# Filename: FuseBranch.py
# Note: To fuse the branches with different kenerl size and delete the redundant layers.


import torch
import torch.nn as nn


def find_module(model, stride2_weight, stride2_bias):
    global in_channels, out_channels, groups, kernel_size, stride
    stride2_weight, stride2_bias = stride2_weight, stride2_bias

    children = list(model.named_children())

    if len(children) != 0:
        for name, child in children:
            if isinstance(child, nn.Conv2d) and child.stride == (2,2):
                in_channels = child.in_channels
                groups = child.groups
                out_channels = child.out_channels
                stride = child.stride
                kernel_size = child.kernel_size
                if (stride2_weight != None and stride2_weight != "False") and (stride2_bias != None and stride2_bias != "False"):
                    stride2_weight += child.weight.data
                    stride2_bias += child.bias.data
                else:
                    stride2_weight = child.weight.data
                    stride2_bias = child.bias.data
                del model._modules[name]

                return stride2_weight, stride2_bias

            elif isinstance(child, nn.Conv2d) and child.stride == (1,1):
                in_channels = child.in_channels
                groups = child.groups
                out_channels = child.out_channels
                stride = child.stride
                kernel_size = child.kernel_size
                if (stride2_weight != None and stride2_weight != "False") and (stride2_bias != None and stride2_bias != "False"):
                    stride2_weight += child.weight.data
                    stride2_bias += child.bias.data
                else:
                    stride2_weight = child.weight.data
                    stride2_bias = child.bias.data
                del model._modules[name]

                return stride2_weight, stride2_bias

            elif isinstance(child, nn.LeakyReLU):
                if stride == (2, 2):
                    module = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=groups)
                    module.weight.data = nn.Parameter(stride2_weight)
                    module.bias.data = nn.Parameter(stride2_bias)
                    # 添加新的Branch，但是默认该模块最后，暂时没找到指定插入顺序的办法
                    model.add_module('Conv{}'.format(kernel_size[0]), module)

                    del model._modules[name]
                    # 不明白顺序影不影响推理，所以现矫正顺序
                    act = nn.LeakyReLU()
                    model.add_module("activation", act)

                    stride2_weight = None
                    stride2_bias = None

                    return stride2_weight, stride2_bias

                elif stride == (1, 1):
                    module = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1,
                                       groups=groups)
                    module.weight.data = nn.Parameter(stride2_weight)
                    module.bias.data = nn.Parameter(stride2_bias)
                    # 添加新的Branch，但是默认该模块最后，暂时没找到指定插入顺序的办法
                    model.add_module('Conv{}'.format(kernel_size[0]), module)

                    del model._modules[name]
                    # 不明白顺序影不影响推理，所以现矫正顺序
                    act = nn.LeakyReLU()
                    model.add_module("activation", act)

                    stride2_weight = None
                    stride2_bias = None

                    return stride2_weight, stride2_bias


            elif isinstance(child, nn.Module):
                stride2_weight, stride2_bias = find_module(child, stride2_weight, stride2_bias)
                if stride2_bias == None:
                    # 回退的bias一直不会等于零
                    stride2_weight="False“"
                    stride2_bias="False"
                if stride2_bias == "False" and child == children[-1][1]:
                    print(children)
                    return stride2_weight, stride2_bias
                else:
                    pass


if __name__ == "__main__":

    model_path = "logs/CreateIdentity.pth"

    model = torch.load(model_path)

    for name, net in model.named_children():
        if name == "backbone":
            find_module(net, stride2_weight=None, stride2_bias=None)
            print(net)

    save_path = "logs/FuseBranch.pth"
    torch.save(model.state_dict(),  save_path)

    print("done!\nPlease check the net!")


