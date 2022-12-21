# RPQ-pytorch
Reverse Product Quantization (RPQ) of weights to reduce static memory usage.


<img src="./assets/rpq_diagram.gif" width="1280px"></img>

<!-- Go into how the method works. -->

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This repository contains an implementation of the Reverse Product Quantization (RPQ) method, which is a method for reducing the memory footprint of deep neural networks.

This method is an extension of the Product Quantization (PQ) method, which was introduced by [Jegou et al. (2012)](https://arxiv.org/abs/1206.4136) and [Jegou et al. (2013)](https://arxiv.org/abs/1308.1492). This method is a method for reducing the memory footprint of deep neural networks by reducing the number of bits used to store the weights of the network. The method works by quantizing the weights of the network into a smaller number of bits, and then storing the quantized weights in a lookup table. When the network is run, the weights are loaded from the lookup table, rather than the original weight matrix.

The original PQ method works by quantizing the weights of a network into a smaller number of bits, and then storing the quantized weights in a lookup table. When the network is run, the weights are loaded from the lookup table, rather than the original weight matrix. The method works by quantizing the weights of the network into a smaller number of bits, and then storing the quantized weights in a lookup table. When the network is run, the weights are loaded from the lookup table, rather than the original weight matrix.

## Installation

```bash
pip install rpq-pytorch
```

## Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from rpq import RPQ

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = Net()
net = RPQ(net)

x = torch.randn(1, 1, 28, 28)
y = net(x)
```


