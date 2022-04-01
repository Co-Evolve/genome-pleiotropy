# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch
import torch.nn.functional as F


def bss(x):
    # Bipolar Steepened Sigmoid
    return 2 * torch.sigmoid(5 * x) - 1


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def abs_activation(x):
    return torch.abs(x)


def nabs_activation(x):
    return torch.neg(torch.abs(x))


def square_activation(x):
    return torch.pow(x, 2)


def nsquare_activation(x):
    return torch.neg(square_activation(x))


def sqrt_activation(x):
    return torch.sign(x) * torch.sqrt(abs_activation(x))


def nsqrt_activation(x):
    return torch.neg(sqrt_activation(x))


def gauss_activation(x):
    return torch.exp(-5.0 * x ** 2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def cos_activation(x):
    return torch.cos(x)


def relu_activation(x):
    return F.relu(x)


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'nabs': nabs_activation,
    'sqrt': sqrt_activation,
    'nsqrt': nsqrt_activation,
    'square': square_activation,
    'nsquare': nsquare_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
    'bss': bss,
    'cos': cos_activation
}
