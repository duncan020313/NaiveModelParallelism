from typing import List
import torch
import torch.nn as nn
from .utils import get_layer_param_count


def get_devices():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def split_model_across_gpus(model: torch.nn.Module, devices: List[torch.device]):
    layer_param_counts = {}
    layers = list(model.children())
    total_params = 0

    for layer in layers:
        layer_param_counts[layer] = get_layer_param_count(layer)
        total_params += layer_param_counts[layer]

    target_params_per_device = total_params / len(devices)

    model_splits = [[] for _ in devices]
    current_params = [0 for _ in devices]
    device_idx = 0
    for layer in layers:
        model_splits[device_idx].append(layer)
        current_params[device_idx] += layer_param_counts[layer]
        if current_params[device_idx] > target_params_per_device:
            device_idx += 1

    return [
        nn.Sequential(*split).to(devices[i]) for i, split in enumerate(model_splits)
    ]


class NaiveModelParallel(nn.Module):
    def __init__(self, model, devices):
        super(NaiveModelParallel, self).__init__()
        self.devices = devices
        self.model_splits = split_model_across_gpus(model, devices)
        self.input_device = devices[0]
        self.output_device = devices[-1]

    def forward(self, x):
        x = x.to(self.input_device)

        for i, model_split in enumerate(self.model_splits):
            x = model_split(x)
            if i < len(self.model_splits) - 1:
                x = x.to(self.devices[i + 1])

        return x
