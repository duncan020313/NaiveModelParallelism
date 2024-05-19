import torch.nn as nn


def get_layer_param_count(layer):
    return sum(p.numel() for p in layer.parameters())


def get_children(model):
    children = list(model.children())
    flat_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flat_children.extend(get_children(child))
            except TypeError:
                flat_children.append(get_children(child))
    return flat_children


def move_module_to_device(module, device):
    for name, param in module.named_parameters(recurse=False):
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(device)
    for name, buf in module.named_buffers(recurse=False):
        buf.data = buf.data.to(device)
    module.to(device)
