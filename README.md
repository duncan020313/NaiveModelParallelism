# Naive Model Parallel

Naive Model Parallel is a Python library for distributing PyTorch model layers across multiple GPUs. It balances the layers based on the number of parameters.

## Features

- **Easy to use**: Distribute any PyTorch `torch.nn.Module` model across multiple GPUs with minimal code changes.
- **Parameter-based balancing**: Layers are distributed based on the number of parameters to balance the load across GPUs.

## Installation

You can install the library directly from GitHub:

```bash
pip install git+https://github.com/yourusername/naive_model_parallel.git
```

## Warning

This project doesn't consider the efficiency of model parallelism. I'll add such optimization later

