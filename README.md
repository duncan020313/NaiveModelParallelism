# Naive Model Parallel

Naive Model Parallel is a Python library for distributing PyTorch model layers across multiple GPUs. It balances the layers based on the number of parameters, supporting both sequential and tree-like architectures.

## Features

- **Easy to use**: Distribute any PyTorch `torch.nn.Module` model across multiple GPUs with minimal code changes.
- **Parameter-based balancing**: Layers are distributed based on the number of parameters to balance the load across GPUs.
- **Support for complex architectures**: Handles both sequential and tree-like model architectures.

## Installation

You can install the library directly from GitHub:

```bash
pip install git+https://github.com/yourusername/naive_model_parallel.git
