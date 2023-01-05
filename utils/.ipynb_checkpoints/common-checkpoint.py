"""
Common utility functions used in training and evaluation processes.
Author: JiaWei Jiang
"""
from decimal import Decimal

import torch.nn as nn


def count_params(model: nn.Module) -> str:
    """Count number of parameters in model.

    Parameters:
        model: model instance

    Return:
        n_params: number of parameters in model, represented in
            scientific notation
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = f"{Decimal(str(n_params)):.4E}"

    return n_params
