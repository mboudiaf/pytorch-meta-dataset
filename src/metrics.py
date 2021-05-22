from typing import List

import torch
from torch import Tensor


def ECE_(soft_preds: Tensor,
         targets: Tensor,
         reduc_dims: List[int]) -> Tensor:
    '''
    args:

    soft_preds: tensor of shape [*, K]
    targets: tensor of shape [*,]
    '''
    device = soft_preds.device

    highest_prob, hard_preds = soft_preds.max(-1)
    bins = torch.linspace(0, 1, 10).to(device)
    binned_indexes = vectorized_binning(highest_prob, bins)  # one-hot [*, M]

    accurate = (hard_preds == targets).float().unsqueeze(-1)

    B = binned_indexes.sum(reduc_dims) + 1e-10  # [M,]
    A = (accurate * binned_indexes).sum(reduc_dims) / B  # [M,]
    C = (highest_prob.unsqueeze(-1) * binned_indexes).sum(reduc_dims) / B  # [M,]
    N = torch.tensor(targets.size()).prod()
    ECE = ((B * torch.abs(A - C))).sum(-1, keepdim=True) / N

    return ECE


def vectorized_binning(probs: Tensor,
                       bins: Tensor) -> Tensor:
    '''
    args:

    probs: tensor of shape [*,]
    bins: tensor of shape [M]
    '''
    batch_shape = probs.shape
    new_bin_shape = [1 for _ in range(len(batch_shape))]
    new_bin_shape.append(-1)

    bins = bins.view(*new_bin_shape)  # [*, M]
    probs = probs.unsqueeze(-1)  # [*, 1]
    bins_indexes = (probs >= bins).long().sum(-1, keepdim=True)  # [*,]

    ones_hot_indexes = torch.zeros(batch_shape + bins.shape[-1:]).to(probs.device)  # [*,]
    ones_hot_indexes.scatter_(-1, bins_indexes, 1.)  #

    return ones_hot_indexes
