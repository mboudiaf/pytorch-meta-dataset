import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import math


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor):
    """
    inputs:
        z_s : torch.Tensor of size [*, s_shot, d]
        y_s : torch.Tensor of size [*, s_shot]
    updates :
        centroids : torch.Tensor of size [*, num_class, d]
    """

    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(-2, -1)  # [*, K, s_shot]
    centroids = one_hot.matmul(z_s) / one_hot.sum(-1, keepdim=True)  # [*, K, d]
    return centroids


def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [*]
        returns
            one_hot : torch.Tensor of shape [*, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot


def extract_features(bs: int,
                     support: torch.tensor,
                     query: torch.tensor,
                     model: nn.Module):
    """
    Extract features from support and query set using the provided model
        args:
            x_s : torch.Tensor of size [batch, s_shot, c, h, w]
        returns
            z_s : torch.Tensor of shape [batch, s_shot, d]
            z_s : torch.Tensor of shape [batch, q_shot, d]
    """
    # Extract support and query features
    n_tasks, shots_s, C, H, W = support.size()
    shots_q = query.size(1)
    device = dist.get_rank()
    if bs > 0:
        if n_tasks > 1:
            raise ValueError("Multi task and feature batching not yet supported")
        feat_s = batch_feature_extract(model, support, bs, device)
        feat_q = batch_feature_extract(model, query, bs, device)
    else:
        support = support.to(device)
        query = query.to(device)
        feat_s = model(support.view(n_tasks * shots_s, C, H, W), feature=True)
        feat_q = model(query.view(n_tasks * shots_q, C, H, W), feature=True)
        feat_s = feat_s.view(n_tasks, shots_s, -1)
        feat_q = feat_q.view(n_tasks, shots_q, -1)
    return feat_s, feat_q


def batch_feature_extract(model: nn.Module,
                          t: torch.tensor,
                          bs: int,
                          device: torch.device):
    n_tasks, shots, C, H, W = t.size()
    feats = []
    for i in range(math.ceil(shots / bs)):
        start = i * bs
        end = min(shots, (i+1) * bs)
        x = t[0, start:end, ...]
        x = x.to(device)
        feat = model(x, feature=True)
        feats.append(feat)
    feats = torch.cat(feats, 0).unsqueeze(0)
    return feats