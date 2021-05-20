import torch
import argparse
from typing import Dict, Tuple
from torch import tensor

from .utils import get_one_hot, compute_centroids, extract_features
from .method import FSmethod
from ..metrics import Metric


class ProtoNet(FSmethod):
    """
    Implementation of ProtoNet method https://arxiv.org/abs/1703.05175
    """

    def __init__(self, args: argparse.Namespace):
        self.extract_batch_size = args.extract_batch_size
        self.normamlize = False
        super().__init__(args)

    def record_info(self,
                    metrics: dict,
                    task_ids: tuple,
                    iteration: int,
                    preds_q: tensor,
                    y_q: tensor):
        """
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot] :
        """
        if metrics:
            kwargs = {'gt': y_q, 'preds': preds_q}

            for metric_name in metrics:
                metrics[metric_name].update(task_ids[0],
                                            task_ids[1],
                                            iteration,
                                            **kwargs)

    def forward(self,
                model: torch.nn.Module,
                support: tensor,
                query: tensor,
                y_s: tensor,
                y_q: tensor,
                metrics: Dict[str, Metric] = None,
                task_ids: Tuple[int, int] = None) -> Tuple[tensor, tensor]:
        """
        inputs:
            support : tensor of size [s_shot, c, h, w]
            query : tensor of size [q_shot, c, h, w]
            y_s : tensor of size [s_shot]
            y_q : tensor of size [q_shot]
        """
        num_classes = y_s.unique().size(0)
        with torch.set_grad_enabled(self.training):
            z_s, z_q = extract_features(self.extract_batch_size,
                                        support, query, model)
        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]
        l2_distance = (- 2 * z_q.matmul(centroids.transpose(-2, -1)) \
                        + (centroids**2).sum(-1).unsqueeze(-2)  # noqa: E127
                        + (z_q**2).sum(-1).unsqueeze(-1))  # [batch, q_shot, num_class]

        log_probas = (-l2_distance).log_softmax(-1)  # [batch, q_shot, num_class]
        one_hot_q = get_one_hot(y_q, num_classes)  # [batch, q_shot, num_class]
        ce = - (one_hot_q * log_probas).sum(-1)  # [batch, q_shot, num_class]

        soft_preds_q = log_probas.detach().exp()
        self.record_info(iteration=0,
                         metrics=metrics,
                         task_ids=task_ids,
                         preds_q=soft_preds_q.argmax(-1),
                         y_q=y_q)
        return ce, soft_preds_q