import argparse
import torch
import time
from typing import Dict, Tuple
from torch import tensor

from .utils import extract_features, compute_centroids
from .method import FSmethod
from ..metrics import Metric


class SimpleShot(FSmethod):
    """
    Implementation of SimpleShot method https://arxiv.org/abs/1911.04623
    """
    def __init__(self,
                 args: argparse.Namespace):
        self.iter = args.iter
        self.episodic_training = False
        self.extract_batch_size = args.extract_batch_size
        super().__init__(args)

    def get_logits(self, samples: tensor) -> tensor:
        """
        inputs:
            samples : tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = (samples.matmul(self.weights.transpose(1, 2)) \
                  - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                  - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples: tensor) -> tensor:
        """
        inputs:
            samples : tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds

    def record_info(self,
                    metrics: dict,
                    task_ids: tuple,
                    iteration: int,
                    new_time: float,
                    support: tensor,
                    query: tensor,
                    y_s: tensor,
                    y_q: tensor) -> tensor:
        """
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot] :
        """
        if metrics:
            logits_s = self.get_logits(support).detach()
            probs_s = logits_s.softmax(-1)
            logits_q = self.get_logits(query).detach()
            preds_q = logits_q.argmax(2)
            probs_q = logits_q.softmax(2)
            kwargs = {'probs': probs_q, 'probs_s': probs_s, 'preds': preds_q,
                      'gt': y_q, 'z_s': support, 'z_q': query, 'gt_s': y_s,
                      'weights': self.weights}

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

        model.eval()
        with torch.no_grad():
            feat_s, feat_q = extract_features(self.extract_batch_size,
                                              support, query, model)

        # Initialize weights
        t0 = time.time()
        self.weights = compute_centroids(feat_s, y_s)
        self.record_info(iteration=0,
                         metrics=metrics,
                         task_ids=task_ids,
                         new_time=time.time()-t0,
                         support=feat_s,
                         query=feat_q,
                         y_s=y_s,
                         y_q=y_q)

        P_q = self.get_logits(feat_q).softmax(2)
        return None, P_q