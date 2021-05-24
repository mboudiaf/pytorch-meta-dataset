import time
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .method import FSmethod
from ..metrics import Metric
from .utils import get_one_hot, extract_features, compute_centroids


class BDCSPN(FSmethod):

    """
    Implementation of BD-CSPN (ECCV 2020) https://arxiv.org/abs/1911.10713
    """
    def __init__(self,
                 args: argparse.Namespace):
        self.temp = args.temp
        self.iter = 1
        self.episodic_training = False
        self.extract_batch_size = args.extract_batch_size
        if args.val_batch_size > 1:
            raise ValueError("For now, only val_batch_size=1 is support for LaplacianShot")
        super().__init__(args)

    def record_info(self,
                    metrics: Optional[Dict],
                    task_ids: Optional[Tuple],
                    iteration: int,
                    new_time: float,
                    support: Tensor,
                    query: Tensor,
                    y_s: Tensor,
                    y_q: Tensor) -> None:
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

            assert task_ids is not None
            for metric_name in metrics:
                metrics[metric_name].update(task_ids[0],
                                            task_ids[1],
                                            iteration,
                                            **kwargs)

    def rectify(self,
                support: Tensor,
                query: Tensor,
                y_s: Tensor,
                y_q: Tensor) -> None:
        num_classes = y_s.unique().size(0)
        one_hot_s = get_one_hot(y_s, num_classes)
        eta = support.mean(1, keepdim=True) - query.mean(1, keepdim=True)
        query = query + eta

        logits_s = self.get_logits(support).exp()  # [n_task, shot_s, num_class]
        logits_q = self.get_logits(query).exp()  # [n_task, shot_q, num_class]

        preds_q = logits_q.argmax(-1)
        one_hot_q = get_one_hot(preds_q, num_classes)

        normalization = ((one_hot_s * logits_s).sum(1) + (one_hot_q * logits_q).sum(1)).unsqueeze(1)
        w_s = (one_hot_s * logits_s) / normalization  # [n_task, shot_s, num_class]
        w_q = (one_hot_q * logits_q) / normalization  # [n_task, shot_q, num_class]

        # assert np.allclose((torch.cat([w_s, w_q], 1).sum(1) - 1.).sum().item(), 0.),
        #                    (torch.cat([w_s, w_q], 1).sum(1) - 1.).sum()

        self.weights = ((w_s * one_hot_s).transpose(1, 2).matmul(support) \
                        + (w_q * one_hot_q).transpose(1, 2).matmul(query))

    def get_logits(self, samples: Tensor):
        """
        inputs:
            samples : tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : tensor of shape [n_task, shot, num_class]
        """
        # weights = self.weights, 2)
        # samples = samples, 2)
        # logits = self.temp * (samples.matmul(weights.transpose(1, 2)))  #
        cosine = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
        logits = cosine(samples[:, :, None, :], self.weights[:, None, :, :])
        assert logits.max() <= self.temp and logits.min() >= -self.temp, (logits.min(), logits.max())
        return logits

    def forward(self,
                model: torch.nn.Module,
                support: Tensor,
                query: Tensor,
                y_s: Tensor,
                y_q: Tensor,
                metrics: Dict[str, Metric] = None,
                task_ids: Tuple[int, int] = None) -> Tuple[Optional[Tensor], Tensor]:
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot]


        updates :
            self.weights : tensor of shape [n_task, num_class, feature_dim]
        """
        model.eval()

        # Extract support and query features
        with torch.no_grad():
            feat_s, feat_q = extract_features(self.extract_batch_size,
                                              support,
                                              query,
                                              model)

        # Perform required normalizations
        feat_s = F.normalize(feat_s, dim=2)
        feat_q = F.normalize(feat_q, dim=2)

        # Initialize weights
        t0 = time.time()
        self.weights = compute_centroids(feat_s, y_s)  # [batch, num_class, d]
        self.record_info(iteration=0,
                         task_ids=task_ids,
                         metrics=metrics,
                         new_time=time.time() - t0,
                         support=feat_s,
                         query=feat_q,
                         y_s=y_s,
                         y_q=y_q)

        self.rectify(support=feat_s, y_s=y_s,
                     query=feat_q, y_q=y_q)

        self.record_info(iteration=1,
                         task_ids=task_ids,
                         metrics=metrics,
                         new_time=time.time() - t0,
                         support=feat_s,
                         query=feat_q,
                         y_s=y_s,
                         y_q=y_q)
        q_probs = self.get_logits(feat_q)

        return None, q_probs
