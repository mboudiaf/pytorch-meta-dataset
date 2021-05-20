import argparse
import torch
from typing import Dict, Tuple
import torch.distributed as dist
import torch.nn.functional as F
from torch import tensor

from .utils import get_one_hot, extract_features
from .method import FSmethod
from ..metrics import Metric
from tqdm import tqdm


class Finetune(FSmethod):

    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    """
    def __init__(self,
                 args: argparse.Namespace):
        self.temp = args.temp
        self.iter = args.iter
        self.extract_batch_size = args.extract_batch_size
        self.finetune_all_layers = args.finetune_all_layers
        self.episodic_training = False
        self.lr = args.finetune_lr
        super().__init__(args)

    def record_info(self,
                    metrics: dict,
                    task_ids: tuple,
                    iteration: int,
                    preds_q: tensor,
                    probs_s: tensor,
                    y_q: tensor,
                    y_s: tensor) -> None:
        """
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot] :
        """
        if metrics:
            kwargs = {'preds': preds_q, 'gt': y_q, 'probs_s': probs_s,
                      'gt_s': y_s}

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
        Corresponds to the TIM-GD inference
        inputs:
            support : tensor of shape [n_task, s_shot, feature_dim]
            query : tensor of shape [n_task, q_shot, feature_dim]
            y_s : tensor of shape [n_task, s_shot]
            y_q : tensor of shape [n_task, q_shot]


        updates :
            self.weights : tensor of shape [n_task, num_class, feature_dim]
        """
        device = dist.get_rank()
        model.eval()
        n_tasks = support.size(0)
        if n_tasks > 1:
            raise ValueError('Finetune method can only deal with 1 task at a time. \
                             Currently {} tasks.'.format(n_tasks))
        y_s = y_s[0]
        y_q = y_q[0]
        num_classes = y_s.unique().size(0)
        y_s_one_hot = get_one_hot(y_s, num_classes)

        # Initialize classifier
        with torch.no_grad():
            feat_s, feat_q = extract_features(self.extract_batch_size,
                                              support,
                                              query,
                                              model)

            classifier = torch.nn.Linear(feat_s.size(-1), num_classes).to(device)
            preds_q = classifier(feat_q[0]).argmax(-1)
            probs_s = classifier(feat_s[0]).softmax(-1)
            self.record_info(iteration=0,
                             task_ids=task_ids,
                             metrics=metrics,
                             preds_q=preds_q,
                             probs_s=probs_s,
                             y_q=y_q,
                             y_s=y_s)

        # Define optimizer
        if self.finetune_all_layers:
            params = list(model.parameters()) + list(classifier.parameters())
        else:
            params = classifier.parameters()  # noqa: E127
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Run adaptation
        with torch.set_grad_enabled(self.finetune_all_layers):
            feat_s, feat_q = extract_features(self.extract_batch_size,
                                              support,
                                              query,
                                              model)
            feat_s = F.normalize(feat_s, dim=-1)
            feat_q = F.normalize(feat_q, dim=-1)

        for i in range(1, self.iter):
            probs_s = classifier(feat_s[0]).softmax(-1)
            loss = - (y_s_one_hot * probs_s.log()).sum(-1).mean(-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds_q = classifier(feat_q[0]).argmax(-1)
                self.record_info(iteration=i,
                                 task_ids=task_ids,
                                 metrics=metrics,
                                 preds_q=preds_q,
                                 probs_s=probs_s,
                                 y_q=y_q,
                                 y_s=y_s)

        probs_q = classifier(feat_q[0]).softmax(-1).unsqueeze(0)
        return loss.detach(), probs_q.detach()
