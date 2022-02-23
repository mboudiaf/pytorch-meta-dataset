import argparse
from typing import Dict, Optional, Tuple
from loguru import logger
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn

from .method import FSmethod
from ..metrics import Metric
from .utils import get_one_hot, extract_features


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
        self.cosine_head = args.finetune_cosine_head
        self.weight_norm = args.finetune_weight_norm
        self.lr = args.finetune_lr

        self.episodic_training = False

        super().__init__(args)

    def record_info(self,
                    metrics: Optional[Dict],
                    task_ids: Optional[Tuple],
                    iteration: int,
                    preds_q: Tensor,
                    probs_s: Tensor,
                    y_q: Tensor,
                    y_s: Tensor) -> None:
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

            assert task_ids is not None
            for metric_name in metrics:
                metrics[metric_name].update(task_ids[0],
                                            task_ids[1],
                                            iteration,
                                            **kwargs)

    def _do_data_dependent_init(self, classifier: nn.Module, feat_s: Tensor):
        """Returns ops for the data-dependent init of g and maybe b_fc."""
        w_fc_normalized = F.normalize(classifier.weight_v, dim=1)  # [num_classes, d]
        output_init = feat_s @ w_fc_normalized.t()  # [n_s, num_classes]
        var_init = output_init.var(0, keepdim=True)  # [num_classes]
        # Data-dependent init values.
        classifier.weight_g.data = 1. / torch.sqrt(var_init + 1e-10)

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
        device = dist.get_rank()
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

            classifier = nn.Linear(feat_s.size(-1), num_classes, bias=False).to(device)
            if self.weight_norm:
                classifier = nn.utils.weight_norm(classifier, name='weight')

            # self._do_data_dependent_init(classifier, feat_s)

            if self.cosine_head:
                feat_s = F.normalize(feat_s, dim=-1)
                feat_q = F.normalize(feat_q, dim=-1)

            logits_q = self.temp * classifier(feat_q[0])
            logits_s = self.temp * classifier(feat_s[0])
            preds_q = logits_q.argmax(-1)
            probs_s = logits_s.softmax(-1)
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
            params = list(classifier.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Run adaptation
        for i in range(1, self.iter):
            if self.finetune_all_layers:
                model.train()
                feat_s, feat_q = extract_features(self.extract_batch_size,
                                                  support,
                                                  query,
                                                  model)
                if self.cosine_head:
                    feat_s = F.normalize(feat_s, dim=-1)
                    feat_q = F.normalize(feat_q, dim=-1)

            logits_s = self.temp * classifier(feat_s[0])
            probs_s = logits_s.softmax(-1)
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
