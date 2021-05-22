import argparse

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

from .utils import rand_bbox


class _Loss(nn.Module):
    def __init__(self,
                 args: argparse.Namespace,
                 num_classes: int,
                 reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()

        self.reduction: str = reduction
        self.augmentation = None if args.augmentation not in ['mixup', 'cutmix'] \
                            else eval(f'self.{args.augmentation}')  # noqa: E127

        assert 0 <= args.label_smoothing < 1
        self.label_smoothing: float = args.label_smoothing

        self.num_classes: int = num_classes
        self.beta: int = args.beta

        self.cutmix_prob: float = args.cutmix_prob  # Clash with the method `cutmix`

    def smooth_one_hot(self,
                       targets: Tensor):
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=targets.device)
            new_targets.fill_(self.label_smoothing / (self.num_classes - 1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - self.label_smoothing)

        return new_targets

    def mixup(self,
              input_: Tensor,
              one_hot_targets: Tensor,
              model: nn.Module):
        # Forward pass
        device = one_hot_targets.device

        # generate mixed sample and targets
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input_.size()[0]).to(device)

        target_a = one_hot_targets
        target_b = one_hot_targets[rand_index]

        mixed_input_ = lam * input_ + (1 - lam) * input_[rand_index]
        output = model(mixed_input_)

        loss = self.loss_fn(output, target_a) * lam + self.loss_fn(output, target_b) * (1. - lam)

        return loss

    def cutmix(self,
               input_: Tensor,
               one_hot_targets: Tensor,
               model: nn.Module):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input_.size()[0]).cuda()

        target_a = one_hot_targets
        target_b = one_hot_targets[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(input_.size(), lam)
        input_[:, :, bbx1:bbx2, bby1:bby2] = input_[rand_index, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_.size()[-1] * input_.size()[-2]))
        output = model(input_)

        loss = self.loss_fn(output, target_a) * lam + self.loss_fn(output, target_b) * (1. - lam)

        return loss

    def loss_fn(self,
                logits: Tensor,
                one_hot_targets: Tensor):
        raise NotImplementedError

    def forward(self,
                logits: Tensor,
                targets: Tensor,
                model: torch.nn.Module):
        raise NotImplementedError


class _CrossEntropy(_Loss):
    def loss_fn(self,
                logits: Tensor,
                one_hot_targets: Tensor):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        loss = - (one_hot_targets * logsoftmax).sum(1)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

    def forward(self,
                input_: Tensor,
                targets: Tensor,
                model: nn.Module):

        one_hot_targets = self.smooth_one_hot(targets)
        if self.augmentation:
            return self.augmentation(input_, one_hot_targets, model)
        else:
            logits = model(input_)

            return self.loss_fn(logits, one_hot_targets)


class _FocalLoss(_Loss):
    def __init__(self,
                 **kwargs) -> None:
        super(_FocalLoss, self).__init__(**kwargs)
        self.gamma = kwargs['args'].focal_gamma

    def loss_fn(self,
                logits: Tensor,
                one_hot_targets: Tensor):
        softmax = logits.softmax(1)
        logsoftmax = torch.log(softmax + 1e-10)
        loss = - (one_hot_targets * (1 - softmax)**self.gamma * logsoftmax).sum(1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss

    def forward(self,
                input_: Tensor,
                targets: Tensor,
                model: nn.Module):
        one_hot_targets = self.smooth_one_hot(targets)
        if self.augmentation:
            return self.augmentation(input_, one_hot_targets, model)
        else:
            logits = model(input_)

            return self.loss_fn(logits, one_hot_targets)


__losses__ = {'xent': _CrossEntropy, 'focal': _FocalLoss}
