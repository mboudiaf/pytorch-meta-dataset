from argparse import Namespace

import torch
import numpy as np
import seaborn as sn
from matplotlib.axes import Axes

from ..utils import compute_confidence_interval, get_one_hot


class Metric(object):
    def __init__(self, args: Namespace):
        assert hasattr(self, 'name')

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self,
             ax: Axes,
             iteration: int):

        raise NotImplementedError


class ScalarMetric(Metric):
    def __init__(self,
                 name: str,
                 args: Namespace):
        self.values = torch.zeros(args.val_episodes, args.iter)
        self.name: str = name

        super().__init__(args)

    def plot(self,
             ax: Axes,
             iteration: int) -> None:
        """
        Plot the metrics using the values filled up so far
        """
        mean, std = compute_confidence_interval(self.values[:iteration])
        x = torch.arange(mean.shape[0])
        ax.clear()
        ax.plot(x, mean)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_title(self.name)
        ax.set_xlabel('Iteration')
        ax.grid(True)

    def update(self,
               start: int,
               end: int,
               iteration: int,
               **kwargs) -> None:
        """
        Update the internal value table of the metric
        """
        value = self(**kwargs)
        if isinstance(value, torch.Tensor):
            value = value.cpu()

        self.values[start:end, iteration] = value.squeeze()


class Acc(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Accuracy', args)

    def __call__(self, **kwargs):
        preds = kwargs['preds']
        gt = kwargs['gt']

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

        if len(preds.shape) == 3:
            acc = (preds == gt).mean(-1)
        else:
            acc = (preds == gt).mean()

        return acc


class MI(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Mutual information', args)

    def __call__(self, **kwargs):
        probs = kwargs['probs']

        cond_ent = - (probs * torch.log(probs + 1e-12)).sum(-1).mean(-1, keepdim=True)
        ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(-1, keepdim=True)

        return ent - cond_ent


class CondEnt(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Conditional Entropy', args)

    def __call__(self, **kwargs):
        probs = kwargs['probs']
        cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)

        return cond_ent


class MargEnt(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Marginal Entropy', args)

    def __call__(self, **kwargs):
        probs = kwargs['probs']
        ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)

        return ent


class XEnt(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Cross Entropy', args)

    def __call__(self, **kwargs):
        probs = kwargs['probs_s']
        gt = kwargs['gt_s']

        num_classes = gt.unique().size(0)

        gt = get_one_hot(gt, num_classes)
        xent_ent = - (gt * torch.log(probs + 1e-12)).sum(-1).mean(-1, keepdim=True)

        return xent_ent


class LaplacianEnergy(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Laplacian Energy', args)

    def __call__(self, **kwargs):
        """
        Energy used in LaplacianShot
        """
        Y = kwargs['Y']
        kernel = kwargs['kernel']
        unary = kwargs['unary']
        lmd = kwargs['lmd']

        pairwise = kernel.dot(Y)
        temp = (unary * Y) + (-lmd * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()

        return E


class MaxEigS(ScalarMetric):
    def __init__(self, args: Namespace):
        super(MaxEigS, self).__init__('Max Eig S', args)

    def __call__(self, **kwargs):
        z = kwargs['z_s'].cpu()
        probs_q = kwargs['probs_s'].cpu()  # [n_task, shot, K]
        weights = kwargs['weights'].cpu()

        n_tasks, shot, d = z.size()
        diff_s = z.unsqueeze(2) - weights.unsqueeze(1).detach()  # [n_task, shot, K, d]
        # assert diff_s.size() == (n_tasks, shot, K, d), (diff_s.size(), (n_tasks, shot, K, d))

        H = (diff_s ** 2).sum(-1)  # [n_task, shot, K]
        I = torch.ones_like(H) # [n_task, shot, K]
        p = probs_q.detach()  # [n_task, shot, K]

        hessian = p * (H - I) - (p**2) * H  # [n_task, shot, K]
        hessian = hessian.sum(1)  # [n_task, K]
        max_ = hessian.max(1).values

        return max_


class MaxEigQ(ScalarMetric):
    def __init__(self, args: Namespace):
        super().__init__('Max Eig Q', args)

    def __call__(self, **kwargs):
        z = kwargs['z_q'].cpu()
        probs_q = kwargs['probs'].cpu()  # [n_task, shot, K]
        weights = kwargs['weights'].cpu()

        n_tasks, shot, d = z.size()
        diff_s = z.unsqueeze(2) - weights.unsqueeze(1).detach()  # [n_task, shot, K, d]
        # assert diff_s.size() == (n_tasks, shot, K, d), (diff_s.size(), (n_tasks, shot, K, d))
        H = (diff_s ** 2).sum(-1)  # [n_task, shot, K]
        I = torch.ones_like(H) # [n_task, shot, K]
        p = probs_q.detach()  # [n_task, shot, K]

        hessian = p * (H - I) - (p**2) * H  # [n_task, shot, K]
        hessian = hessian.sum(1)  # [n_task, K]
        max_ = hessian.max(1).values
        return max_


class ConfMatrix(Metric):
    def __init__(self, args: Namespace):
        self.name = ""

        if not args.num_ways:
            raise ValueError("Confusion Matrix is only available for fixed tasks")

        self.values = torch.zeros(args.val_episodes, args.iter, args.num_ways, args.num_ways)

        super().__init__(args)

    def update(self,
               start,
               end,
               iteration,
               **kwargs):
        value = self(**kwargs)
        self.values[start:end, iteration] = value.squeeze().cpu()

    def __call__(self, **kwargs):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
            gt : torch.Tensor of shape [n_task, shot]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        preds = kwargs['preds']
        gt = kwargs['gt']

        num_classes = gt.unique().size(0)

        one_hot_preds = get_one_hot(preds, num_classes)
        one_hot_gt = get_one_hot(gt, num_classes)  # [n_task, shot, num_classes]

        conf = one_hot_preds.permute(0, 2, 1).matmul(one_hot_gt)  # [n_task, num_classes, num_classes]
        conf /= conf.sum(dim=(1, 2), keepdim=True)
        # conf /= one_hot_gt.sum(1, keepdim=True)  # [num_classes, num_classes]

        return conf.mean(0)

    def plot(self, ax: Axes, iteration: int):
        conf = self.values[:iteration].mean(0)[-1]  # averaging the initial conf matrix over all tasks

        sn.set(font_scale=1.4)
        sn.heatmap(conf, annot=True, annot_kws={"size": 10}, ax=ax)

        if len(self.name):
            ax.set_title(self.name)

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
