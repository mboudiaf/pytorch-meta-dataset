import argparse
from typing import Dict, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from .method import FSmethod
from ..models.meta.metamodules import MetaModule
from .utils import get_one_hot
from ..metrics import Metric


class MAML(FSmethod):
    """
    Implementation of MAML (ICML 2017) https://arxiv.org/abs/1703.03400.
    Inspired by https://github.com/tristandeleu/pytorch-meta/tree/master/examples/maml
    """

    def __init__(self, args: argparse.Namespace):
        self.step_size = args.step_size
        self.first_order = args.first_order
        self.iter = args.iter
        self.train_iter = args.iter

        super().__init__(args)

    def forward(self,
                model: torch.nn.Module,
                support: Tensor,
                query: Tensor,
                y_s: Tensor,
                y_q: Tensor,
                metrics: Dict[str, Metric] = None,
                task_ids: Tuple[int, int] = None) -> Tuple[Optional[Tensor], Tensor]:
        iter_ = self.train_iter if self.training else self.iter

        model.train()
        device = torch.distributed.get_rank()

        outer_loss = torch.tensor(0., device=device)
        soft_preds = torch.zeros_like(get_one_hot(y_q, y_s.unique().size(0)))

        for task_idx, (x_s, y_support, x_q, y_query) in enumerate(zip(support,
                                                                      y_s,
                                                                      query,
                                                                      y_q)):
            params = None
            x_s, x_q = x_s.to(device), x_q.to(device)

            for i in range(iter_):
                train_logit = model(x_s, params=params)
                inner_loss = F.cross_entropy(train_logit, y_support)

                model.zero_grad()
                params = self.gradient_update_parameters(model=model,
                                                         loss=inner_loss,
                                                         params=params)

        if not self.training:  # if doing evaluation, put back the model in eval()
            model.eval()

        with torch.set_grad_enabled(self.training):
            query_logit = model(x_q, params=params)

            outer_loss += F.cross_entropy(query_logit, y_query)
            soft_preds[task_idx] = query_logit.detach().softmax(-1)

        return outer_loss, soft_preds

    def gradient_update_parameters(self,
                                   model,
                                   loss,
                                   params=None) -> OrderedDict:
        """Update of the meta-parameters with one step of gradient descent on the
        loss function.
        Parameters
        ----------
        model : `torchmeta.modules.MetaModule` instance
            The model.
        loss : `torch.Tensor` instance
            The value of the inner-loss. This is the result of the training dataset
            through the loss function.
        params : `collections.OrderedDict` instance, optional
            Dictionary containing the meta-parameters of the model. If `None`, then
            the values stored in `model.meta_named_parameters()` are used. This is
            useful for running multiple steps of gradient descent as the inner-loop.
        step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
            The step size in the gradient update. If an `OrderedDict`, then the
            keys must match the keys in `params`.
        first_order : bool (default: `False`)
            If `True`, then the first order approximation of MAML is used.
        Returns
        -------
        updated_params : `collections.OrderedDict` instance
            Dictionary containing the updated meta-parameters of the model, with one
            gradient update wrt. the inner-loss.
        """
        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.'
                             'MetaModule`, got `{0}`'.format(type(model)))

        if params is None:
            params = OrderedDict(model.meta_named_parameters())

        create_graph = (not self.first_order) and self.training
        grads = torch.autograd.grad(loss,
                                    params.values(),
                                    create_graph=create_graph)

        updated_params = OrderedDict()

        if isinstance(self.step_size, (dict, OrderedDict)):
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - self.step_size[name] * grad

        else:
            for (name, param), grad in zip(params.items(), grads):
                updated_params[name] = param - self.step_size * grad

        return updated_params
