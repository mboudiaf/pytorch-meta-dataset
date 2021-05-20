import torch
import torch.nn as nn
import argparse
from typing import Dict, Tuple
from ..metrics import Metric
from torch import tensor


class FSmethod(nn.Module):
    '''
    Abstract class for few-shot methods
    '''
    def __init__(self, args: argparse.Namespace):
        super(FSmethod, self).__init__()

    def forward(self,
                model: torch.nn.Module,
                support: tensor,
                query: tensor,
                y_s: tensor,
                y_q: tensor,
                metrics: Dict[str, Metric] = None,
                task_ids: Tuple[int, int] = None) -> Tuple[tensor, tensor]:
        '''
        args:
            model: Network to train/test with
            support: Tensor representing support images, of shape [n_tasks, n_support, C, H, W]
                     where n_tasks is the batch dimension (only useful for fixed-dimension tasks)
                     and n_support the total number of support samples
            query: Tensor representing query images, of shape [n_tasks, n_query, C, H, W]
                     where n_tasks is the batch dimension (only useful for fixed-dimension tasks)
                     and n_query the total number of query samples
            y_s: Tensor representing the support labels of shape [n_tasks, n_support]
            y_q: Tensor representing the query labels of shape [n_tasks, n_query]
            metrics: A dictionnary of Metric objects to be filled during inference
                    (mostly useful if the method performs test-time inference). Refer to tim.py for
                    an instance of usage
            task_ids: Start and end tasks ids. Only used to fill the metrics dictionnary.

        returns:
            loss: Tensor of shape [] representing the loss to be minimized (for methods using episodic training)
            soft_preds: Tensor of shape [n_tasks, n_query, K], where K is the number of classes in the task,
                        representing the soft predictions of the method for the input query samples. 
        '''
