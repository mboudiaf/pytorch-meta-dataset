from argparse import Namespace
from typing import Optional

import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR


def get_scheduler(args: Namespace,
                  optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:

    SCHEDULER = {'step': StepLR(optimizer, args.lr_stepsize, args.gamma),
                 'multi_step': MultiStepLR(optimizer,
                                           milestones=[int(.5 * args.num_updates),
                                                       int(.75 * args.num_updates)],
                                           gamma=args.gamma),
                 'cosine': CosineAnnealingLR(optimizer, args.num_updates, eta_min=1e-9),
                 None: None}

    assert args.scheduler in SCHEDULER.keys()

    return SCHEDULER[args.scheduler]


def get_optimizer(args: Namespace,
                  model: torch.nn.Module) -> torch.optim.Optimizer:
    OPTIMIZER = {'SGD': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)}

    assert args.optimizer_name in OPTIMIZER.keys()

    return OPTIMIZER[args.optimizer_name]
