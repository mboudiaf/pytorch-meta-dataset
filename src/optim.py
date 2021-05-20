import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import argparse


def config():
    gamma = 0.1
    lr = 0.1
    lr_stepsize = 30
    nesterov = False
    weight_decay = 1e-4
    optimizer_name = 'SGD'
    scheduler = 'multi_step'


def get_scheduler(args: argparse.Namespace,
                  optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

    SCHEDULER = {'step': StepLR(optimizer, args.lr_stepsize, args.gamma),
                 'multi_step': MultiStepLR(optimizer,
                                           milestones=[int(.5 * args.num_updates),
                                                       int(.75 * args.num_updates)],
                                           gamma=args.gamma),
                 'cosine': CosineAnnealingLR(optimizer, args.num_updates, eta_min=1e-9),
                 None: None}
    return SCHEDULER[args.scheduler]


def get_optimizer(args: argparse.Namespace,
                  model: torch.nn.Module) -> torch.optim.Optimizer:
    OPTIMIZER = {'SGD': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)}
    return OPTIMIZER[args.optimizer_name]