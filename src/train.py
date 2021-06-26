import os
import time
import random
import argparse
from functools import partial
from typing import Dict, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as tmp
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch import tensor, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from .losses import __losses__
from .methods import FSmethod
from .methods import __dict__ as all_methods
from .optim import get_optimizer, get_scheduler
from .datasets.utils import Split
from .datasets.loader import get_dataloader
from .models.ingredient import get_model
from .models.meta.metamodules.module import MetaModule
from .utils import (AverageMeter, save_checkpoint, get_model_dir,
                    load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port,
                    setup, cleanup, main_process, copy_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--base_config', type=str, required=True, help='config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert args.base_config is not None

    cfg = load_cfg_from_cfg_file(Path(args.base_config))
    cfg.update(load_cfg_from_cfg_file(Path(args.method_config)))

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    return cfg


def meta_val(args: argparse.Namespace,
             model: torch.nn.Module,
             method: FSmethod,
             val_loader: torch.utils.data.DataLoader) -> Tuple[Tensor, Tensor]:
    # Device
    device = dist.get_rank()
    model.eval()
    method.eval()

    # Metrics
    episode_acc = tensor([0.], device=device)

    total_episodes = int(args.val_episodes / args.val_batch_size)
    tqdm_bar = tqdm(val_loader, total=total_episodes)
    for i, (support, query, support_labels, query_labels) in enumerate(tqdm_bar):
        if i >= total_episodes:
            break

        y_s = support_labels.to(device, non_blocking=True)
        y_q = query_labels.to(device, non_blocking=True)

        _, soft_preds_q = method(model=model,
                                 support=support,
                                 query=query,
                                 y_s=y_s,
                                 y_q=y_q)

        soft_preds_q = soft_preds_q.to(device).detach()
        episode_acc += (soft_preds_q.argmax(-1) == y_q).float().mean()

        tqdm_bar.set_description('Acc {:.2f}'.format((episode_acc / (i + 1) * 100).item()))

    n_episodes = tensor(total_episodes, device=device)

    model.train()
    method.train()

    return episode_acc, n_episodes


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:
    print(f"==> Running process rank {rank}.")
    setup(args.port, rank, world_size)
    device: int = rank

    if args.seed is not None:
        args.seed += rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # ============ Define loaders ================
    train_loader, num_classes = get_dataloader(args=args,
                                               source=args.base_source,
                                               batch_size=args.batch_size,
                                               world_size=world_size,
                                               split=Split["TRAIN"],
                                               episodic=args.episodic_training,
                                               version=args.loader_version)

    val_loader, _ = get_dataloader(args=args,
                                   source=args.val_source,
                                   batch_size=args.val_batch_size,
                                   world_size=world_size,
                                   split=Split["VALID"],
                                   episodic=True,
                                   version=args.loader_version)

    # ============ Define model ================
    num_classes = args.num_ways if args.episodic_training else num_classes
    if main_process(args):
        print("=> Creating model '{}' with {} classes".format(args.arch,
                                                              num_classes))
    model = get_model(args=args, num_classes=num_classes).to(rank)
    if not isinstance(model, MetaModule) and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    if main_process(args):
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model_dir = get_model_dir(args)
    copy_config(args, model_dir)

    # ============ Define metrics ================
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if main_process(args):
        metrics: Dict[str, Tensor] = {"train_loss": torch.zeros(args.num_updates // args.print_freq,
                                                                dtype=torch.float32),
                                      "val_acc": torch.zeros(args.num_updates // args.eval_freq,
                                                             dtype=torch.float32)}

    # ============ Optimizer ================
    optimizer = get_optimizer(args=args, model=model)
    scheduler = get_scheduler(args=args, optimizer=optimizer)

    # ============ Method ================
    method = all_methods[args.method](args=args)
    if not args.episodic_training:
        if args.loss not in __losses__:
            raise ValueError(f'Please set the loss among : {list(__losses__.keys())}')
        loss_fn = __losses__[args.loss]
        loss_fn = loss_fn(args=args, num_classes=num_classes, reduction='none')
    eval_fn = partial(meta_val, method=method, val_loader=val_loader, model=model, args=args)

    # ============ Start training ================
    model.train()
    method.train()

    best_val_acc1 = 0.
    tqdm_bar = tqdm(train_loader, total=args.num_updates)
    for i, data in enumerate(tqdm_bar):
        if i >= args.num_updates:
            break

        # ======== Forward / Backward pass =========
        t0 = time.time()
        if args.episodic_training:
            support, query, support_labels, target = data
            support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
            query, target = query.to(device), target.to(device, non_blocking=True)

            loss, preds_q = method(support=support,
                                   query=query,
                                   y_s=support_labels,
                                   y_q=target,
                                   model=model)  # [batch, q_shot]
        else:
            (input_, target) = data
            input_, target = input_.to(device), target.to(device, non_blocking=True).long()
            loss = loss_fn(input_, target, model)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # ============ Log metrics ============
        b_size = tensor([args.batch_size]).to(device)  # type: ignore
        if args.distributed:
            dist.all_reduce(b_size)
            dist.all_reduce(loss)
        if main_process(args):
            loss = loss.sum() / b_size
            losses.update(loss.item(), b_size.item(), i == 0)
            batch_time.update(time.time() - t0, i == 0)
            t0 = time.time()

        # ============ Validation ============
        if i % args.eval_freq == 0:
            val_acc, n_episodes = eval_fn()
            if args.distributed:
                dist.all_reduce(val_acc)
                dist.all_reduce(n_episodes)

            val_acc /= n_episodes
            is_best = (val_acc > best_val_acc1)
            best_val_acc1 = max(val_acc, best_val_acc1)

            if main_process(args):
                save_checkpoint(state={'iter': i,
                                       'arch': args.arch,
                                       'state_dict': model.state_dict(),
                                       'best_prec1': best_val_acc1},
                                is_best=is_best,
                                folder=model_dir)

                for k in metrics:
                    if 'val' in k:
                        metrics[k][int(i / args.eval_freq)] = eval(k)

                for k, e in metrics.items():
                    path = os.path.join(model_dir, f"{k}.npy")
                    np.save(path, e.cpu().numpy())

        # ============ Print / log metrics ============
        if i % args.print_freq == 0 and main_process(args):
            print('Iration: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, args.num_updates, batch_time=batch_time,  # noqa: E121
                   loss=losses, top1=top1))

            train_loss = losses.avg
            for k in metrics:
                if 'train' in k:
                    metrics[k][int(i / args.print_freq)] = eval(k)

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    if args.debug:
        args.batch_size = 16
        args.val_episodes = 10
    world_size = len(args.gpus)
    distributed = world_size > 1
    args.world_size = world_size
    args.distributed = distributed
    args.port = find_free_port()
    tmp.spawn(main_worker,
              args=(world_size, args),
              nprocs=world_size,
              join=True)
