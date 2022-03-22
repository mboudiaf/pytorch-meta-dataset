import os
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from tqdm import trange
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from .datasets.utils import Split
from .datasets.loader import get_dataloader
from .methods import __dict__ as all_methods
from .metrics import __dict__ as all_metrics
from .models.ingredient import get_model
from .models.meta.metamodules.module import MetaModule
from .train import parse_args
from .utils import make_episode_visualization, plot_metrics
from .utils import (compute_confidence_interval, load_checkpoint, get_model_dir,
                    find_free_port, setup, cleanup, copy_config)


def hash_config(args: argparse.Namespace) -> str:
    res = 0
    for i, (key, value) in enumerate(args.items()):
        if key != "port":
            if type(value) == str:
                hash_ = sum(value.encode())
            elif type(value) in [float, int, bool]:
                hash_ = round(value, 3)
            else:
                hash_ = sum([int(v) if type(v) in [float, int, bool] else sum(v.encode()) for v in value])
            res += hash_ * random.randint(1, int(1e6))

    return str(res)[-10:].split('.')[0]


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:
    """
    Run the evaluation over all the tasks in parallel
    inputs:
        model : The loaded model containing the feature extractor
        loaders_dic : Dictionnary containing training and testing loaders
        model_path : Where was the model loaded from
        model_tag : Which model ('final' or 'best') to load
        method : Which method to use for inference ("baseline", "tim-gd" or "tim-adm")
        shots : Number of support shots to try

    returns :
        results : List of the mean accuracy for each number of support shots
    """
    logger.info(f"==> Running process rank {rank}.")
    setup(args.port, rank, world_size)

    # ===============> Setup directories for current exp. <=================
    # ======================================================================
    exp_root = Path(os.path.join(args.res_path, args.method))
    exp_root.mkdir(exist_ok=True, parents=True)
    exp_no = hash_config(args)
    exp_root = exp_root / str(exp_no)
    copy_config(args, exp_root)

    logger.info(f"==>  Saving all at {exp_root}")
    device = rank
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    # ===============> Load data <=================
    # =============================================
    current_split = "TEST" if args.eval_mode == 'test' else "VALID"

    # _, num_classes_base = get_dataloader(args=args,
    #                                      source=args.base_source,
    #                                      batch_size=args.batch_size,
    #                                      world_size=world_size,
    #                                      split=Split["TRAIN"],
    #                                      episodic=True,
    #                                      version=args.loader_version)

    test_loader, num_classes_test = get_dataloader(args=args,
                                                   source=args.test_source,
                                                   batch_size=args.val_batch_size,
                                                   world_size=world_size,
                                                   split=Split[current_split],
                                                   episodic=True,
                                                   version=args.loader_version)

    # logger.info(f"BASE dataset: {args.base_source} ({num_classes_base} classes)")
    logger.info(f"{current_split} dataset: {args.test_source} ({num_classes_test} classes)")

    # ===============> Load model <=================
    # ==============================================
    # num_classes = 5 if args.episodic_training else num_classes_base
    model = get_model(args=args, num_classes=5).to(rank)
    if not isinstance(model, MetaModule) and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if not args.load_from_timm:  # then we load the model from a local checkpoint obtained through training
        model_path = get_model_dir(args=args)
        load_checkpoint(model=model, model_path=model_path)
    model.eval()

    # ===============> Define metrics <=================
    # ==================================================
    metrics = {}
    for metric_name in args.eval_metrics:
        metrics[metric_name] = all_metrics[metric_name](args)

    iter_loader = iter(test_loader)

    # ===============> Load method <=================
    # ===============================================
    method = all_methods[args.method](args=args)
    method.eval()

    # ===============> Run method <=================
    # ==============================================
    acc = 0.
    tqdm_bar = trange(int(args.val_episodes / args.val_batch_size))
    for i in tqdm_bar:
        # ======> Reload model checkpoint (some methods may modify model) <=======
        support, query, support_labels, query_labels = next(iter_loader)
        # logger.info(query_labels.size())

        support_labels = support_labels.to(device, non_blocking=True)
        query_labels = query_labels.to(device, non_blocking=True)
        task_ids = (i * args.val_batch_size, (i + 1) * args.val_batch_size)
        loss, soft_preds_q = method(model=model,
                                    metrics=metrics,
                                    task_ids=task_ids,
                                    support=support,
                                    query=query,
                                    y_s=support_labels,
                                    y_q=query_labels)
        soft_preds_q = soft_preds_q.to(device)
        acc += (soft_preds_q.argmax(-1) == query_labels).float().mean()

        tqdm_bar.set_description('Acc {:.2f}'.format(100 * acc / (i + 1)))

        # ======> Plot inference metrics <=======
        if i % args.plot_freq == 0 and args.eval_mode == 'test' and args.iter > 1:
            path = os.path.join(exp_root, f"{str(args.base_source)}->{str(args.test_source)}.pdf")
            plot_metrics(metrics, path, task_ids[1], args)

        # =======> Visualize a randomly chosen episode <==========
        if args.visu and i % args.visu_freq == 0 and args.eval_mode == 'test':
            visu_path = os.path.join(exp_root, 'episode_samples', args.loader_version)
            os.makedirs(visu_path, exist_ok=True)
            path = os.path.join(visu_path, f'visu_{i}.png')
            make_episode_visualization(args,
                                       support[0].cpu().numpy(),
                                       query[0].cpu().numpy(),
                                       support_labels[0].cpu().numpy(),
                                       query_labels[0].cpu().numpy(),
                                       soft_preds_q[0].cpu().numpy(),
                                       path)

        # ======> Plot metrics <=======
        if i % args.plot_freq == 0:
            update_csv(args=args,
                       metrics=metrics,
                       task_id=i + 1,
                       path=os.path.join(exp_root, f'{args.eval_mode}.csv'))
    cleanup()


def update_csv(args: argparse.Namespace,
               task_id: int,
               metrics: dict,
               path: str):
    if 'Acc' not in metrics:
        raise ValueError('Cannot save csv result without Accuracy metric')

    # res = OrderedDict()
    try:
        res = pd.read_csv(path)
    except FileNotFoundError:
        res = pd.DataFrame({})

    records = res.to_dict('records')
    l2n_mean, l2n_conf = compute_confidence_interval(metrics['Acc'].values[:task_id, -1])

    # If entry did not exist, just create it
    new_entry = {param: args[param]
                 for param in args.hyperparams}
    new_entry['task'] = task_id
    new_entry['acc'] = round(l2n_mean, 4)
    new_entry['std'] = round(l2n_conf, 4)

    records = [new_entry]

    # Save back to dataframe
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
