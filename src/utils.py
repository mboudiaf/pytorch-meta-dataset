import yaml
import copy
import pickle
import shutil
import argparse
import json
from os import environ
from pathlib import Path
from ast import literal_eval
from typing import Any, List, Tuple, Union, cast

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import ImageGrid
from loguru import logger
plt.style.use('ggplot')


def plot_metrics(metrics: dict,
                 path: str,
                 iteration: int,
                 args: argparse.Namespace) -> None:
    plt.rc('font',
           size=16)

    n_cols = min(2, len(metrics))
    n_rows = (len(metrics) - 1) // n_cols + 1
    _, axs = plt.subplots(nrows=n_rows,
                          ncols=n_cols,
                          figsize=(6 * n_cols, 5 * n_rows),
                          squeeze=False)

    for j, (metric_name, metric) in enumerate(metrics.items()):
        ax = axs[j // n_cols, j % n_cols]
        metric.plot(ax, iteration)

    plt.tight_layout()
    plt.savefig(path, dpi=300)


def make_episode_visualization(args: argparse.Namespace,
                               img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               mean: List[float] = [0.485, 0.456, 0.406],
                               std: List[float] = [0.229, 0.224, 0.225]):
    max_support = args.max_s_visu
    max_query = args.max_q_visu
    max_classes = args.max_class_visu

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : Ks x 3 x H x W or Ks x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 4, f"Query shape expected : Kq x 3 x H x W or Kq x H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 2, f"Predictions shape expected : Kq x num_classes. Currently: {preds.shape}"
    assert len(gt_s.shape) == 1, f"Support GT shape expected : Ks. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 1, f"Query GT shape expected : Kq. Currently: {gt_q.shape}"

    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[1] == 3:
        img_q = np.transpose(img_q, (0, 2, 3, 1))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1], f"Support's resolution is {img_s.shape[-3:-1]} \
                                                      Query's resolution is {img_q.shape[-3:-1]}"

    if img_s.min() < 0:
        logger.info(f"Support images between {img_s.min()} and {img_s.max()} -> Renormalizing")
        img_s *= std
        img_s += mean
        logger.info(f"Post normalization : {img_s.min()} and {img_s.max()}")

    if img_q.min() < 0:
        logger.info(f"Query images between {img_q.min()} and {img_q.max()} -> Renormalizing")
        img_q *= std
        img_q += mean
        logger.info(f"Post normalization : {img_q.min()} and {img_q.max()}")

    Kq, num_classes = preds.shape

    # Group samples by class
    samples_s = {}
    samples_q = {}
    preds_q = {}

    for class_ in np.unique(gt_s):
        samples_s[class_] = img_s[gt_s == class_]
        samples_q[class_] = img_q[gt_q == class_]
        preds_q[class_] = preds[gt_q == class_]

    # Create Grid
    max_s = min(max_support, np.max([v.shape[0] for v in samples_s.values()]))
    max_q = min(max_query, np.max([v.shape[0] for v in samples_q.values()]))
    n_rows = max_s + max_q
    n_columns = min(num_classes, max_classes)
    assert n_columns > 0

    fig = plt.figure(figsize=(4 * n_columns, 4 * n_rows), dpi=100)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_rows, n_columns),
                     axes_pad=(0.4, 0.4),
                     direction='row',
                     )

    # 1) visualize the support set
    handles = []
    labels = []
    for i in range(max_s):
        for j in range(n_columns):
            ax = grid[n_columns * i + j]
            if i < len(samples_s[j]):
                img = samples_s[j][i]
                # logger.info(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())
                make_plot(ax, img)

            ax.axis('off')
            if i == 0:
                ax.set_title(f'Class {j+1}', size=20)

            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

    # 1) visualize the query set
    for i in range(max_s, max_s + max_q):
        for j in range(n_columns):
            ax = grid[n_columns * i + j]
            if i - max_s < len(samples_q[j]):
                img = samples_q[j][i - max_s]
                # logger.info(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())

                make_plot(ax, img, preds_q[j][i - max_s], j, n_columns)

            ax.axis('off')
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

    acc = (np.argmax(preds, axis=1) == gt_q).mean()
    fig.suptitle(f'Method={args.method}   /    Episode Accuracy={acc:.2f}',
                 size=32,
                 weight='bold',
                 y=0.97)
    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(),
               by_label.keys(),
               bbox_to_anchor=(0.5, 0.05),
               loc='center',
               ncol=3,
               prop={'size': 30})

    fig.savefig(save_path)
    fig.clf()
    logger.info(f"Figure saved at {save_path}")


def frame_image(img: np.ndarray, color: list, frame_width: int = 3) -> np.ndarray:
    b = frame_width  # border size in pixel
    ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y

    framed_img = color * np.ones((b + ny + b, b + nx + b, img.shape[2]))
    framed_img[b:-b, b:-b] = img

    return framed_img


def make_plot(ax: Axes,
              img: np.ndarray,
              preds: np.ndarray = None,
              label: int = None,
              n_columns: int = 0) -> None:

    if preds is not None:
        assert label is not None
        assert n_columns > 0

        titles: List[str] = ['{:.2f}'.format(p) for p in preds]

        pred_class: int = int(np.argmax(preds))
        titles[pred_class] = r'$\mathbf{{{}}}$'.format(titles[pred_class])
        titles = titles[:n_columns]

        title: str = '/'.join(titles)
        # ax.set_title(title, size=12)

        well_classified: bool = int(np.argmax(preds)) == label
        color = [0, 0.8, 0] if well_classified else [0.9, 0, 0]
        img = frame_image(img, color)
        ax.plot(0,
                0,
                "-",
                c=color,
                label='{} Queries'.format('Well classified' if well_classified
                                          else 'Misclassified'),
                linewidth=4)
    else:  # Support images
        color = [0., 0., 0.]
        img = frame_image(img, color)
        ax.plot(0, 0, "-", c=color, label='Support', linewidth=4)

    ax.imshow(img)


def main_process(args) -> bool:
    if args.distributed:
        return dist.get_rank() == 0

    return True


def setup(port: int,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    return port


def get_one_hot(y_s: Tensor, num_classes: int) -> Tensor:
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)

    return one_hot


def rand_bbox(size: torch.Size,
              lam: float):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_model_dir(args: argparse.Namespace) -> Path:
    model_type = args.method if args.episodic_training else 'standard'

    return Path(args.ckpt_path,
                f'base={args.base_source}',
                f'val={args.val_source}',
                f'arch={args.arch}',
                f'method={model_type}')


def get_logs_path(model_path: Path, method: str, shot: int) -> Path:
    exp_path: str = '_'.join(str(model_path).split('/')[1:])

    file_path: Path = Path('tmp') / exp_path / method
    file_path.mkdir(parents=True, exist_ok=True)

    return file_path / f'{shot}.txt'


def get_features(model, samples):
    features, _ = model(samples, True)
    features = F.normalize(features.view(features.size(0), -1), dim=1)

    return features


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0

    def update(self, val, init, alpha=0.2):
        self.val = val

        if init:
            self.avg = val
        else:
            self.avg = alpha * val + (1 - alpha) * self.avg


def save_pickle(file: Union[Path, str], data: Any) -> None:
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file: Union[Path, str]) -> Any:
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state: Any,
                    is_best: bool,
                    filename: str = 'checkpoint.pth.tar',
                    folder: Path = None) -> None:
    if not folder:
        folder = Path('result/default')
    folder.mkdir(parents=False, exist_ok=True)

    torch.save(state, folder / filename)

    if is_best:
        shutil.copyfile(folder / filename, folder / 'model_best.pth.tar')


def load_checkpoint(model, model_path, type='best') -> None:
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
        logger.info(f'Loaded model from {model_path}/model_best.pth.tar')
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
        logger.info(f'Loaded model from {model_path}/checkpoint.pth.tar')
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)

    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys} \n Unexpected keys: {unexpected_keys}")


def copy_config(args: argparse.Namespace, exp_root: Path, code_root: Path = Path("src/")):
    # ========== Copy source code ==========
    python_files = list(code_root.glob('**/*.py'))
    filtered_list = [file
                     for file in python_files
                     if 'checkpoints' not in str(file) and 'results' not in str(file)]
    for file in filtered_list:
        file_dest = exp_root / 'src_code' / file
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, file_dest)

    # ========== Copy yaml files ==========
    with open(exp_root / 'config.json', 'w') as fp:
        json.dump(args, fp, indent=4)


def compute_confidence_interval(data: Union[np.ndarray, torch.Tensor], axis=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    # a = 1.0 * np.array(data)
    # a = data.astype(np.float64)

    # Casting is a pure mypy operation (no real impact), but its better to
    # explicit what is going on
    a: np.ndarray = (cast(np.ndarray, data) if type(data) == np.ndarray
                     else cast(Tensor, data).numpy())
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)

    pm = 1.96 * (std / np.sqrt(a.shape[axis]))

    m = m.astype(np.float64)
    pm = pm.astype(np.float64)

    return m, pm


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list

        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])

        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_

            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s

            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)

        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v

    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass

    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)

        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: Path):
    cfg = {}
    assert file.suffix == '.yaml', f"{file} is not a yaml file"

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)

    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):

    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list

    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)

        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )

        setattr(new_cfg, subkey, value)

    return new_cfg
