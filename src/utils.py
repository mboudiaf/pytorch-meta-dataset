import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
import argparse
import torch.distributed as dist
import yaml
import copy
from typing import List, Dict
from ast import literal_eval
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.axes import Axes
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
        print(f"Support images between {img_s.min()} and {img_s.max()} -> Renormalizing")
        img_s *= std
        img_s += mean
        print(f"Post normalization : {img_s.min()} and {img_s.max()}")

    if img_q.min() < 0:
        print(f"Query images between {img_q.min()} and {img_q.max()} -> Renormalizing")
        img_q *= std
        img_q += mean
        print(f"Post normalization : {img_q.min()} and {img_q.max()}")

    Kq, num_classes = preds.shape
    Ks = img_s.shape[0]

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
                # print(img.min(), img.max())
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
                # print(img.min(), img.max())
                # assert img.min() >= 0. and img.max() <= 1.0, (img.min(), img.max())
                make_plot(ax, img, preds_q[j][i - max_s], j, n_columns)
            ax.axis('off')
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]
    acc = (np.argmax(preds, axis=1) == gt_q).mean()
    fig.suptitle(f'Method={args.method}   /    Episode Accuracy={acc:.2f}', size=32, weight='bold', y=0.97)
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.05),
               loc='center', ncol=3, prop={'size': 30})
    fig.savefig(save_path)
    fig.clf()
    print(f"Figure saved at {save_path}")


def frame_image(img: np.ndarray, color: list, frame_width: int =3) -> np.ndarray:
    b = frame_width  # border size in pixel
    ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
    framed_img = color * np.ones((b + ny + b, b + nx + b, img.shape[2]))
    framed_img[b:-b, b:-b] = img
    return framed_img


def make_plot(ax: Axes,
              img: np.ndarray,
              preds: np.ndarray = None,
              label: np.ndarray = None,
              n_columns: int = 0) -> None:

    if preds is not None:
        title = ['{:.2f}'.format(p) for p in preds]
        title[np.argmax(preds)] = r'$\mathbf{{{}}}$'.format(title[np.argmax(preds)])
        title = title[:n_columns]
        title = '/'.join(title)
        # ax.set_title(title, size=12)
        well_classified = np.argmax(preds) == label
        color = [0, 0.8, 0] if well_classified else [0.9, 0, 0]
        img = frame_image(img, color)
        ax.plot(0, 0, "-", c=color, label='{} Queries'\
            .format('Well classified' if well_classified else 'Misclassified'), linewidth=4)
    else:  # Support images
        color = [0., 0., 0.]
        img = frame_image(img, color)
        ax.plot(0, 0, "-", c=color, label='Support', linewidth=4)
    ax.imshow(img)


def main_process(distributed) -> bool:
    if distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(port: int,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

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


def get_one_hot(y_s: torch.tensor, num_classes: int) -> torch.tensor:
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
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_model_dir(args: argparse.Namespace):
    model_type = args.method if args.episodic_training else 'standard'
    return os.path.join(args.ckpt_path,
                        f'base={str(args.base_sources)}',
                        f'val={str(args.val_sources)}',
                        f'arch={args.arch}',
                        f'method={model_type}')


def get_logs_path(model_path, method, shot):
    exp_path = '_'.join(model_path.split('/')[1:])
    file_path = os.path.join('tmp', exp_path, method)
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, f'{shot}.txt')


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


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def warp_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


def load_checkpoint(model, model_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/model_best.pth.tar')
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(model_path))
        print(f'Loaded model from {model_path}/checkpoint.pth.tar')
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    state_dict = checkpoint['state_dict']
    names = []
    for k, v in state_dict.items():
        names.append(k)
    model.load_state_dict(state_dict)


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
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
    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

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


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

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
