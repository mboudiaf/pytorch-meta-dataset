from .utils import compute_confidence_interval
from pathlib import Path
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import Any

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

colors = ["g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

styles = ['--', '-.', ':', '-']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str,
                        help='Folder to search')
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--linewidth', type=int, default=2.)
    parser.add_argument('--fontfamily', type=str, default='sans-serif')
    parser.add_argument('--fontweight', type=str, default='normal')
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 10])
    parser.add_argument('--dpi', type=int, default=200,
                        help='Dots per inch when saving the fig')
    parser.add_argument('--max_col', type=int, default=1,
                        help='Maximum number of columns for legend')

    parser.add_argument('--reduce_by', type=str, default='seed')
    parser.add_argument('--simu_params', type=str, nargs='+',
                        default=['method', 'base_source', 'arch'])

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    plt.rc('font',
           size=args.fontsize,
           family=args.fontfamily,
           weight=args.fontweight)

    # Recover all files that match .npy pattern in folder/
    p = Path(args.folder)
    all_files = list(p.glob('**/*.npy'))

    if not len(all_files):
        print("No .pny files found in this subtree. Cancelling plotting operations.")
        return

    # Group files by metric name
    filenames_dic = nested_default_dict(4, str)
    for path in all_files:
        root = path.parent
        metric = path.stem
        with open(root / 'config.json') as f:
            config = json.load(f)
        fixed_key = tuple([config[key] for key in args.simu_params])
        reduce_key = config[args.reduce_by]
        filenames_dic[metric][fixed_key][reduce_key]['num_updates'] = config['num_updates']
        filenames_dic[metric][fixed_key][reduce_key]['log_freq'] = config['print_freq'] if 'train' in metric else config['eval_freq']
        filenames_dic[metric][fixed_key][reduce_key]['path'] = path

    # Do one plot per metric
    for metric in filenames_dic:
        fig = plt.Figure(args.figsize)
        ax = fig.gca()
        for style, color, simu_args in zip(cycle(styles), cycle(colors), filenames_dic[metric]):
            values = []
            for _, res_dic in filenames_dic[metric][simu_args].items():
                values.append(np.load(res_dic['path'])[None, :])
            y = np.concatenate(values, axis=0)
            mean, std = compute_confidence_interval(y, axis=0)
            n_epochs = mean.shape[0]
            x = np.linspace(0, n_epochs - 1, (n_epochs)) * res_dic['log_freq']
            valid_iter = (mean != 0)

            label = "/".join([f"{k}={v}" for k, v in zip(args.simu_params, simu_args)])
            ax.plot(x[valid_iter], mean[valid_iter], label=label, color=color, linestyle=style)
            ax.fill_between(x[valid_iter], mean[valid_iter] - std[valid_iter],
                            mean[valid_iter] + std[valid_iter], color=color, alpha=0.3)

        n_cols = min(args.max_col, len(filenames_dic[metric]))
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=n_cols, shadow=True)
        ax.set_xlabel("Training iterations")
        ax.grid(True)
        fig.tight_layout()
        save_path = p / 'plots'
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'{metric}.png', dpi=args.dpi)
    print(f"Plots saved at {args.folder}")


def nested_default_dict(depth: int, final_type: Any, i: int = 1):
    if i == depth:
        return defaultdict(final_type)
    fn = partial(nested_default_dict, depth=depth, final_type=final_type, i=i+1)
    return defaultdict(fn)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
