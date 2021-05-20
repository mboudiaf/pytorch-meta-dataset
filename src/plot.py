from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import argparse
from collections import defaultdict
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
    parser.add_argument('--figsize', type=list, default=[10, 10])
    parser.add_argument('--dpi', type=list, default=200,
                        help='Dots per inch when saving the fig')
    parser.add_argument('--max_col', type=int, default=1,
                        help='Maximum number of columns for legend')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    plt.rc('font',
           size=args.fontsize,
           family=args.fontfamily,
           weight=args.fontweight)

    # Recover all files that match .npy pattern in folder/
    p = Path(args.folder)
    all_files = p.glob('**/*.npy')

    # Group files by metric name
    filenames_dic = defaultdict(list)
    for path in all_files:
        filenames_dic[path.stem].append(path)

    # Do one plot per metric
    for metric in filenames_dic:
        fig = plt.Figure(args.figsize)
        ax = fig.gca()
        for style, color, filepath in zip(cycle(styles), cycle(colors), filenames_dic[metric]):
            y = np.load(filepath)
            n_epochs = y.shape[0]
            x = np.linspace(0, n_epochs - 1, (n_epochs))

            label = filepath

            ax.plot(x, y, label=label, color=color, linewidth=args.linewidth, linestyle=style)

        n_cols = min(args.max_col, len(filenames_dic[metric]))
        ax.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=n_cols, shadow=True)
        ax.set_xlabel("Epochs")
        ax.grid(True)
        fig.tight_layout()
        save_path = p / f'{metric}.png'
        fig.savefig(save_path, dpi=args.dpi)
    print(f"Plots saved at {args.folder}")


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
