from .standard import __dict__ as standard_dict
from .meta import __dict__ as meta_dict
import argparse


def get_model(args: argparse.Namespace,
              num_classes: int):
    if 'MAML' in args.method:
        print(f"Meta {args.arch} loaded")
        return meta_dict[args.arch](num_classes=num_classes, use_fc=args.use_fc)
    else:
        print(f"Standard {args.arch} loaded")
        return standard_dict[args.arch](num_classes=num_classes, use_fc=args.use_fc)
