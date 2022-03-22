from .standard import __dict__ as standard_dict
from .meta import __dict__ as meta_dict
from loguru import logger
import argparse


def get_model(args: argparse.Namespace,
              num_classes: int):
    if 'MAML' in args.method:
        logger.info(f"Meta {args.arch} loaded")
        return meta_dict[args.arch](num_classes=num_classes)
    else:
        logger.info(f"Standard {args.arch} loaded")
        return standard_dict[args.arch](num_classes=num_classes,
                                        pretrained=args.load_from_timm)
