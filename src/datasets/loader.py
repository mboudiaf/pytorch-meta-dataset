from torch.utils.data import DataLoader
from typing import List
import torch
import numpy as np
from functools import partial
import argparse
import os

from . import dataset_spec as dataset_spec_lib
from . import config as config_lib
from . import pipeline as torch_pipeline
from .utils import Split
from .pipeline import worker_init_fn_
from . import config as config_lib



def get_dataspecs(args: argparse.Namespace,
                  sources: List[str]):
    # Recovering data
    data_config = config_lib.DataConfig(args=args)
    episod_config = config_lib.EpisodeDescriptionConfig(args=args)

    use_bilevel_ontology_list = [False] * len(sources)
    use_dag_ontology_list = [False] * len(sources)
    if episod_config.num_ways:
        if len(sources) > 1:
            raise ValueError('For fixed episodes, not tested yet on > 1 dataset')
    else:
        # Enable ontology aware sampling for Omniglot and ImageNet.
        if 'omniglot' in sources:
            use_bilevel_ontology_list[sources.index('omniglot')] = True
        if 'ilsvrc_2012' in sources:
            use_dag_ontology_list[sources.index('ilsvrc_2012')] = True

    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list
    episod_config.use_dag_ontology_list = use_dag_ontology_list

    all_dataset_specs = []
    for dataset_name in sources:
        dataset_records_path = os.path.join(data_config.path, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)
    return all_dataset_specs, data_config, episod_config


def get_dataloader(args: argparse.Namespace,
                   sources: argparse.Namespace,
                   batch_size: int,
                   split: Split,
                   world_size: int,
                   version: str,
                   episodic: bool):
    all_dataset_specs, data_config, episod_config = get_dataspecs(args, sources)
    num_classes = sum([len(d_spec.get_classes(split=split)) for d_spec in all_dataset_specs])

    if version == 'pytorch':
        pipeline_fn = torch_pipeline.make_episode_pipeline if episodic else torch_pipeline.make_batch_pipeline
        dataset = pipeline_fn(dataset_spec_list=all_dataset_specs,
                              data_config=data_config,
                              split=split,
                              episode_descr_config=episod_config)

        worker_init_fn = partial(worker_init_fn_, seed=args.seed)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=int(batch_size / world_size),
                                 num_workers=data_config.num_workers,
                                 worker_init_fn=worker_init_fn)
    elif version == 'tf':
        import gin
        import tensorflow as tf
        from .original_meta_dataset.data import pipeline as tf_pipeline
        tf.compat.v1.disable_eager_execution()
        pipeline_fn = tf_pipeline.make_one_source_episode_pipeline if episodic else tf_pipeline.make_one_source_batch_pipeline
        GIN_FILE_PATH = 'src/datasets/original_meta_dataset/learn/gin/setups/data_config.gin'
        # 2
        gin.parse_config_file(GIN_FILE_PATH)
        dataset = pipeline_fn(
            dataset_spec=all_dataset_specs[0],
            use_dag_ontology=episod_config.use_dag_ontology_list[0],
            use_bilevel_ontology=episod_config.use_bilevel_ontology_list[0],
            episode_descr_config=episod_config,
            split=split,
            batch_size=int(batch_size / world_size),
            image_size=84,
            shuffle_buffer_size=1000)
        iterator = dataset.make_one_shot_iterator().get_next()
        to_torch_labels = lambda a: torch.from_numpy(a).long()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        def to_torch_imgs(img):
            img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
            img -= mean
            img /= std
            return img
        # 2
        session = tf.compat.v1.Session()

        def infinite_loader():
            while True:
                (e, source_id) = session.run(iterator)
                if episodic:
                    yield (to_torch_imgs(e[0]).unsqueeze(0), to_torch_imgs(e[3]).unsqueeze(0),
                           to_torch_labels(e[1]).unsqueeze(0), to_torch_labels(e[4]).unsqueeze(0))
                else:
                    yield (to_torch_imgs(e[0]), to_torch_labels(e[1]))
        data_loader = infinite_loader()
    else:
        raise ValueError(f"Wrong loader version, got {version}, \
                           expected to be in ['pytorch', 'tf']")

    return data_loader, num_classes
