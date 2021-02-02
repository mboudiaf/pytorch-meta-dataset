from . import pipeline
from torch.utils.data import DataLoader
from .utils import Split
from typing import List, Union
from .dataset_spec import HierarchicalDatasetSpecification as HDS
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import DatasetSpecification as DS
from .config import EpisodeDescriptionConfig, DataConfig


def get_episodic_dataloader(split: Split,
                            all_dataset_specs: List[Union[HDS, BDS, DS]],
                            episod_config: EpisodeDescriptionConfig,
                            data_config: DataConfig):
    dataset = pipeline.make_episode_pipeline(dataset_spec_list=all_dataset_specs,
                                             split=split,
                                             data_config=data_config,
                                             episode_descr_config=episod_config)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             num_workers=data_config.num_workers)
    return data_loader


def get_batch_dataloader(split: Split,
                         all_dataset_specs: List[Union[HDS, BDS, DS]],
                         data_config: DataConfig):
    dataset = pipeline.make_batch_pipeline(dataset_spec_list=all_dataset_specs,
                                           data_config=data_config,
                                           split=split)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=data_config.batch_size,
                             num_workers=data_config.num_workers)

    return data_loader