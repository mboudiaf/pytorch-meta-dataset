import torchvision
from . import reader
from . import sampling
import torch
from .transform import get_transforms
import numpy as np
from .utils import Split, cycle_
from typing import List, Union
from .dataset_spec import HierarchicalDatasetSpecification as HDS
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import DatasetSpecification as DS
from .config import EpisodeDescriptionConfig, DataConfig
from tfrecord.torch.dataset import TFRecordDataset
from .sampling import EpisodeDescriptionSampler


def make_episode_pipeline(dataset_spec_list: List[Union[HDS, BDS, DS]],
                          split: Split,
                          episode_descr_config: EpisodeDescriptionConfig,
                          data_config: DataConfig,
                          ignore_hierarchy_probability: int = 0.0,
                          **kwargs):
    """Returns a pipeline emitting data from potentially multiples source as Episodes.

    Args:
      dataset_spec_list: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.

    Returns:
    """

    episodic_dataset_list = []
    for i in range(len(dataset_spec_list)):
        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=split,
                                       shuffle_queue_size=data_config.shuffle_queue_size,
                                       offset=0)
        class_datasets = episode_reader.construct_class_datasets()
        sampler = sampling.EpisodeDescriptionSampler(
            dataset_spec=episode_reader.dataset_spec,
            split=split,
            episode_descr_config=episode_descr_config,
            use_dag_hierarchy=episode_descr_config.use_dag_ontology_list[i],
            use_bilevel_hierarchy=episode_descr_config.use_bilevel_ontology_list[i],
            ignore_hierarchy_probability=ignore_hierarchy_probability)
        transforms = get_transforms(data_config, split)
        _, max_support_size, max_query_size = sampler.compute_chunk_sizes()
        episodic_dataset_list.append(EpisodicDataset(class_datasets=class_datasets,
                                                     sampler=sampler,
                                                     max_support_size=max_support_size,
                                                     max_query_size=max_query_size,
                                                     transforms=transforms))

    return ZipDataset(episodic_dataset_list)


def make_batch_pipeline(dataset_spec_list: List[Union[HDS, BDS, DS]],
                        data_config: DataConfig,
                        split: Split,
                        **kwargs):
    """Returns a pipeline emitting data from potentially multiples source as batches.

    Args:
      dataset_spec_list: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
    Returns:
    """

    offset = 0
    dataset_list = []
    for dataset_spec in dataset_spec_list:
        batch_reader = reader.Reader(dataset_spec=dataset_spec,
                                     split=split,
                                     shuffle_queue_size=data_config.shuffle_queue_size,
                                     offset=offset)

        class_datasets = batch_reader.construct_class_datasets()

        transforms = get_transforms(data_config=data_config, split=split)
        dataset = BatchDataset(class_datasets=class_datasets,
                               transforms=transforms)
        dataset_list.append(dataset)
        offset += len(class_datasets)
    dataset = ZipDataset(dataset_list)
    return dataset


class EpisodicDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 sampler: EpisodeDescriptionSampler,
                 transforms: torchvision.transforms,
                 max_support_size: int,
                 max_query_size: int):
        super(EpisodicDataset).__init__()
        self.class_datasets = class_datasets
        self.sampler = sampler
        self.transforms = transforms
        self.max_query_size = max_query_size
        self.max_support_size = max_support_size
        self.random_gen = None

    def __iter__(self):
        while True:
            episode_description = self.sampler.sample_episode_description(self.random_gen)
            support_images = []
            support_labels = []
            query_images = []
            query_labels = []
            episode_classes = list({class_ for class_, _, _ in episode_description})
            for class_id, nb_support, nb_query in episode_description:
                for _ in range(nb_support):
                    sample_dic = self.get_next(class_id)
                    support_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                for _ in range(nb_query):
                    sample_dic = self.get_next(class_id)
                    query_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                support_labels.extend([episode_classes.index(class_id)] * nb_support)
                query_labels.extend([episode_classes.index(class_id)] * nb_query)
            support_images = torch.cat(support_images, 0)
            query_images = torch.cat(query_images, 0)
            support_labels = torch.tensor(support_labels)
            query_labels = torch.tensor(query_labels)
            yield support_images, query_images, support_labels, query_labels

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        return sample_dic


class BatchDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 transforms: torchvision.transforms):
        super(BatchDataset).__init__()
        self.class_datasets = class_datasets
        self.transforms = transforms

    def __iter__(self):
        while True:
            rand_class = self.random_gen.randint(len(self.class_datasets))
            sample_dic = self.get_next(rand_class)
            transformed_image = self.transforms(sample_dic['image'])
            target = sample_dic['label'][0]
            yield transformed_image, target

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        return sample_dic


class ZipDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 dataset_list: List[EpisodicDataset]):
        self.dataset_list = dataset_list
        self.random_gen = None

    def __iter__(self):
        while True:
            rand_source = self.random_gen.randint(len(self.dataset_list))
            next_e = self.get_next(rand_source)
            yield next_e

    def get_next(self, source_id):
        try:
            dataset = next(self.dataset_list[source_id])
        except:
            self.dataset_list[source_id] = iter(self.dataset_list[source_id])
            dataset = next(self.dataset_list[source_id])
        return dataset