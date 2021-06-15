from typing import List, Union

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from . import reader
from . import sampling
from .utils import Split
from .transform import get_transforms
from .sampling import EpisodeDescriptionSampler
from .tfrecord.torch.dataset import TFRecordDataset
from .config import EpisodeDescriptionConfig, DataConfig
from .dataset_spec import DatasetSpecification as DS
from .dataset_spec import BiLevelDatasetSpecification as BDS
from .dataset_spec import HierarchicalDatasetSpecification as HDS


class EpisodicDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 sampler: EpisodeDescriptionSampler,
                 transforms: torchvision.transforms,
                 max_support_size: int,
                 max_query_size: int):
        super().__init__()

        self.class_datasets = class_datasets
        self.sampler = sampler
        self.transforms = transforms
        self.max_query_size = max_query_size
        self.max_support_size = max_support_size
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            support_images = []
            support_labels = []
            query_images = []
            query_labels = []

            episode_description = self.sampler.sample_episode_description(self.random_gen)
            episode_classes = list({class_ for class_, _, _ in episode_description})

            for class_id, nb_support, nb_query in episode_description:
                used_ids = []
                sup_added = 0
                query_added = 0

                while sup_added < nb_support:
                    sample_dic = self.get_next(class_id)

                    if sample_dic['id'] not in used_ids:
                        sample_dic = parse_record(sample_dic)
                        used_ids.append(sample_dic['id'])

                        support_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                        sup_added += 1

                while query_added < nb_query:
                    sample_dic = self.get_next(class_id)

                    if sample_dic['id'] not in used_ids:
                        sample_dic = parse_record(sample_dic)

                        used_ids.append(sample_dic['id'])
                        query_images.append(self.transforms(sample_dic['image']).unsqueeze(0))

                        query_added += 1

                # print(f"Class {class_id} contains duplicate: {contains_duplicates(used_ids)}")
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
        except (StopIteration, TypeError) as e:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])
        return sample_dic


class BatchDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 class_datasets: List[TFRecordDataset],
                 transforms: torchvision.transforms):
        super().__init__()

        self.class_datasets = class_datasets
        self.transforms = transforms
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            rand_class = self.random_gen.randint(len(self.class_datasets))
            sample_dic = self.get_next(rand_class)
            sample_dic = parse_record(sample_dic)
            transformed_image = self.transforms(sample_dic['image'])
            target = sample_dic['label'][0]
            yield transformed_image, target

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except (StopIteration, TypeError) as e:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])

        return sample_dic


class ZipDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 dataset_list: Union[List[EpisodicDataset], List[BatchDataset]]):
        self.dataset_list = dataset_list
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            rand_source = self.random_gen.randint(len(self.dataset_list))
            next_e = self.get_next(rand_source)

            yield next_e

    def get_next(self, source_id):
        try:
            dataset = next(self.dataset_list[source_id])
        except (StopIteration, TypeError) as e:
            self.dataset_list[source_id] = iter(self.dataset_list[source_id])
            dataset = next(self.dataset_list[source_id])

        return dataset


def worker_init_fn_(worker_id, seed):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    random_gen = np.random.RandomState(seed + worker_id)

    dataset.random_gen = random_gen
    for source_dataset in dataset.dataset_list:
        source_dataset.random_gen = random_gen

        for class_dataset in source_dataset.class_datasets:
            class_dataset.random_gen = random_gen


def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def parse_record(feat_dic):
    # typename_mapping = {
    #     "byte": "bytes_list",
    #     "float": "float_list",
    #     "int": "int64_list"
    # }
    # get BGR image from bytes
    image = cv2.imdecode(feat_dic["image"], -1)
    # from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    feat_dic["image"] = image
    return feat_dic


def contains_duplicates(X):
    return len(np.unique(X)) != len(X)


def make_episode_pipeline(dataset_spec: Union[HDS, BDS, DS],
                          split: Split,
                          episode_descr_config: EpisodeDescriptionConfig,
                          data_config: DataConfig,
                          ignore_hierarchy_probability: float = 0.0,
                          **kwargs) -> Dataset:
    """Returns a pipeline emitting data from potentially multiples source as Episodes.

    Args:
      dataset_spec: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.

    Returns:
    """

    episodic_dataset_list = []
    offset = 0
    episode_reader = reader.Reader(dataset_spec=dataset_spec,
                                   split=split,
                                   shuffle=data_config.shuffle,
                                   offset=offset)

    class_datasets = episode_reader.construct_class_datasets()
    sampler = sampling.EpisodeDescriptionSampler(
        dataset_spec=episode_reader.dataset_spec,
        split=split,
        episode_descr_config=episode_descr_config,
        use_dag_hierarchy=episode_descr_config.use_dag_ontology,
        use_bilevel_hierarchy=episode_descr_config.use_bilevel_ontology,
        ignore_hierarchy_probability=ignore_hierarchy_probability)

    transforms = get_transforms(data_config, split)

    _, max_support_size, max_query_size = sampler.compute_chunk_sizes()
    episodic_dataset_list.append(EpisodicDataset(class_datasets=class_datasets,
                                                 sampler=sampler,
                                                 max_support_size=max_support_size,
                                                 max_query_size=max_query_size,
                                                 transforms=transforms))
    offset += len(class_datasets)

    return ZipDataset(episodic_dataset_list)


def make_batch_pipeline(dataset_spec: Union[HDS, BDS, DS],
                        data_config: DataConfig,
                        split: Split,
                        **kwargs) -> Dataset:
    """Returns a pipeline emitting data from potentially multiples source as batches.

    Args:
      dataset_spec: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
    Returns:
    """

    offset = 0
    dataset_list: List[BatchDataset] = []
    batch_reader = reader.Reader(dataset_spec=dataset_spec,
                                 split=split,
                                 shuffle=data_config.shuffle,
                                 offset=offset)

    class_datasets = batch_reader.construct_class_datasets()

    transforms = get_transforms(data_config=data_config, split=split)
    dataset = BatchDataset(class_datasets=class_datasets,
                           transforms=transforms)
    dataset_list.append(dataset)

    return ZipDataset(dataset_list)
