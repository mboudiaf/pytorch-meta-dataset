from . import reader
from . import sampling
import torch
from .transform import get_transforms
import numpy as np
from ..utils import cycle
RNG = np.random.RandomState(seed=None)


def make_episode_pipeline(dataset_spec_list,
                          split,
                          episode_descr_config,
                          data_config,
                          num_prefetch=0,
                          num_to_take=None,
                          ignore_hierarchy_probability=0.0,
                          **kwargs):
    """Returns a pipeline emitting data from one single source as Episodes.

    Args:
      dataset_spec: A DatasetSpecification object defining what to read from.
      use_dag_ontology: Whether to use source's ontology in the form of a DAG to
        sample episodes classes.
      use_bilevel_ontology: Whether to use source's bilevel ontology (consisting
        of superclasses and subclasses) to sample episode classes.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      pool: String (optional), for example-split datasets, which example split to
        use ('train', 'valid', or 'test'), used at meta-test time only.
      shuffle_buffer_size: int or None, shuffle buffer size for each Dataset.
      read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
      num_prefetch: int, the number of examples to prefetch for each class of each
        dataset. Prefetching occurs just after the class-specific Dataset object
        is constructed. If < 1, no prefetching occurs.
      image_size: int, desired image size used during decoding.
      num_to_take: Optional, an int specifying a number of elements to pick from
        each class' tfrecord. If specified, the available images of each class
        will be restricted to that int. By default no restriction is applied and
        all data is used.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.
      simclr_episode_fraction: Float, fraction of episodes that will be converted
        to SimCLR Episodes as described in the CrossTransformers paper.


    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
        episodes zipped with the ID of their data source of origin.
    """
    num_unique_episodes = episode_descr_config.num_unique_episodes

    episodic_dataset_list = []
    for i in range(len(dataset_spec_list)):
        episode_reader = reader.Reader(dataset_spec=dataset_spec_list[i],
                                       split=split,
                                       data_config=data_config,
                                       num_prefetch=num_prefetch,
                                       num_to_take=num_to_take,
                                       num_unique_episodes=num_unique_episodes)
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


def make_one_source_batch_pipeline(dataset_spec_list,
                                   data_config,
                                   split,
                                   num_prefetch=0,
                                   num_to_take=None,
                                   **kwargs):
    """Returns a pipeline emitting data from one single source as Episodes.

    Args:
      dataset_spec: A DatasetSpecification object defining what to read from.
      use_dag_ontology: Whether to use source's ontology in the form of a DAG to
        sample episodes classes.
      use_bilevel_ontology: Whether to use source's bilevel ontology (consisting
        of superclasses and subclasses) to sample episode classes.
      split: A learning_spec.Split object identifying the source (meta-)split.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      pool: String (optional), for example-split datasets, which example split to
        use ('train', 'valid', or 'test'), used at meta-test time only.
      shuffle_buffer_size: int or None, shuffle buffer size for each Dataset.
      read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
      num_prefetch: int, the number of examples to prefetch for each class of each
        dataset. Prefetching occurs just after the class-specific Dataset object
        is constructed. If < 1, no prefetching occurs.
      image_size: int, desired image size used during decoding.
      num_to_take: Optional, an int specifying a number of elements to pick from
        each class' tfrecord. If specified, the available images of each class
        will be restricted to that int. By default no restriction is applied and
        all data is used.
      ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
        the sampler ignore the hierarchy for this proportion of episodes and
        instead sample categories uniformly.
      simclr_episode_fraction: Float, fraction of episodes that will be converted
        to SimCLR Episodes as described in the CrossTransformers paper.


    Returns:
      A Dataset instance that outputs tuples of fully-assembled and decoded
        episodes zipped with the ID of their data source of origin.
    """
    if num_to_take is None:
        num_to_take = -1

    batch_reader = reader.Reader(dataset_spec=dataset_spec_list[0],
                                 split=split,
                                 data_config=data_config,
                                 num_prefetch=num_prefetch)

    class_datasets = batch_reader.construct_class_datasets()

    transforms = get_transforms(data_config=data_config, split=split)
    dataset = BatchDataset(class_datasets=class_datasets,
                           transforms=transforms)
    return dataset


class EpisodicDataset(torch.utils.data.IterableDataset):
    def __init__(self, class_datasets, sampler, transforms, max_support_size, max_query_size):
        super(EpisodicDataset).__init__()
        self.class_datasets = [iter(dataset) for dataset in class_datasets]
        self.sampler = sampler
        self.transforms = transforms
        self.max_query_size = max_query_size
        self.max_support_size = max_support_size

    def __iter__(self):
        while True:
            episode_description = self.sampler.sample_episode_description()
            support_images = []
            support_labels = []
            query_images = []
            query_labels = []
            episode_classes = list({class_ for class_, _, _ in episode_description})
            for class_id, nb_support, nb_query in episode_description:
                for _ in range(nb_support):
                    sample_dic = next(self.class_datasets[class_id])
                    support_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                for _ in range(nb_query):
                    sample_dic = next(self.class_datasets[class_id])
                    query_images.append(self.transforms(sample_dic['image']).unsqueeze(0))
                support_labels.extend([episode_classes.index(class_id)] * nb_support)
                query_labels.extend([episode_classes.index(class_id)] * nb_query)
            support_images = torch.cat(support_images, 0)
            query_images = torch.cat(query_images, 0)
            support_labels = torch.tensor(support_labels)
            query_labels = torch.tensor(query_labels)
            yield support_images, query_images, support_labels, query_labels


class BatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, class_datasets, transforms):
        super(BatchDataset).__init__()
        self.class_datasets = [cycle(dataset) for dataset in class_datasets]
        self.transforms = transforms

    def __iter__(self):
        while True:
            rand_class = RNG.randint(len(self.class_datasets))
            sample_dic = next(self.class_datasets[rand_class])
            transformed_image = self.transforms(sample_dic['image'])
            yield transformed_image, rand_class


class ZipDataset(torch.utils.data.IterableDataset):
    def __init__(self, episodic_dataset_list):
        self.episodic_dataset_list = [cycle(dataset) for dataset in episodic_dataset_list]

    def __iter__(self):
        while True:
            rand_source = RNG.randint(len(self.episodic_dataset_list))
            next_e = next(self.episodic_dataset_list[rand_source])
            yield next_e