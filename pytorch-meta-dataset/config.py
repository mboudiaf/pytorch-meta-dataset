from sacred import Ingredient
from typing import List, Union

dataset_ingredient = Ingredient('data')


@dataset_ingredient.config
def config():
    # Data config
    image_size = 84  # noqa: F841
    read_buffer_size_bytes = None  # noqa: F841
    sources = ['ilsvrc_2012']  # noqa: F841
    batch_size = 256  # noqa: F841
    episodic = False  # noqa: F841
    path = 'data'  # noqa: F841
    split = 'train'  # noqa: F841
    num_workers = 2  # noqa: F841
    train_transforms = ['random_resized_crop', 'jitter', 'random_flip', 'normalize']  # noqa: F841
    test_transforms = ['resize', 'center_crop', 'normalize']  # noqa: F841

    num_ways = None  # noqa: F841
    num_support = None  # noqa: F841
    num_query = None  # noqa: F841
    min_ways = 5  # noqa: F841
    max_ways_upper_bound = 50  # noqa: F841
    max_num_query = 10  # noqa: F841
    max_support_set_size = 100  # noqa: F841
    max_support_size_contrib_per_class = 10  # noqa: F841
    min_log_weight = -0.69314718055994529  # noqa: F841
    max_log_weight = 0.69314718055994529  # noqa: F841
    ignore_dag_ontology = False  # noqa: F841
    ignore_bilevel_ontology = False  # noqa: F841
    ignore_hierarchy_probability = 0  # noqa: F841
    min_examples_in_class = 0  # noqa: F841
    num_unique_episodes = 0  # noqa: F841


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""
    @dataset_ingredient.capture
    def __init__(
            self,
            image_size: int,
            batch_size: int,
            sources: List[str],
            path: str,
            num_workers: int,
            train_transforms: List[str],
            test_transforms: List[str]
    ):
        """Initialize a DataConfig.

        Args:
            image_size: An integer, the desired height for the images output by the
                data pipeline. Images are squared and have 3 channels (RGB), so each
                image will have shape [image_size, image_size, 3],
            shuffle_buffer_size: An integer, the size of the example buffer in the
                tf.data.Dataset.shuffle operations (there is typically one shuffle per
                class in the episodic setting, one per dataset in the batch setting).
                Classes with fewer examples as this number are shuffled in-memory.
            read_buffer_size_bytes: An integer, the size (in bytes) of the read buffer
                for each tf.data.TFRecordDataset (there is typically one for each class
                of each dataset).
            num_prefetch: int, the number of examples to prefetch for each class of
                each dataset. Prefetching occurs just after the class-specific Dataset
                object is constructed. If < 1, no prefetching occurs.
        """

        # General info
        self.sources = sources
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transforms and augmentations
        self.image_size = image_size
        self.test_transforms = test_transforms
        self.train_transforms = train_transforms


class EpisodeDescriptionConfig(object):
    """Configuration options for episode characteristics."""
    @dataset_ingredient.capture
    def __init__(self,
                 num_ways: Union[None, int],
                 num_support: Union[None, int],
                 num_query: Union[None, int],
                 min_ways: int,
                 max_ways_upper_bound: int,
                 max_num_query: int,
                 max_support_set_size: int,
                 max_support_size_contrib_per_class: int,
                 min_log_weight: int,
                 max_log_weight: int,
                 ignore_dag_ontology: bool,
                 ignore_bilevel_ontology: bool,
                 ignore_hierarchy_probability: bool,
                 min_examples_in_class: int,
                 num_unique_episodes: int):
        """Initialize a EpisodeDescriptionConfig.

        This is used in sampling.py in Trainer and in EpisodeDescriptionSampler to
        determine the parameters of episode creation relating to the ways and shots.

        Args:
            num_ways: Integer, fixes the number of classes ("ways") to be used in each
                episode. None leads to variable way.
            num_support: An integer, a tuple of two integers, or None. In the first
                case, the number of examples per class in the support set. In the
                second case, the range from which to sample the number of examples per
                class in the support set. Both of these cases would yield class-balanced
                episodes, i.e. all classes have the same number of support examples.
                Finally, if None, the number of support examples will vary both within
                each episode (introducing class imbalance) and across episodes.
            num_query: Integer, fixes the number of examples for each class in the
                query set.
            min_ways: Integer, the minimum value when sampling ways.
            max_ways_upper_bound: Integer, the maximum value when sampling ways. Note
                that the number of available classes acts as another upper bound.
            max_num_query: Integer, the maximum number of query examples per class.
            max_support_set_size: Integer, the maximum size for the support set.
            max_support_size_contrib_per_class: Integer, the maximum contribution for
                any given class to the support set size.
            min_log_weight: Float, the minimum log-weight to give to any particular
                class when determining the number of support examples per class.
            max_log_weight: Float, the maximum log-weight to give to any particular
                class.
            ignore_dag_ontology: Whether to ignore ImageNet's DAG ontology when
                sampling classes from it. This has no effect if ImageNet is not part of
                the benchmark.
            ignore_bilevel_ontology: Whether to ignore Omniglot's DAG ontology when
                sampling classes from it. This has no effect if Omniglot is not part of
                the benchmark.
            ignore_hierarchy_probability: Float, if using a hierarchy, this flag makes
                the sampler ignore the hierarchy for this proportion of episodes and
                instead sample categories uniformly.
            simclr_episode_fraction: Float, fraction of episodes that will be
                converted to SimCLR Episodes as described in the CrossTransformers
                paper.
            min_examples_in_class: An integer, the minimum number of examples that a
                class has to contain to be considered. All classes with fewer examples
                will be ignored. 0 means no classes are ignored, so having classes with
                no examples may trigger errors later. For variable shots, a value of 2
                makes it sure that there are at least one support and one query samples.
                For fixed shots, you could set it to `num_support + num_query`.
            num_unique_episodes: An integer, the number of unique episodes to use.
                If set to x > 0, x number of episodes are pre-generated, and repeatedly
                iterated over. This is also helpful when running on TPUs as it avoids
                the use of tf.data.Dataset.from_generator. If set to x = 0, no such
                upper bound on number of unique episodes is set. Note that this is the
                number of unique episodes _for each source dataset_, not total unique
                episodes.

        Raises:
            RuntimeError: if incompatible arguments are passed.
        """
        arg_groups = {
                'num_ways': (num_ways, ('min_ways', 'max_ways_upper_bound'), (min_ways, max_ways_upper_bound)),
                'num_query': (num_query, ('max_num_query',), (max_num_query,)),
                'num_support':
                        (num_support,
                        ('max_support_set_size', 'max_support_size_contrib_per_class',
                         'min_log_weight', 'max_log_weight'),
                        (max_support_set_size, max_support_size_contrib_per_class,
                         min_log_weight, max_log_weight)),
        }

        for first_arg_name, values in arg_groups.items():
            first_arg, required_arg_names, required_args = values
            if ((first_arg is None) and any(arg is None for arg in required_args)):
                # Get name of the nones
                none_arg_names = [
                        name for var, name in zip(required_args, required_arg_names)
                        if var is None
                ]
                raise RuntimeError(
                        'The following arguments: %s can not be None, since %s is None. '
                        'Arguments can be set up with gin, for instance by providing '
                        '`--gin_file=learn/gin/setups/data_config.gin` or calling '
                        '`gin.parse_config_file(...)` in the code. Please ensure the '
                        'following gin arguments of EpisodeDescriptionConfig are set: '
                        '%s' % (none_arg_names, first_arg_name, none_arg_names))

        self.num_ways = num_ways
        self.num_support = num_support
        self.num_query = num_query
        self.min_ways = min_ways
        self.max_ways_upper_bound = max_ways_upper_bound
        self.max_num_query = max_num_query
        self.max_support_set_size = max_support_set_size
        self.max_support_size_contrib_per_class = max_support_size_contrib_per_class
        self.min_log_weight = min_log_weight
        self.max_log_weight = max_log_weight
        self.ignore_dag_ontology = ignore_dag_ontology
        self.ignore_bilevel_ontology = ignore_bilevel_ontology
        self.ignore_hierarchy_probability = ignore_hierarchy_probability
        self.min_examples_in_class = min_examples_in_class
        self.num_unique_episodes = num_unique_episodes

    @property
    def max_ways(self):
        """Returns the way (maximum way if variable) of the episode."""
        return self.num_ways or self.max_ways_upper_bound
