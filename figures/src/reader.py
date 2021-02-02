import cv2
from tfrecord.torch.dataset import TFRecordDataset
import os
import torch


class Reader(object):
    """Class reading data from one source and assembling examples.

    Specifically, it holds part of a tf.data pipeline (the source-specific part),
    that reads data from TFRecords and assembles examples from them.
    """

    def __init__(self,
                 dataset_spec,
                 data_config,
                 split,
                 num_prefetch,
                 num_to_take=-1,
                 num_unique_episodes=0):
        """Initializes a Reader from a source.

        The source is identified by dataset_spec and split.

        Args:
          dataset_spec: DatasetSpecification, dataset specification.
          split: A learning_spec.Split object identifying the source split.
          shuffle_buffer_size: An integer, the shuffle buffer size for each Dataset
            object. If 0, no shuffling operation will happen.
          read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
          num_prefetch: int, the number of examples to prefetch for each class of
            each dataset. Prefetching occurs just after the class-specific Dataset
            object is constructed. If < 1, no prefetching occurs.
          num_to_take: Optional, an int specifying a number of elements to pick from
            each tfrecord. If specified, the available images of each class will be
            restricted to that int. By default (-1) no restriction is applied and
            all data is used.
          num_unique_episodes: Optional, an int specifying the number of unique
            episodes to use. If set to x > 0, x number of episodes are pre-generated
            and repeatedly iterated over. This is also helpful when running on
            TPUs as it avoids the use of tf.data.Dataset.from_generator. If set to
            x = 0, no such upper bound on number of unique episodes is set.
        """
        self.split = split
        self.shuffle_buffer_size = data_config.shuffle_buffer_size
        self.read_buffer_size_bytes = data_config.read_buffer_size_bytes
        self.num_prefetch = num_prefetch
        self.num_to_take = num_to_take
        self.num_unique_episodes = num_unique_episodes

        self.base_path = self.dataset_spec.path
        self.class_set = self.dataset_spec.get_classes(self.split)
        self.num_classes = len(self.class_set)

    def construct_class_datasets(self,
                                 pool=None,
                                 repeat=True,
                                 shuffle=True,
                                 shuffle_seed=None):
        """Constructs the list of class datasets.

        Args:
          pool: A string (optional) indicating whether to only read examples from a
            given example-level split.
          repeat: Boolean indicating whether each of the class datasets should be
            repeated (to provide an infinite stream) or not.
          shuffle: Boolean indicating whether each of the class datasets should be
            shuffled or not.
          shuffle_seed: Optional, an int containing the seed passed to
            tf.data.Dataset.shuffle.

        Returns:
          class_datasets: list of tf.data.Dataset, one for each class.
        """
        file_pattern = self.dataset_spec.file_pattern
        # We construct one dataset object per class. Each dataset outputs a stream
        # of `(example_string, dataset_id)` tuples.
        class_datasets = []
        for dataset_id in range(self.num_classes):
            class_id = self.class_set[dataset_id]  # noqa: E111
            if file_pattern.startswith('{}_{}'):
                # TODO(lamblinp): Add support for sharded files if needed.
                raise NotImplementedError('Sharded files are not supported yet. '  # noqa: E111
                                          'The code expects one dataset per class.')
            elif file_pattern.startswith('{}'):
                filename = os.path.join(self.base_path, file_pattern.format(class_id))  # noqa: E111
            else:
                raise ValueError('Unsupported file_pattern in DatasetSpec: %s. '  # noqa: E111
                                 'Expected something starting with "{}" or "{}_{}".' %
                                 file_pattern)
            description = {"image": "byte", "label": "int"}
            index_path = None

            def decode_image(features):
                # get BGR image from bytes
                features["image"] = torch.tensor(cv2.imdecode(features["image"], -1)).permute(2, 0, 1) / 255
                return features

            dataset = TFRecordDataset(data_path=filename,
                                      index_path=index_path,
                                      description=description,
                                      transform=decode_image,)
                                      # shuffle_queue_size=self.shuffle_buffer_size)

            class_datasets.append(dataset)

        assert len(class_datasets) == self.num_classes
        return class_datasets


def add_offset_to_target(example_strings, targets, offset):
    """Adds offset to the targets.

    This function is intented to be passed to tf.data.Dataset.map.

    Args:
    example_strings: 1-D Tensor of dtype str, Example protocol buffers.
    targets: 1-D Tensor of dtype int, targets representing the absolute class
      IDs.
    offset: int, optional, number to add to class IDs to get targets.

    Returns:
    example_strings, labels: Tensors, a batch of examples and labels.
    """
    labels = targets + offset
    return (example_strings, labels)