import enum
import torch
import numpy as np


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


class Split(enum.Enum):
    """The possible data splits."""
    TRAIN = 0
    VALID = 1
    TEST = 2
