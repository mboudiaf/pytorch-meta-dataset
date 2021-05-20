import enum


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
