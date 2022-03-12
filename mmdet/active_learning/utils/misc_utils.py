import inspect
import logging
from collections import defaultdict
from typing import List, Any

from mmdet.datasets import DATASETS

from .dist_utils import gather_and_compare


al_config = None
train_config = None


def register_al_config(config):
    global al_config
    al_config = config


def get_al_config():
    return al_config


def register_train_config(config):
    global train_config
    train_config = config


def get_train_config():
    return train_config


def assert_dataset_has_attr(dataset, attr):
    if not hasattr(dataset, attr):
        raise NotImplementedError(
            f"Dataset of type {type(dataset)} has no attribute {attr}, which "
            f"is required for active learning. See 'BaseLMDBDataset' for an "
            f"example of such attribute / function."
        )


def assert_dataset_has_parameter(dataset_cfg, param_key):
    """
    Assert whether a dataset class accepts the given parameter (keyword argument)
    in its __init__ function by recursively checking the dataset class and its
    parent classes.
    """
    dataset_class = DATASETS.get(dataset_cfg.type)
    if dataset_class is None:
        raise KeyError(
            f"{dataset_cfg.type} is not in the {DATASETS.name} registry"
        )

    def recursion(cls):
        if cls is object:
            return False

        inspected = inspect.getfullargspec(cls.__init__)
        args = inspected.args[1:]
        has_param = [param_key in args]

        # Only check for its parent if `kwargs` is presented
        if inspected.varkw is not None:
            for parent_cls in cls.__bases__:
                has_param.append(recursion(parent_cls))

        return any(has_param)

    has_param = recursion(dataset_class)
    if not has_param:
        raise KeyError(
            f"Dataset class '{dataset_cfg.type}' does not accept '{param_key}' "
            f"as a keyword argument in its __init__ function"
        )


def assert_has_config(cfg, path):
    cfg_names = path.split(".")
    sub_cfg = cfg

    for cfg_name in cfg_names:
        if sub_cfg is None:
            break
        sub_cfg = sub_cfg.get(cfg_name)

    if sub_cfg is None:
        raise NotImplementedError(
            f"Config has no path 'cfg.{path}', which is required for active "
            f"learning. See 'configs/_active_learning_/"
            f"deeplabv3plus_r50-d8_512x512_40k_voc12aug.py' for an example."
        )


def get_unique_indices(
    x: List[Any],
    device: str,
    buffer_size: int = None,
):
    """
    Get indices of unique values in a given list. If a value has a duplicate
    in the list, only its first occurence is recorded. Elements in the list
    should be all hash-able.
    """
    # Use a dict to count occurences
    count = defaultdict(list)
    for i, obj in enumerate(x):
        count[obj].append(i)

    unique_indices = []
    for indices in count.values():
        unique_indices.append(indices[0])
    gather_and_compare(unique_indices, device, buffer_size=buffer_size)
    return unique_indices


class DisableLogger:
    def __enter__(self):
       logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)
