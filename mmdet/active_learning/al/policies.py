"""
Implementations of image-level active learning policies for object detection.
"""


import pickle
from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Set,
    Union,
    Any,
    Iterator,
    Iterable,
    TYPE_CHECKING,
)

import numpy as np
from mmcv.utils import print_log

# AL
from .utils import calculate_entropy_np, get_unique_indices
from .aggregators import BaseAggregator
from .learners import (
    BaseOutputType,
    SingleForwardPassOutputs,
)

from ..utils.misc_utils import assert_dataset_has_attr
from ..utils.dist_utils import (
    broadcast_object,
    gather,
    gather_and_compare,
)
from ..utils.registry import register
if TYPE_CHECKING:
    from ..master_trainer import MasterTrainer

# mmdet
from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger
from mmdet.utils.comm import is_main_process


def to_set(x: List[Any]) -> Set[Any]:
    if isinstance(x, set):
        return x

    y = set(x)
    assert len(x) == len(y)
    return y


class PoolManager:
    """
    Active learning data pool manager, used for either labeled or unlabeled
    pool.
    """
    def __init__(
        self,
        cfg,  # mmdet config
    ):
        self.cfg = cfg
        self.pool: Set[Union[str, int]] = set()

    def verify(self):
        """
        Ensure that all processes have the same pool.
        """
        gather_and_compare(self.pool, device=self.cfg.device)

    def add_samples(
        self,
        samples: List[Union[str, int]],
    ):
        """
        Add samples to the current pool.
        """
        # Check overlap
        samples = to_set(samples)
        overlap = samples & self.pool
        if len(overlap) > 0:
            raise ValueError(
                f"[{self.__class__.__name__}] Trying to add already-existed "
                f"image(s): {list(overlap)}"
            )

        # Update and verify
        self.pool.update(samples)
        self.verify()

    def remove_samples(
        self,
        samples: List[Union[str, int]],
    ):
        """
        Remove samples from the current pool.
        """
        # Check
        samples = to_set(samples)
        non_overlap = samples - self.pool
        if len(non_overlap) > 0:
            raise ValueError(
                f"[{self.__class__.__name__}] Trying to remove non-existent "
                f"image(s): {list(non_overlap)} "
            )

        # Update
        self.pool = self.pool - samples
        self.verify()

    def get_pool(self, as_set: bool = False) -> List[Union[str, int]]:
        if as_set:
            return self.pool.copy()
        return sorted(list(self.pool))

    def __len__(self) -> int:
        """
        Returns the total number of images in the pool.
        """
        return len(self.pool)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for checkpointing.
        """
        state_dict = dict(
            pool_class=self.__class__.__name__,
            pool_data=self.get_pool(),
        )
        return state_dict


class BasePolicy:
    """
    Base class for all policies.

    Args:
        aggregator (BaseAggregator): Initialized aggregator (aggregation
            function).
        init_size (int): Initial pool size, i.e., the number of images to
            initialize for the initial labeled pool.
        step_size (int): Step size. The number of images to select at each
            active learning step.
    """
    accepted_output_types = None

    def __init__(
        self,
        master_trainer: "MasterTrainer",
        aggregator: BaseAggregator,
        init_size: int,
        step_size: int,
    ):
        self.master_trainer = master_trainer
        self.cfg = self.master_trainer.train_config
        self.aggregator = aggregator
        self.seed = self.cfg.seed

        self.init_size = init_size
        self.step_size = step_size

        # Initialize pools with samples
        self._records = {"results": []}  # records for all steps
        self.initialize()

        # Sanity check
        self.verify()

    def initialize(self):
        """
        Initialize pool managers.
        """
        # Initialize empty pools
        self.labeled_pool = PoolManager(cfg=self.cfg)
        self.unlabeled_pool = PoolManager(cfg=self.cfg)

        # Read all samples
        dataset = build_dataset(
            self.cfg.data.train,
            default_args=dict(debugging=self.master_trainer.debugging),
        )
        assert_dataset_has_attr(dataset, "get_image_ids")
        self.all_image_ids = to_set(dataset.get_image_ids())

        # Initialize unlabeled pool with all available images
        self.unlabeled_pool.add_samples(self.all_image_ids)

    @property
    def initialized(self) -> bool:
        return len(self.labeled_pool) > 0

    def reset(self):
        return self.initialize()

    def verify(self):
        # Check sample keys
        labeled_pool = self.labeled_pool.get_pool(as_set=True)
        unlabeled_pool = self.unlabeled_pool.get_pool(as_set=True)
        all_image_ids = labeled_pool | unlabeled_pool
        assert all_image_ids == self.all_image_ids

    def is_termination(self) -> bool:
        """
        Whether termination condition (in terms of budget) has been reached.
        """
        return (len(self.unlabeled_pool) == 0)

    def get_rng(self) -> np.random.RandomState:
        rng = np.random.RandomState(seed=self.seed)
        self.seed  = (self.seed + 1) % (2 ** 31)
        return rng

    def record_stats(self, step: int, mAP: float, **kwargs):
        update_dict = {
            "step": step,
            "mAP": mAP,
            "num_acquired_samples": len(self.labeled_pool),
            "num_not_acquired_samples": len(self.unlabeled_pool),
        }

        # Optional stats
        assert all(key not in update_dict for key in kwargs.keys())
        update_dict.update(kwargs)

        self._records["results"].append(update_dict)

    def get_records(self):
        records = deepcopy(self._records)
        return records

    def save_pool_data(self, save_path):
        """
        Pickle pool data and arbitrary additional data.
        """
        if is_main_process():
            # Basic data
            save_dict = {
                "labeled_pool": self.labeled_pool.state_dict(),
                "unlabeled_pool": self.unlabeled_pool.state_dict(),
            }

            # Save
            with open(save_path, "wb") as fout:
                pickle.dump(save_dict, fout, protocol=4)
            print_log(
                f"Pool data saved to {save_path}", logger=get_root_logger())

    def get_step_size(self):
        """
        Get number of images to acquire at the current step.
        """
        size = self.step_size if self.initialized else self.init_size
        step_size = min(len(self.unlabeled_pool), size)
        return step_size

    def select_random_subset(self, subset_size: int) -> List[Union[str, int]]:
        """
        Select a random subset of the unlabeled pool.
        """
        rng = self.get_rng()

        if is_main_process():
            if subset_size > len(self.unlabeled_pool):
                raise ValueError(
                    f"The number of desired images ({subset_size}) exceeds the "
                    f"number of available images ({len(self.unlabeled_pool)})"
                )

            image_ids = self.unlabeled_pool.get_pool()
            rng.shuffle(image_ids)
            selected_image_ids = image_ids[:subset_size]
            broadcast_object(selected_image_ids)
        else:
            selected_image_ids = broadcast_object()

        return selected_image_ids

    def lazy_init(self):
        """
        Do a first round of selection at random.
        """
        assert len(self.labeled_pool) == 0
        return self.select_random_subset(self.get_step_size())

    def verify_iterator(self, outputs: Iterable[BaseOutputType]):
        """
        Verify that results from the learner is an iterator.
        """
        if not isinstance(outputs, (Iterator, Iterable)):
            raise TypeError(
                f"Invalid input type. Expected an iterator or iterable, got "
                f"{type(outputs)} instead."
            )

    def verify_outputs(self, outputs_batch: BaseOutputType):
        """
        Batch-wise verify model outputs. To be called inside the `select`
        function.
        """
        if self.accepted_output_types is None:
            raise ValueError(
                f"Policy {self.__class__.__name__} does not implement "
                f"`accepted_output_types`"
            )
        if type(outputs_batch) not in self.accepted_output_types:
            raise TypeError(
                f"Invalid input type. Expected one of {self.accepted_output_types}, "
                f"got {type(outputs_batch)} instead."
            )

    def select(
        self,
        unlabeled_outputs: Iterable[BaseOutputType],
        labeled_outputs: Iterable[BaseOutputType],
    ) -> Tuple[
        List[Union[str, int]],
    ]:
        """
        Implementation of the active learning acquisition function. Given
        outputs as an iterator from the learner, this function must return a
        a list of selected images.

        This function must also take care of DDP, i.e., outputs of this function
        should be identical for all processes.

        Args:
            unlabeled_outputs (Iterable[BaseOutputType]): An iterator of any
                pre-defined output type from the corresponding learner,
                associated with the unlabeled pool.
            labeled_outputs (Iterable[BaseOutputType]): An iterator of any
                pre-defined output type from the corresponding learner,
                associated with the labeled pool.
        """
        raise NotImplementedError

    def acquire(
        self,
        unlabeled_outputs: Iterable[BaseOutputType],
        labeled_outputs: Iterable[BaseOutputType],
    ):
        """
        Perform a full acquistion step. This function also handles updating
        the labeled targets.

        Args:
            unlabeled_outputs (Iterable[BaseOutputType]): An iterator of any
                pre-defined output type from the corresponding learner,
                associated with the unlabeled pool.
            labeled_outputs (Iterable[BaseOutputType]): An iterator of any
                pre-defined output type from the corresponding learner,
                associated with the labeled pool.
        """
        if self.initialized:
            self.verify_iterator(unlabeled_outputs)  # must be an iterator
            self.verify_iterator(labeled_outputs)  # must be an iterator
            selected_image_ids = self.select(
                unlabeled_outputs=unlabeled_outputs,
                labeled_outputs=labeled_outputs,
            )

        else:
            assert unlabeled_outputs is None
            assert labeled_outputs is None
            selected_image_ids = self.lazy_init()

        self.labeled_pool.add_samples(selected_image_ids)
        self.unlabeled_pool.remove_samples(selected_image_ids)
        self.verify()

        # Log
        print_log(
            f"[{self.__class__.__name__}] Acquired {len(selected_image_ids)} "
            f"samples. Number of samples in labeled pool: "
            f"{len(self.labeled_pool)}. Number of samples in the unlabeled "
            f"pool: {len(self.unlabeled_pool)}",
            logger=get_root_logger(),
        )


@register("policy")
class RandomPolicy(BasePolicy):
    """
    Randomly select a new pool of images at each acquisition step.
    """
    accepted_output_types = [
        SingleForwardPassOutputs,
    ]

    def select(
        self,
        unlabeled_outputs: Iterable[Union[SingleForwardPassOutputs]],
        labeled_outputs: Iterable[Union[SingleForwardPassOutputs]],
    ):
        return self.select_random_subset(self.get_step_size())


@register("policy")
class MaxEntropyPolicy(BasePolicy):
    """
    Active learning by max entropy.
    """
    accepted_output_types = [
        SingleForwardPassOutputs,
    ]

    def select(
        self,
        unlabeled_outputs: Iterable[Union[SingleForwardPassOutputs]],
        labeled_outputs: Iterable[Union[SingleForwardPassOutputs]],
    ):
        num_images_to_acquire = self.get_step_size()
        keys = []
        entropy_values = []

        for outputs_batch in unlabeled_outputs:
            self.verify_outputs(outputs_batch)

            # Unpack data
            batch_keys = outputs_batch.keys
            batch_preds = outputs_batch.preds  # list of lists of arrays of (N, C)

            assert len(batch_keys) == len(batch_preds)
            batch_entropy_values = []
            for pred in batch_preds:
                # Concatenate all classes' predictions and calculate entropy
                pred = np.concatenate(pred, axis=0)  # (N, C)
                entropy = calculate_entropy_np(
                    pred, dim=1, normalized=True, assert_normalized=True,
                )  # (N,)

                # Aggregate
                entropy = self.aggregator(entropy, axis=0)  # scalar
                batch_entropy_values.append(entropy)

            # Aggregate across processes
            batch_entropy_values = np.array(batch_entropy_values)
            batch_keys, batch_entropy_values = gather(
                [batch_keys, batch_entropy_values],
                self.cfg.device,
            )
            keys.extend(sum(batch_keys, []))
            entropy_values.extend(batch_entropy_values)

        # Aggregate and sanity check
        entropy_values = np.concatenate(entropy_values, axis=0)
        assert len(keys) == len(entropy_values)

        # Remove duplicates due to DDP
        unique_indices = get_unique_indices(keys, self.cfg.device)
        keys = [keys[i] for i in unique_indices]
        entropy_values = entropy_values[unique_indices]

        # Sort and select
        idxs = np.argsort(
            -entropy_values,
            axis=0,
        )[:num_images_to_acquire]  # relative indices
        keys = [keys[i] for i in idxs]
        return keys
