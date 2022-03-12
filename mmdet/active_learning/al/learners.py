"""
Implementation of learners for image-level active learning.

This family of classes defines how to obtain prediction probabilities
(forward passes) or other data to feed into the chosen active learning
policy.

Disclaimer:
1. These are not really "learners". I just cannot think of any better
name.
2. Although `forward` is the main protocol in this family of classes,
it does not have anything to do with the `torch.nn.Module`'s forward
function.
"""


from copy import copy
from typing import (
    List,
    Generator,
    Iterable,
    TYPE_CHECKING,
)

import torch
from torch import Tensor as T
import numpy as np
from mmcv.utils import print_log

# mmdet
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.utils import get_root_logger
from mmdet.utils.logger import ProgressBar

# AL
from ..utils.misc_utils import assert_has_config
from ..utils.registry import register
if TYPE_CHECKING:
    from .policies import PoolManager
    from ..master_trainer import MasterTrainer


class BaseOutputType:
    def __init__(
        self,
        keys: List[str],
        preds: List[List[np.ndarray]],
        **kwargs,
    ):
        self.keys = keys
        self.preds = preds
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseLearner:
    """Base class for all image-level learners.

    Args:
        batch_size_multiplier (float): the batch size of learner is equal to
            training batch size multiplied by this number.
        subset_size (int): subset size of the current unlabeled pool to sample
            from. If set to `None`, sample from the entire unlabeled pool.
        debugging (bool): if the debugging mode is on, only forward a small
            subset of available images.
    """
    def __init__(
        self,
        master_trainer: "MasterTrainer",
        batch_size_multiplier: float = 1.0,
        subset_size: int = None,
        debugging: bool = False,
    ):
        self.master_trainer = master_trainer
        self.cfg = self.master_trainer.train_config
        self.batch_size_multiplier = batch_size_multiplier
        self.subset_size = subset_size
        self.debugging = debugging

    def initialize_dataset(self, data_pool: "PoolManager"):
        assert_has_config(self.cfg, path="data.learner")
        dataset = build_dataset(
            self.cfg.data.learner,
            default_args=dict(
                data_pool=data_pool,
                debugging=self.master_trainer.debugging,
            ),
        )
        return dataset

    def initialize_dataloader(self, data_pool: "PoolManager"):
        dataset = self.initialize_dataset(data_pool=data_pool)
        dataloader = build_dataloader(
            dataset,
            round(self.cfg.data.samples_per_gpu * self.batch_size_multiplier),
            self.cfg.data.workers_per_gpu,
            num_gpus=len(self.cfg.gpu_ids),
            dist=self.cfg.distributed,
            seed=self.cfg.seed,
            drop_last=False,
            runner_type=self.cfg.runner['type'],
        )
        return dataloader

    def forward(
        self,
        model,
        data_pool: "PoolManager",
    ) -> Iterable[BaseOutputType]:
        """
        Prepare the learner iterator. We dynamically pass the model
        here because model might be changed during active learning.
        """
        self.model = model.module if self.cfg.distributed else model

        # This function returns an iterable; since we want the
        # initialized dataloader to be "passed" to the iterable,
        # we will in fact create a shallow copy of the current object
        dataloader = self.initialize_dataloader(data_pool=data_pool)
        iterable = copy(self)
        iterable.dataloader = dataloader

        return iterable

    @torch.no_grad()
    def process_one_batch(
        self,
        images: T,
        image_ids: List[str],
        image_metas: List[dict],
    ) -> BaseOutputType:
        """
        Process one batch of data to generate outputs. Subclasses must
        implement this function, which returns the processed batch.
        """
        raise NotImplementedError

    @torch.no_grad()
    def __iter__(self) -> Generator[BaseOutputType, None, None]:
        self.model.eval()
        tot_size = 0
        prog_bar = ProgressBar(
            self.dataloader,
            mininterval=5,
        )

        for i, data in enumerate(self.dataloader):
            image_metas = data["img_metas"].data[0]

            # Stop if reached desired subset size
            tot_size += len(image_metas)
            if self.subset_size is not None and tot_size >= self.subset_size:
                print_log(
                    f"Random subset size of {self.subset_size} reached.",
                    logger=get_root_logger(),
                )
                break

            if self.debugging and i == 10:
                print_log(
                    f"[Debugging mode] {i} iterations reached. Stopping generating "
                    f"predictions...",
                    logger=get_root_logger(),
                )
                break

            image_ids = [img_meta["ori_filename"] for img_meta in image_metas]
            images = data["img"].to(self.cfg.device)

            prog_bar.update()
            yield self.process_one_batch(images, image_ids, image_metas)

        prog_bar.close()
        self.model.train()

    def step(self):
        """
        To be called at the end of every active learning step.
        """
        pass


"""
Single forward-pass learner.
"""


class SingleForwardPassOutputs(BaseOutputType):
    pass


@register("learner")
class SingleForwardPassLearner(BaseLearner):
    """
    Single forward pass learner, which perform a single forward pass through
    the model to obtain the predicted probabilities.
    """
    @torch.no_grad()
    def process_one_batch(
        self,
        images: T,
        image_ids: List[str],
        image_metas: List[dict],
    ) -> SingleForwardPassOutputs:
        """
        Process one batch of data.
        """
        _, preds = self.model.simple_test(
            img=images, img_metas=image_metas, return_probs=True,
        )  # list[list[ndarray]] (outer: image level; inner: class level)

        return SingleForwardPassOutputs(
            keys=image_ids,
            preds=preds,
        )

    def step(self):
        pass
