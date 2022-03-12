from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict

import torch
from mmcv.utils import print_log
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# AL
from ..master_trainer import MasterTrainer
from ..utils.dist_utils import synchronize_weights
from ..utils.registry import register
from ..utils.misc_utils import assert_dataset_has_parameter, DisableLogger

# mmdet
from mmdet.core import DistEvalHook, EvalHook
from mmdet.models import build_detector
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
)
from mmdet.utils import get_root_logger
from mmdet.utils.comm import is_main_process, synchronize


@register("trainer")
class BaseObjectDetectionTrainer:
    def __init__(
        self,
        master_trainer: MasterTrainer,
        log_filename: str,
    ):
        self.master_trainer = master_trainer
        self.cfg = self.master_trainer.train_config
        self.log_filename = log_filename

        # Pre-initialize
        self.pre_initialize()
        self.trained = False  # whether `train_and_validate` has been called

    def pre_initialize(self):
        """
        Main interface to pre-initialize everything needed to build the
        runner, including:
        1. Model (including distributed training)
        2. Datasets and dataloaders
        3. Optimizer

        In this function, the initialization order remains the same as
        in mmdet training script. Overwrite this function if a custom
        initialization order is desired.

        After pre-initialization, the runner initialization will be
        invoked.
        """
        self.initialize_models()
        self.initialize_dataloaders()
        self.parallelize()
        self.initialize_optimizers()

        # Build runner
        self.initialize_mmdet_runner()

    def initialize_models(self, verbose: bool = False):
        """
        Initialize model(s).
        """
        self.model = build_detector(
            self.cfg.model,
            train_cfg=self.cfg.get("train_cfg"),
            test_cfg=self.cfg.get("test_cfg"),
        )
        with DisableLogger():
            self.model.init_weights()
        if verbose:
            print_log(self.model, logger=get_root_logger())

    def initialize_dataloaders(self):
        """
        Initialize datasets and dataloaders
        """
        cfg = self.cfg
        assert len(cfg.workflow) == 1

        # Initialize train dataset
        self.train_dataset = build_dataset(
            cfg.data.train,
            default_args=dict(
                data_pool=self.master_trainer.policy.labeled_pool,
                debugging=self.master_trainer.debugging,
            ),
        )
        self.model.CLASSES = self.train_dataset.CLASSES

        # Initialize train dataloader
        self.train_dataloader = build_dataloader(
            self.train_dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=self.cfg.distributed,
            seed=cfg.seed,
            runner_type=cfg.runner['type'],
            persistent_workers=cfg.data.get('persistent_workers', False),
        )

        # Initialize validation dataset and dataloader
        assert_dataset_has_parameter(cfg.data.val, "debugging")
        self.val_dataset = build_dataset(
            cfg.data.val,
            default_args=dict(
                test_mode=True,
                debugging=self.master_trainer.debugging,
            ),
        )
        self.val_dataloader = build_dataloader(
            self.val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=self.cfg.distributed,
            shuffle=False,
        )

    def parallelize(self):
        """
        Set up distributed training.
        """
        if self.cfg.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            self.model = MMDistributedDataParallel(
                self.model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model = MMDataParallel(
                self.model.cuda(self.cfg.gpu_ids[0]),
                device_ids=self.cfg.gpu_ids,
            )

        # Save model parameters to be used to reset model
        if is_main_process():
            self.model_state_dict = deepcopy(self.model.state_dict())
        synchronize()

    def initialize_optimizers(self):
        """
        Initialize optimizer(s).
        """
        self.optimizer = build_optimizer(self.model, self.cfg.optimizer)

    def initialize_mmdet_runner(self):
        cfg = deepcopy(self.cfg)  # need to copy since it will be modified in-place
        # Build runner
        self.runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=self.model,
                optimizer=self.optimizer,
                work_dir=cfg.work_dir,
                logger=get_root_logger(),
                meta=None,
            ),
        )
        self.runner.meta = dict()

        # An ugly walkaround to make the .log and .log.json filenames the same
        self.runner.timestamp = self.log_filename

        # FP16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=self.cfg.distributed)
        elif self.cfg.distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        # Register hooks
        self.runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get("momentum_config", None),
            custom_hooks_config=cfg.get('custom_hooks', None),
        )

        if self.cfg.distributed and isinstance(self.runner, EpochBasedRunner):
            self.runner.register_hook(DistSamplerSeedHook())

        # Register eval hooks
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if self.cfg.distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        self.runner.register_hook(
            eval_hook(self.val_dataloader, **eval_cfg),
            priority='LOW',
        )

    def reset(self, reset_model: bool = True):
        """
        Reset data loaders, model, optimizer and learning rate scheduler.
        """
        self.trained = False

        # Reset dataloaders
        print_log("Resetting dataloaders...", logger=get_root_logger())
        self.initialize_dataloaders()

        # Reset model; note that we cannot just re-initialize another model
        # since it might lead to memory leak; instead, we need to reset the
        # entire model parameters in-place and then re-load pretrained
        # parts
        if reset_model:
            # Reset model on the master process, save it to local and re-load
            # it on other processes to ensure synchronization
            print_log("Resetting model...", logger=get_root_logger())
            if is_main_process():
                state_dict = deepcopy(self.model_state_dict)
                self.model.load_state_dict(state_dict)

            # Synchronize weights
            synchronize_weights(
                model=self.model,
                device=self.cfg.device,
            )

        else:
            print_log(f"Model {type(self.model)} is not reset.", logger=get_root_logger())

        # Reset optimizer
        print_log("Resetting optimizer(s)...", logger=get_root_logger())
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups[0]["lr"] = self.cfg.optimizer.lr

        # Re-initialize runner
        print_log("Resetting mmdet runner...", logger=get_root_logger())
        self.initialize_mmdet_runner()

    def train_and_validate(self):
        if self.trained:
            raise RuntimeError(
                "Attempting to train and validate the second time when "
                "the trainer has not been reset!"
            )
        self.runner.run([self.train_dataloader], self.cfg.workflow)
        self.trained = True

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dict to save.
        """
        model = self.model.module if self.cfg.distributed else self.model
        state_dict = {"det_model": model.state_dict()}
        return state_dict
