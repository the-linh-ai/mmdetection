import os
import json
import shutil
import logging
from typing import TYPE_CHECKING

import yaml
import torch.nn as nn
from mmcv.utils import print_log
from mmcv.utils import Config as MMCVConfig

# AL
from .utils.misc_utils import register_al_config, register_train_config
from .utils.dist_utils import (
    gather_and_compare,
    setup_ddp,
    broadcast_object,
)
from .utils.config_utils import (
    DictConfig,
    initialize_from_config,
)

# mmdet
from mmdet.utils import collect_env, get_root_logger
from mmdet.utils.comm import is_main_process, synchronize
from mmdet.apis import set_random_seed

# Base class for annotations
if TYPE_CHECKING:
    from .trainers.base_trainer import BaseObjectDetectionTrainer
from .al.aggregators import BaseAggregator
from .al.policies import BasePolicy
from .al.learners import BaseLearner


class MasterTrainer:
    """
    Master trainer, handling almost everything end-to-end. This class
    should NOT be subclassed. Instead, if custom logic needs to be
    defined, it should be implemented as a (sub-)trainer of this
    trainer, e.g., subclass of `BaseObjectDetectionTrainer`.

    See `trainers/base_trainer.py` for the base sub-trainer class, in
    which all mentioned functions are implemented. Overwrite or
    implement additional logic if needed.
    """
    def __init__(
        self,
        al_config: DictConfig,
        train_config: MMCVConfig,
    ):
        self.config = al_config
        self.train_config = train_config
        self.debugging = self.config.general.debugging

        # Setup DDP if needed
        self._setup_ddp()

        # Get save dir
        self._get_save_dir(
            self.config.general.work_dir,
            self.config.general.sub_dir,
        )

        # Get logger
        self._get_logger()

        # Finalize mmdet config and download datasets if needed
        self._finalize_config()

        # Initialize active learners
        self._initialize_policy(
            aggregator=self.config.aggregator,
            policy=self.config.policy,
        )
        self._initialize_active_learner(active_learner=self.config.active_learner)

        # Initialize detection trainer
        self._initialize_det_trainer()

        # Save config and command
        self._save_metadata()

        # Set additional attributes
        self._set_step(0)  # step index

    def _setup_ddp(self):
        """
        Setup distributed training
        """
        # Single-gpu, non-distributed training
        if self.config.local_rank == -1:
            self.config.distributed = False
            self.config.local_rank = 0
        # Single- or multi-gpu, distributed training
        else:
            self.config.distributed = True

        setup_ddp(config=self.config)
        self.train_config.local_rank = self.config.local_rank
        self.train_config.device = self.config.device
        self.train_config.distributed = self.config.distributed
        self.train_config.distributed_world_size = self.config.distributed_world_size

        self.det_trainer = None
        self.model = None

    def _get_save_dir(self, work_dir, sub_dir):
        """
        Get save directory
        """
        if work_dir is not None:
            save_dir = os.path.join(work_dir, sub_dir)
            if is_main_process():
                assert not os.path.exists(save_dir)
                os.makedirs(save_dir, exist_ok=False)
        else:
            save_dir = None

        self.config.general.save_dir = self.train_config.work_dir = \
            self.save_dir = save_dir

    def _get_logger(self):
        """
        Get logger
        """
        if self.save_dir is None:
            log_file = None
        else:
            log_file = os.path.join(self.save_dir, "training.log")
        get_root_logger(
            log_file=log_file,
            log_level=self.train_config.log_level,
        )

        # Log environment info
        env_info_dict = collect_env()
        env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        print_log(
            "Environment info:\n" + dash_line + env_info + "\n" + dash_line,
            logger=get_root_logger(),
        )

    def _parse_custom_logic(self):
        """
        Parse custom logic; mainly for ugly workarounds
        """
        al_config = self.config
        train_config = self.train_config

        # Custom logic for AL step size
        if getattr(al_config.custom, "al_step_size", None) is not None:
            step_size = al_config.custom.al_step_size
            print_log(
                f"Overriding `init_size` and `step_size` in the AL policy with "
                f"`al_step_size={step_size}`",
                logger=get_root_logger(),
            )

            al_config.policy.policy_init_kwargs.init_size = step_size
            al_config.policy.policy_init_kwargs.step_size = step_size

        # Custom logic for optimizer type
        optimizer_type = getattr(al_config.custom, "optimizer", None)
        if optimizer_type is not None:
            train_config.optimizer.type = optimizer_type

    def _finalize_config(self):
        """
        Overwrite mmdet config
        """
        al_config = self.config
        train_config = self.train_config

        # Register configs to enable global use
        register_al_config(al_config)
        register_train_config(train_config)

        # Train batch size; note that in mmdet, validation batch size is always 1
        train_config.data.samples_per_gpu = al_config.training.batch_size
        train_config.data.workers_per_gpu = al_config.training.num_workers

        # Learning rate
        train_config.optimizer.lr = al_config.training.lr

        # Number of classes
        train_config.model.roi_head.bbox_head.num_classes = \
            al_config.training.num_classes

        # Seed
        seed = al_config.general.seed
        assert seed is not None
        self.seed = train_config.seed = seed
        set_random_seed(seed=seed, deterministic=False)

        # Other configs
        al_config.general.log_interval = train_config.log_config.interval
        train_config.gpu_ids = [0]  # for either single-gpu OR multi-gpu distributed

        # Checkpointing
        # Dont save after every epoch; used in conjunction with `max_keep_ckpts`
        train_config.checkpoint_config.interval = int(2 ** 32)
        # Always only keep the latest one, if there is; this does not affect
        # "save_best"
        train_config.checkpoint_config.max_keep_ckpts = 1
        # Don't save anything by default
        train_config.checkpoint_config.save_last = False
        train_config.evaluation.save_best = None
        train_config.evaluation.gpu_collect = True

        # No-save mode
        if self.save_dir is None:
            train_config.checkpoint_config = None
        else:
            if al_config.general.save_model:
                train_config.checkpoint_config.save_last = True  # save last
                train_config.evaluation.save_best = "bbox_mAP"  # save best

        # Debugging
        if self.debugging:
            if train_config.runner.type != "EpochBasedRunner":
                raise NotImplementedError(
                    f"Debug mode is not implemented for runner "
                    f"{train_config.runner.type}"
                )
            print_log(
                "Debug mode is enabled. Restricting number of epochs, number "
                "of active learning cycles, etc.",
                logger=get_root_logger(),
            )

            al_config.training.num_steps = 2
            train_config.runner.max_epochs = 2
            train_config.evaluation.interval = 1
            al_config.general.log_interval = train_config.log_config.interval = 5

        # Custom logic; mainly for ugly workarounds
        self._parse_custom_logic()

    @property
    def model_for_selection(self):
        """
        Get model for active learners.
        This should be overwritten, for example, if you want to use another
        model for selection, especially in ablation studies.
        """
        return self.model

    def _initialize_policy(
        self,
        aggregator: DictConfig,
        policy: DictConfig,
    ):
        """
        Initialize score aggregator and active learning policy.
        """
        # Initialize score aggregator
        print_log(f"Score aggregator:\n{aggregator}", logger=get_root_logger())
        self.aggregator: BaseAggregator = initialize_from_config(
            cfg=aggregator,
            name="aggregator",
        )

        # Initialize policy
        print_log(f"Policy:\n{policy}", logger=get_root_logger())
        self.policy: BasePolicy = initialize_from_config(
            cfg=policy,
            name="policy",
            master_trainer=self,
            aggregator=self.aggregator,
        )
        self.labeled_pool = self.policy.labeled_pool
        self.unlabeled_pool = self.policy.unlabeled_pool

    def _initialize_active_learner(
        self,
        active_learner: DictConfig,
    ):
        print_log(f"Active learner:\n{active_learner}", logger=get_root_logger())
        self.active_learner: BaseLearner = initialize_from_config(
            cfg=active_learner,
            name="learner",
            master_trainer=self,
            debugging=self.debugging,
        )

    def _save_metadata(self):
        if is_main_process() and self.save_dir is not None:
            # AL config
            save_to = os.path.realpath(
                os.path.join(self.save_dir, "al_config.yaml"))
            with open(save_to, "w") as fout:
                yaml.dump(
                    self.config.to_dict(remove_non_jsonable_objects=True),
                    fout,
                    sort_keys=False,
                )
            # mmdet config
            self.train_config.dump(
                os.path.join(self.save_dir, "train_config.py"),
            )
            # Command
            save_to = os.path.realpath(
                os.path.join(self.save_dir, "command.sh"))
            with open(save_to, "w") as fout:
                fout.write(self.config.general.command + "\n")

        # Print config
        if is_main_process():
            config_str = json.dumps(
                self.config.to_dict(remove_non_jsonable_objects=True),
                indent=2,
            )
            print_log(f"AL config:\n{config_str}", logger=get_root_logger())
            print_log(
                f"Training config:\n{self.train_config.pretty_text}",
                logger=get_root_logger(),
            )

    def _set_step(self, step: int):
        """Active learning step"""
        self.step = step

    def _initialize_det_trainer(self):
        """
        Initialize mmdet trainer (so-called detector).
        """
        self.det_trainer: "BaseObjectDetectionTrainer" = initialize_from_config(
            cfg=self.config.trainer,
            name="trainer",
            master_trainer=self,
            log_filename="training",
        )
        self.model: nn.Module = self.det_trainer.model

    def _train_one_step(self):
        """Train one active learning step"""
        self.det_trainer.train_and_validate()
        synchronize()
        # We only have all metric values in the main process
        if is_main_process():
            all_mAPs = self.det_trainer.runner.meta["all_metrics"]["bbox_mAP"]
            best_mAP = max(all_mAPs)
            broadcast_object(best_mAP)
        else:
            best_mAP = broadcast_object()
        return best_mAP

    def _pre_active_learn_step(self):
        """
        To be called at the beginning at each active learning step.
        """
        # Update seed
        self.seed = (self.seed + 1) % (2 ** 32)
        self.train_config.seed = self.seed
        set_random_seed(seed=self.seed, deterministic=False)

    def _mid_active_learning_step(self):
        """
        To be called at the middle of each active learning step (right before
        the model is trained). Usually used for model resetting.
        """
        self.det_trainer.reset()

    def _post_active_learning_step(self):
        """
        To be called at the end of each active learning step.
        """
        self.start_epoch = 0
        self._stop = False

        # Sanity check
        records = self.policy.get_records()
        gather_and_compare(records, device=self.config.device)

        # Save results at the end of each active learning step
        if is_main_process():
            if self.save_dir is not None:
                # Save main results
                save_path = os.path.join(self.save_dir, "active_learning_main_results.json")
                with open(save_path, "w") as fout:
                    json.dump(self.policy.get_records(), fout, indent=2)
                print_log(f"Main results saved to {save_path}", logger=get_root_logger())

                # Save model weights
                if self.config.general.save_model:
                    # Best model was saved as "best_{metric}_iter_{iter}.pth"
                    src_best_path = self.det_trainer.runner.meta["hook_msgs"]["best_ckpt"]
                    best_path = os.path.join(self.save_dir, f"model.best.step_{self.step}.pth")
                    shutil.move(src_best_path, best_path)

                    # Last model was saved as "iter_{iter}.pth", and we expect
                    # to have only one such file
                    src_last_path = self.det_trainer.runner.meta["hook_msgs"]["last_ckpt"]
                    last_path = os.path.join(self.save_dir, f"model.last.step_{self.step}.pth")
                    shutil.move(src_last_path, last_path)
                    # Remove symlink
                    os.remove(os.path.join(self.save_dir, "latest.pth"))

                    # Log
                    print_log(
                        f"Best checkpoint is saved to {best_path} and last checkpoint is "
                        f"saved to {last_path}",
                        logger=get_root_logger(),
                    )

                # Save pool data
                if self.config.general.save_pool:
                    save_path = os.path.join(self.save_dir, f"pool_data.step_{self.step}.pkl")
                    self.policy.save_pool_data(save_path=save_path)

            else:
                records_str = json.dumps(self.policy.get_records(), indent=2)
                print_log(f"Active learning results:\n{records_str}", logger=get_root_logger())

        synchronize()

    def active_learn(
        self,
    ):
        """
        Termination condition: either `num_steps` steps has been reached, or
        there is no index left in the index pool.
        """
        num_steps = self.config.training.num_steps
        step_count = 0
        p = "=" * 40

        # Start one cycle (active learning step)
        while True:

            self._set_step(step_count)
            print_log(
                f"\n{p}\nActive learning: step: {step_count}/{num_steps}\n{p}\n",
                logger=get_root_logger(),
            )

            # Pre-call
            self._pre_active_learn_step()

            # Generate predictions and select samples
            if self.step == 0:  # first cycle
                unlabeled_outputs = None
                labeled_outputs = None
            else:
                unlabeled_outputs = self.active_learner.forward(
                    model=self.model_for_selection,
                    data_pool=self.unlabeled_pool,
                )
                labeled_outputs = self.active_learner.forward(
                    model=self.model_for_selection,
                    data_pool=self.labeled_pool,
                )

            synchronize()
            self.policy.acquire(unlabeled_outputs, labeled_outputs)

            # Mid-call: reset model, optimizer, scheduler and metrics
            self._mid_active_learning_step()

            # Train model on old data + newly selected data for one step
            best_mAP = self._train_one_step()
            print_log(
                f"Training step {self.step} done. Best mAP: {best_mAP:.4f}",
                logger=get_root_logger(),
            )

            # Take a step for the learner
            self.active_learner.step()
            # Record stats and clear all cache
            self.policy.record_stats(
                step=step_count,
                mAP=best_mAP,
            )

            # Update
            self._post_active_learning_step()
            step_count += 1

            # Termination
            if self.policy.is_termination():
                if num_steps is not None and step_count < num_steps:
                    print_log(
                        f"Early termination: active learning is terminated sooner than expected: "
                        f"last step: {step_count}, number of expected steps: {num_steps}.",
                        logger=get_root_logger(),
                        level=logging.WARNING,
                    )
                print_log(
                    "No index left in the pool. Stopping active learning...",
                    logger=get_root_logger(),
                )
                break

            if num_steps is not None and step_count == num_steps:
                print_log(
                    f"Number of step reached: {num_steps}. Stopping active learning...",
                    logger=get_root_logger(),
                )
                break

        print_log(
            f"Finished active learning. Number of steps in total: {step_count}",
            logger=get_root_logger(),
        )
