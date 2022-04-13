import os
import sys
import argparse
from typing import List, Tuple, Dict, Any

import yaml
import torch

from mmcv.utils import Config
from mmdet.active_learning.master_trainer import MasterTrainer
from mmdet.active_learning.utils.config_utils import DictConfig
from mmdet.utils import get_codebase_root


torch.backends.cudnn.benchmark = True


DESCRIPTION = """
Train and evaluate an active learning model. This code currently only supports
single-gpu, non-distributed training and multi-gpu (or single-gpu) distributed
training. Multi-gpu non-distributed training is not yet supported.

To overwrite config, specify `key=value` when calling with arguments. Nested keys are
to be seperated by "." (e.g. "general.seed=136"). This is supported for both active
learning config and mmdet config. If there are non-parsable options, an error will be
raised.
"""


def parse_unknown_args(
    unknown_args: List[str],
    force_override: bool = False,
) -> Tuple[Dict[str, Any], List[Any]]:

    parsed_unknown_args = {}
    exclude = ["min", "max", "sum"]

    for arg in unknown_args:
        assert "=" in arg

        key, value = arg.split("=")
        if (not force_override) and key in parsed_unknown_args:
            raise ValueError(
                f"Duplicate argument: {key}={value} and {key}={parsed_unknown_args[key]}"
            )

        if value not in exclude:
            try:
                value = eval(value)
            except Exception:
                pass
        parsed_unknown_args[key] = value

    return parsed_unknown_args


def setup_al_config(
    args,
    unknown_args: Dict[str, Any],
    command: str,
):
    """
    Parse and setup active learning config.
    """
    # Active learning config
    with open(args.al_config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    config_dir = os.path.realpath(os.path.split(args.al_config_path)[0])
    config = DictConfig.from_object(
        config,
        config_dir=config_dir,
        unknown_args=unknown_args,
    )

    if args.debug:
        config.general.debugging = True
    if args.no_save:
        config.general.work_dir = None

    config.local_rank = args.local_rank
    config.general.command = command

    return config


def setup_mmdet_config(config_path, unknown_args: Dict[str, Any]):
    config = Config.fromfile(config_path)
    config.merge_from_dict(unknown_args)
    return config


def main(args, command):
    # Parse unknown args
    unknown_args = parse_unknown_args(
        args.unknown_args,
        force_override=args.force_override,
    )

    # Parse AL config
    al_config = setup_al_config(args, unknown_args, command)

    # Workaround: get mmdet config path from AL config
    mmdet_config_path = al_config.mmdet_config.config_path
    if not os.path.isabs(mmdet_config_path):
        mmdet_config_path = os.path.join(
            get_codebase_root(), mmdet_config_path)

    # Parse mmdet config
    train_config = setup_mmdet_config(mmdet_config_path, unknown_args)

    # Initializer trainer
    trainer = MasterTrainer(al_config=al_config, train_config=train_config)

    # Start training
    trainer.active_learn()


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        "-a", "--al-config-path", type=str, required=True,
        help="Config path for active learning."
    )

    parser.add_argument(
        "--debug", default=False, action="store_true",
        help="Enable debug mode."
    )
    parser.add_argument(
        "--no-save", default=False, action="store_true",
        help="Don't save anything locally."
    )
    parser.add_argument(
        "--force-override", default=False, action="store_true",
        help="Whether to allow duplicate arguments."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for DDP. Don't set this value manually."
    )

    args, unknown_args = parser.parse_known_args()
    args.unknown_args = unknown_args

    return args


if __name__ == "__main__":
    command = f"python {' '.join(sys.argv)}"
    args = parse_arguments()

    # TODO: experiment with torch deterministic
    main(args, command)
