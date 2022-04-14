# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.apis.active_learning import (
    custom_logic_pretraining,
    custom_logic_posttraining,
    active_learning_inference,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector for active learning, with the default '
                    'dataset being `DatumaroV1Dataset`'
    )
    parser.add_argument(
        'train_dataset_dir',
        help="""Dataset directory for training. It should have the following
            structure
                train_dataset_dir/
                ├── annotations
                │   ├── train.json
                │   └── val.json
                └── images
                    └── default
                        ├── xxx.jpg
                        ├── ...
                        └── yyy.jpg
            where `train.json` and `val.json` should have already been
            processed with
            `mmdetection/tools/dataset_converters/datumaro_to_coco.py`.
        """,
    )
    parser.add_argument(
        'inference_dataset_dir',
        help="Dataset directory for AL inference. To be used with "
             "`inference_patterns`",
    )
    parser.add_argument('work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/_active_learning_/faster_rcnn_r50_fpn_1x_datumaro.py',
    )
    parser.add_argument(
        '--inference_patterns',
        type=str,
        nargs="+",
        default=["*.jpg", "*.png"],
        help="Search patterns for data. For example, in a image-based task, "
             "one should specify ['*.jpg', '*.png']",
    )
    parser.add_argument(
        '--max-image-width',
        help='Maxium image width',
        default=MAX_IMAGE_WIDTH,
    )
    parser.add_argument(
        '--max-image-height',
        help='Maxium image height',
        default=MAX_IMAGE_HEIGHT,
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    """
    ======================================================================
    Basic settings. Largely remain the same as the original `train.py` script.
    ======================================================================
    """

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.auto_resume = False
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    cfg.distributed = distributed

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: False')
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    """
    ======================================================================
    Custom pre-training logic.
    ======================================================================
    """
    # Set custom attributes
    custom_logic_pretraining(cfg, args, logger)
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    """
    ======================================================================
    Define model, datasets, etc. then start training.
    ======================================================================
    """

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    logger.info(f"Number of training samples: {len(datasets[0])}")
    # save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__ + get_git_hash()[:7],
        CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    runner = train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta)

    """
    ======================================================================
    Custom post-training logic.
    ======================================================================
    """
    custom_logic_posttraining(runner, cfg, logger)

    """
    ======================================================================
    Active learning inference.
    ======================================================================
    """
    results = active_learning_inference(
        cfg=cfg,
        model=model,
        data_dir=args.inference_dataset_dir,
        patterns=args.inference_patterns,
        logger=logger,
    )
    with open(osp.join(cfg.work_dir, "al_inference.pkl"), "wb") as fout:
        pickle.dump(results, fout)


if __name__ == '__main__':
    main()
