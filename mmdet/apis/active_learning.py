from collections import Counter
import json
import os.path as osp
import time

import torch
import numpy as np

from mmal.data_utils import CustomDataset
from mmal.dist_utils import gather
from mmal.uncertainty_utils import calculate_entropy_np, get_unique_indices
import mmcv
from mmcv.runner import get_dist_info
from mmdet.datasets import build_dataloader


# DEFAULT VARIABLES
MAX_IMAGE_HEIGHT = 640
MAX_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 128
MIN_IMAGE_WIDTH = 128


def read_ann(args):
    """Retrieve information from the annotation file"""
    ann_path = osp.join(args.train_dataset_dir, "annotations/train.json")
    with open(ann_path, "r") as fin:
        annotations = json.load(fin)
    image_ratios = []
    for image in annotations["images"]:
        image_ratio = round(image["width"] / image["height"], 1)  # round to 1 d.p.
        image_ratios.append(image_ratio)

    counter = Counter(image_ratios)
    image_ratio, _ = counter.most_common(n=1)[0]  # most common ratio

    return {
        "image_ratio": image_ratio,
        "is_single_ratio": len(counter) == 1,
        "num_classes": len(annotations["categories"])
    }


def get_image_size(args, image_ratio, is_single_ratio):
    """Get the most appropriate image size for training"""
    max_image_height = args.max_image_height
    max_image_width = args.max_image_width
    # If images of varied sizes, we better transform them into square images
    if not is_single_ratio:
        image_width = image_height = min(max_image_height, max_image_width)
    # If single image ratio, we want to maintain this aspect ratio
    else:
        max_image_ratio = round(max_image_width / max_image_height, 1)
        if image_ratio >= max_image_ratio:
            image_width = max_image_width
            image_height = round(image_width / image_ratio)
        else:
            image_height = max_image_height
            image_width = round(image_height * image_ratio)
        # Make image width and height divisible by 32
        image_width = (image_width // 32) * 32
        image_height = (image_height // 32) * 32
        # Fall back to the max image resolution if failed to derive
        # an appropriate image resolution
        if image_width < MIN_IMAGE_WIDTH or image_height < MIN_IMAGE_HEIGHT:
            image_width = max_image_width
            image_height = max_image_height
    return (image_height, image_width)


def custom_logic_pretraining(cfg, args, logger, orig_batch_size):
    """Set custom attributes for the config in-place"""
    # Read the annotation file
    ann_info = read_ann(args)

    # Auto infer an appropriate image sizes
    image_size = get_image_size(
        args, ann_info["image_ratio"], ann_info["is_single_ratio"])
    logger.info(
        f"Chosen image size (h, w): {image_size}, is_single_ratio: "
        f"{ann_info['is_single_ratio']}"
    )
    # Set image sizes
    assert cfg.data.train.pipeline[2].type == "Resize"
    cfg.data.train.pipeline[2].img_scale = image_size
    assert cfg.data.val.pipeline[1].type == "MultiScaleFlipAug"
    cfg.data.val.pipeline[1].img_scale = image_size
    assert cfg.data.test.pipeline[1].type == "MultiScaleFlipAug"
    cfg.data.test.pipeline[1].img_scale = image_size

    # If single ratio, enable torch.cudnn.benchmark
    # Otherwise disable it
    # See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = ann_info["is_single_ratio"]

    # Set data paths
    img_dir = osp.join(args.train_dataset_dir, "images/default/")
    train_ann_file = osp.join(args.train_dataset_dir, "annotations/train.json")
    val_ann_file = osp.join(args.train_dataset_dir, "annotations/val.json")
    assert osp.exists(img_dir) and osp.exists(train_ann_file) and \
        osp.exists(val_ann_file)

    cfg.data.train.ann_file = train_ann_file
    cfg.data.train.img_prefix = img_dir
    cfg.data.val.ann_file = val_ann_file
    cfg.data.val.img_prefix = img_dir
    cfg.data.test.ann_file = val_ann_file
    cfg.data.test.img_prefix = img_dir

    # Set number of classes
    cfg.model.roi_head.bbox_head.num_classes = ann_info["num_classes"]

    # Checkpoint
    cfg.checkpoint_config.interval = int(2 ** 32)  # don't save after every epoch
    cfg.evaluation.save_best = "bbox_mAP"  # save best model based on this metric

    # Learning rate rescaling; note that in MMLab, all configs are
    # to be used with 8 GPUs
    if not args.no_autoscale_lr:
        batch_size = cfg.data.samples_per_gpu
        num_gpus = len(cfg.gpu_ids)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * \
            (batch_size * num_gpus) / (orig_batch_size * 8)
        logger.info(
            f"Learning rate has been rescaled to {cfg.optimizer['lr']}"
        )


def custom_logic_posttraining(runner, cfg, logger):
    """Post-training custom logic"""
    pass


@torch.no_grad()
def _active_learning_inference(model, data, device):
    # pred_bboxes: list[list[ndarray]]; probs: list[list[ndarray]]
    # outer: sample-level; inner: class-level
    pred_bboxes, probs = model.simple_test(
        img=data['img'][0].to(device),
        img_metas=data['img_metas'][0].data[0],
        rescale=True,
        return_probs=True,
    )
    # Unpack
    assert len(pred_bboxes) == len(probs) == 1
    pred_bboxes = pred_bboxes[0]
    probs = probs[0]

    # Calculate entropy
    entropys = [
        calculate_entropy_np(
            prob, dim=1, normalized=True, assert_normalized=True,
        )
        for prob in probs
    ]

    entropys_ = np.concatenate(entropys)
    if len(entropys_) == 0:  # no predictions
        max_entropy = -1
    else:
        max_entropy = entropys_.max()

    return {
        "boxes": pred_bboxes,
        "probs": probs,
        "uncertainties": entropys,
        "final_uncertainty": max_entropy,
    }


@torch.no_grad()
def active_learning_inference(cfg, model, data_dir, patterns, logger):
    # Prepare
    model.eval()
    device = next(model.parameters()).device
    dataset = CustomDataset(cfg, data_dir, patterns, logger)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # always 1
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=len(cfg.gpu_ids),
        dist=cfg.distributed,
        shuffle=False,
        persistent_workers=cfg.data.get('persistent_workers', False),
    )

    # Adapted from mmdet/apis/test.py
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(0.5)  # This line can prevent deadlock problem in some cases.

    # Inference
    keys = []
    preds = []
    for data in dataloader:
        assert len(data["img_metas"]) == len(data["img"]) == 1  # single-scale
        path = data["img_metas"][0].data[0][0]["ori_filename"]
        keys.append(osp.split(path)[1])
        pred = _active_learning_inference(model, data, device)
        preds.append(pred)

        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

    # Gather from multiple GPUs
    keys, preds = gather(
        [keys, preds],
        device=device,
    )
    keys = sum(keys, [])
    preds = sum(preds, [])

    # Remove duplicates
    unique_indices = get_unique_indices(keys, device=device)
    keys = [keys[i] for i in unique_indices]
    preds = [preds[i] for i in unique_indices]
    assert len(keys) == len(preds) == len(dataset)
    results = dict(zip(keys, preds))
    return results
