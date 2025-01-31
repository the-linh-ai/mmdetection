import argparse
import copy
import json
import logging
import os.path as osp

from mmcv.utils import print_log
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert data format from Datumaro to COCO.')
    parser.add_argument(
        'data_dir',
        help="""Dataset directory. It should have the following structure
            dataset_dir/
            ├── annotations
            │   └── default.json
            └── images
                └── default
                    ├── xxx.jpg
                    ├── ...
                    └── yyy.jpg
            This script will write `train.json` and `val.json` to the
            `annotations` subdir.
        """,
    )
    parser.add_argument(
        'chosen_img_path',
        type=str,
        help="Path to the `chosen_imgs.txt` file containing names of files "
             "used for training and evaluation.",
    )
    parser.add_argument(
        'val_split',
        type=float,
        help='Ratio of the validation set.')
    args = parser.parse_args()
    return args


TEMPLATE = {
    "licenses": [
        {"name": "", "id": 0, "url": ""}
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [],
    "images": [],
    "annotations": [],
}


def _datumaro_to_coc(datumaro_data, sample_idxs):
    coco_data = copy.deepcopy(TEMPLATE)

    # Categories
    for i, cat in enumerate(datumaro_data["categories"]["label"]["labels"]):
        coco_data["categories"].append({
            "id": i,
            "name": cat["name"],
            "supercategory": "",
        })

    image_id = 0
    ann_id = 0
    for i in sample_idxs:
        item = datumaro_data["items"][i]

        # Image
        image_height = item["attr"]["height"]
        image_width = item["attr"]["width"]
        coco_data["images"].append({
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": item["attr"]["file_name"],
        })

        # Annotations
        for ann in item["annotations"]:
            if ann["type"] == "bbox":
                x, y, w, h = ann["bbox"]
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": ann["label_id"],
                    "bbox": [x, y, w, h],
                    "area": round(w * h),
                    "segmentation": None,
                    "iscrowd": 0,
                })
                ann_id += 1
            elif ann["type"] in ["polygon", "label"]:
                pass
            else:
                print_log(
                    f"Unrecognized annotation type: '{ann['type']}'",
                    level=logging.WARNING,
                )
        image_id += 1

    return coco_data


def datumaro_to_coco(datumaro_data, chosen_images, val_split):
    # Filter chosen images
    datumaro_data["items"] = [
        item for item in datumaro_data["items"]
        if item["attr"]["file_name"] in chosen_images
    ]

    # Allocations
    assert 0.0 <= val_split < 1.0
    num_samples = len(datumaro_data["items"])
    assert num_samples > 1, f"Too few samples: {num_samples}"

    num_samples_val = max(round(num_samples * val_split), 1)
    num_samples_train = num_samples - num_samples_val
    assert num_samples_val > 0 and num_samples_train > 0

    allocations = ([0] * num_samples_train) + ([1] * num_samples_val)
    allocations = np.array(allocations)
    print_log(
        f"Allocating {num_samples_train} samples for training and "
        f"{num_samples_val} samples for validation.",
        level=logging.WARNING,
    )

    # Shuffle
    rng = np.random.default_rng(136)
    rng.shuffle(allocations)

    # Convert
    train_sample_idxs = (1 - allocations).nonzero()[0]
    train_data = _datumaro_to_coc(datumaro_data, train_sample_idxs)
    val_sample_idxs = allocations.nonzero()[0]
    val_data = _datumaro_to_coc(datumaro_data, val_sample_idxs)

    return train_data, val_data


def main():
    args = parse_args()

    # Read annotations
    with open(osp.join(args.data_dir, "annotations/default.json"), "r") as fin:
        annotations = json.load(fin)
    # Read list of chosen images
    with open(args.chosen_img_path, "r") as fin:
        chosen_images = [line.strip() for line in fin]
    assert len(chosen_images) == len(set(chosen_images))
    chosen_images = set(chosen_images)

    # Convert
    train_data, val_data = datumaro_to_coco(
        annotations, chosen_images, val_split=args.val_split)
    # Save
    with open(osp.join(args.data_dir, "annotations/train.json"), "w") as fout:
        json.dump(train_data, fout, indent=2)
    with open(osp.join(args.data_dir, "annotations/val.json"), "w") as fout:
        json.dump(val_data, fout, indent=2)


if __name__ == '__main__':
    main()
