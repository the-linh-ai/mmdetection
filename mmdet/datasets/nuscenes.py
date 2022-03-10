from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class NuScenesDataset(CocoDataset):
    CLASSES = (
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    )

    def _filter_imgs(self):
        super()._filter_imgs(min_size=0)
