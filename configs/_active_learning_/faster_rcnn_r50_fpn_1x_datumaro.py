_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/datumaro_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=None,  # to be set manually
        )))


optimizer = dict(lr=0.04)  # the batch size in `datumaro_detection.py` is doubled
