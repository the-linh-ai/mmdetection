_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/nuimages_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10)))


custom_hooks = [
    dict(
        type='ModelParametersRecorderHook',
        priority=50)
]
