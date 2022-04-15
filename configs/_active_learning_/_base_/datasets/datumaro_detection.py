# dataset settings
dataset_type = 'DatumaroV1Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=None,  # to be set manually
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,  # to be set manually
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=4),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=None,  # to be set manually
        img_prefix=None,  # to be set manually
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=None,  # to be set manually
        img_prefix=None,  # to be set manually
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=None,  # to be set manually
        img_prefix=None,  # to be set manually
        pipeline=test_pipeline),
    )
evaluation = dict(interval=1, metric='bbox')
