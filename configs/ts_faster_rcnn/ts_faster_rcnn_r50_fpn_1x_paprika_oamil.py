_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_paprika_oamil.py'
]

# model settings
model = dict(
    type='TSFasterRCNN',
    teacher_cfg=dict(
        cfg=dict(
            model=dict(
                type='FasterRCNN',
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch'),
                neck=dict(
                    type='FPN',
                    in_channels=[256, 512, 1024, 2048],
                    out_channels=256,
                    num_outs=5),
                rpn_head=dict(
                    type='RPNHead',
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64]),
                    bbox_coder=dict(
                        type='DeltaXYXYBBoxCoder',
                        target_means=[.0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
                    loss_cls=dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
                roi_head=dict(
                    type='StandardRoIHeadOAMIL',  # This module
                    bbox_roi_extractor=dict(
                        type='SingleRoIExtractor',
                        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32]),
                    bbox_head=dict(
                        type='Shared2FCBBoxHeadOAMIL',  # This module
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type='DeltaXYXYBBoxCoder',
                            target_means=[0., 0., 0., 0.],
                            target_stds=[0.1, 0.1, 0.1, 0.1]),
                        reg_class_agnostic=False,
                        loss_cls=dict(
                            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        loss_bbox=dict(type='L1Loss', loss_weight=1.0),

                        # OA-MIL params
                        oamil_lambda=0.1,

                        # OA-IS params
                        oais_flag=True,
                        oais_epoch=1,  # 2
                        oais_gamma=7.5,
                        oais_theta=0.85,

                        # OA-IE params
                        oaie_flag=True,
                        oaie_num=4,
                        oaie_coef=1.0,
                        oaie_epoch=1,  # 9
                    )
                ),
                test_cfg=dict(
                        rcnn=None
                    ),
            ),
            load_from='weights/epoch_15.pth',
        ),
    ),
    )
