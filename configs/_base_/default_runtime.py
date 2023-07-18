checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/home/multiai3/Jiuqing/OA-MIL-plants/outputs_shot_f1/paprika_control_class/fasterrcnn_paprika_noise_0.4_oamil_9/epoch_110.pth' #
load_from = None
resume_from = None
workflow = [('train', 1)]
