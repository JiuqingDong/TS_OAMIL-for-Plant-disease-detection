# CUDA_VISIBLE_DEVICES=2 \
# python -m torch.distributed.launch  \
# 	--nproc_per_node=1  \
# 	--master_port=11500 \
#     ./tools/train.py \
#     ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika_oamil.py \
#     --work-dir='./outputs/paprika_update/fasterrcnn_paprika_noise_0.4_oamil' \
#     --launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2, \
python -m torch.distributed.launch  \
	--nproc_per_node=3  \
	--master_port=11000 \
    ./tools/train.py \
    ./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika_oamil.py \
    --work-dir='./a_this_is_a_test' \
    --launcher pytorch
    # --work-dir='./outputs_shot_f2/paprika_control_class/fasterrcnn_paprika_noise_0.4_oamil_x' \
