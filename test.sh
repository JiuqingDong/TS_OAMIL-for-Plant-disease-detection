# printf paprika_noise_0.0
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py /home/multiai3/Jiuqing/OA-MIL-plants/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika.py \
#     /home/multiai3/Jiuqing/OA-MIL-plants/outputs/GWHD/fasterrcnn_paprika_noise_0.0/epoch_14.pth --eval 'bbox'
#
# printf paprika_oamil_noise_0.0
# CUDA_VISIBLE_DEVICES=0 \
# python tools/test.py /home/multiai3/Jiuqing/OA-MIL-plants/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika_oamil.py \
#     /home/multiai3/Jiuqing/OA-MIL-plants/outputs/GWHD/fasterrcnn_paprika_noise_0.0_oamil/epoch_14.pth --eval 'bbox'6

CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    /home/multiai3/Jiuqing/OA-MIL-plants/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika.py \
    /home/multiai3/Jiuqing/OA-MIL-plants/GWHD/fasterrcnn_oamil_0/epoch_16.pth \
     --eval 'bbox' \
     --out '/home/multiai3/Jiuqing/OA-MIL-plants/Visualization/GWHD_test/Result_all.pkl' \
     --show-dir '/home/multiai3/Jiuqing/OA-MIL-plants/Visualization/GWHD_test/all/'\
     --show-score-thr 0.5

#CUDA_VISIBLE_DEVICES=1 \
#python tools/test.py \
#    /home/multiai3/Jiuqing/OA-MIL-plants/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_paprika_oamil.py \
#    /home/multiai3/Jiuqing/OA-MIL-plants/weights/epoch_15.pth \
#     --eval 'bbox' \
#     --out '/home/multiai3/Jiuqing/OA-MIL-plants/Result/paprika_0.4/Result.pkl' \
#     --show-dir '/home/multiai3/Jiuqing/OA-MIL-plants/Result/paprika_0.4/image/'\
#     --show-score-thr 0.5