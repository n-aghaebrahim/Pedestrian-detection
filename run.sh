#!/bin/bash
#
#
python3 main_2.py \
    --predict_check True \
	--train_check False \
    --yaml_file_path /Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/data.yaml \
    --data_dirs /Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/ \
    --hist_output ../output/data_eda/ \
    --size_output ../output/data_eda/ \
    --model yolov8.yaml \
    --optimizer Adam \
    --patience 80 \
    --verbose True \
    --yaml_data /Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/data.yaml \
    --epochs 50 \
    --source screen \
    --imgsz 416 \
    --conf 0.1 \
    --iou 0.1 \
    --show True \
    --modeldir /Users/navid/work_station/github/Pedestrian-detection/output/detect/train2/weights/last.pt \

