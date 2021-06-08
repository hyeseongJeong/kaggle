#!/bin/bash

docker run \
--name dvc_mobilenet \
--runtime=nvidia \
-d \
-it \
-e NVIDIA_VISIBLE_DEVICES=3 \
-v /home/storage_disk2/saved_models/dogs_vs_cats:/tf/saved_model \
-v /home/storage_disk2/datasets/public/dogs_vs_cats:/tf/dataset \
-v /home/storage_disk2/saved_models/:/tf/model \
-w /tf/model \
yodj/dlt \
python ./dogs_vs_cats/src/train.py --config_path /tf/saved_model/train.yml

