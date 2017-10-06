#!/usr/bin/env bash
python train.py --model_dir=/home/tensor/tensor/scene/DataSet/checkpoints \
                --train_image_dir=/home/tensor/tensor/scene/DataSet/train \
                --validate_image_dir=/home/tensor/tensor/scene/DataSet/validation \
                --pretrained_model_path=/home/tensor/tensor/scene/DataSet/pre_trained/inception_resnet_v2.ckpt
