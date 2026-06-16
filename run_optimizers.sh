#!/usr/bin/env bash
set -euo pipefail

python train_resnet.py --epochs 50 --batch-size 128 --optimizer sgd --lr 0.1
python train_resnet.py --epochs 50 --batch-size 128 --optimizer momentum --lr 0.1
python train_resnet.py --epochs 50 --batch-size 128 --optimizer adam --lr 0.001
python train_resnet.py --epochs 50 --batch-size 128 --optimizer adamw --lr 0.001
python train_resnet.py --epochs 50 --batch-size 128 --optimizer rmsprop --lr 0.001
