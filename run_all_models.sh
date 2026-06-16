#!/usr/bin/env bash
set -euo pipefail

python train_custom.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
python train_lenet5.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.05 --amp
python train_alexnet.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
python train_vgg.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
python train_googlenet.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
python train_resnet.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
