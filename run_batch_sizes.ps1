$ErrorActionPreference = "Stop"

python train_resnet.py --epochs 50 --batch-size 32 --optimizer momentum --lr 0.025
python train_resnet.py --epochs 50 --batch-size 64 --optimizer momentum --lr 0.05
python train_resnet.py --epochs 50 --batch-size 128 --optimizer momentum --lr 0.1
python train_resnet.py --epochs 50 --batch-size 256 --optimizer momentum --lr 0.2
