# CIFAR-10 Image Classification with PyTorch

This project contains six CIFAR-10 classifiers:

- `train_custom.py`: custom CNN baseline
- `train_lenet5.py`: LeNet-5 adapted to RGB CIFAR-10 images
- `train_alexnet.py`: AlexNet-style CIFAR model
- `train_vgg.py`: VGG-style CIFAR model with BatchNorm
- `train_googlenet.py`: GoogLeNet/Inception-style CIFAR model
- `train_resnet.py`: ResNet-18-style CIFAR model

The shared training code is in `train.py`, and model definitions are in `cifar_models.py`.

## Run One Model

```powershell
python train_resnet.py --epochs 100 --batch-size 128 --optimizer momentum --lr 0.1 --amp
```

The local dataset is expected at:

```text
data/cifar-10-batches-py
```

Training outputs are saved under `runs/<model>/<optimizer>_bs<batch_size>/`:

- `metadata.json`
- `history.csv`
- `best.pt`

## Compare Network Structures

```powershell
.\run_all_models.ps1
```

Recommended fixed setting for the report:

```text
optimizer = SGD + Momentum
lr = 0.1
batch_size = 128
epochs = 100
scheduler = CosineAnnealingLR
augmentation = RandomCrop + RandomHorizontalFlip + Normalize
```

## Compare Optimizers

```powershell
.\run_optimizers.ps1
```

This compares `sgd`, `momentum`, `adam`, `adamw`, and `rmsprop` on the ResNet model.

## Compare Batch Sizes

```powershell
.\run_batch_sizes.ps1
```

The script uses a simple linear learning-rate rule:

```text
batch_size 32  -> lr 0.025
batch_size 64  -> lr 0.05
batch_size 128 -> lr 0.1
batch_size 256 -> lr 0.2
```

## Debug Smoke Test

Use small data limits to verify the code quickly:

```powershell
python train_resnet.py --epochs 1 --batch-size 8 --limit-train 32 --limit-test 16 --cpu --num-workers 0
```
