import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cifar_models import build_model, count_parameters


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(data_dir, batch_size, num_workers, limit_train=0, limit_test=0):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_tf)
    if limit_train:
        train_set = Subset(train_set, range(min(limit_train, len(train_set))))
    if limit_test:
        test_set = Subset(test_set, range(min(limit_test, len(test_set))))
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def build_optimizer(name, model, lr, weight_decay):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError("optimizer must be one of: sgd, momentum, adam, adamw, rmsprop")


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy(logits, targets)
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy(logits, targets)
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def save_history(history, out_dir):
    csv_path = out_dir / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return csv_path


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = get_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.limit_train,
        args.limit_test,
    )
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    out_dir = Path(args.out_dir) / args.model / f"{args.optimizer}_bs{args.batch_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": args.model,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "parameters": count_parameters(model),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_acc = 0.0
    history = []
    start_time = time.time()
    print(json.dumps(metadata, ensure_ascii=False))
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        record = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
            "lr": f"{lr_now:.8f}",
            "seconds": f"{time.time() - epoch_start:.2f}",
        }
        history.append(record)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "test_acc": best_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} lr={lr_now:.6f}"
        )

    csv_path = save_history(history, out_dir)
    print(f"Best test acc: {best_acc:.4f}")
    print(f"History: {csv_path}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    print(f"Total time: {time.time() - start_time:.1f}s")


def parse_args(extra_args=None):
    parser = argparse.ArgumentParser(description="CIFAR-10 classification experiments")
    parser.add_argument("--model", default="resnet", choices=["custom", "lenet5", "alexnet", "vgg", "googlenet", "resnet"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", default="momentum", choices=["sgd", "momentum", "adam", "adamw", "rmsprop"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="use mixed precision on CUDA")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="debug only: use first N train samples")
    parser.add_argument("--limit-test", type=int, default=0, help="debug only: use first N test samples")
    return parser.parse_args(extra_args)


def main(extra_args=None):
    run(parse_args(extra_args))


if __name__ == "__main__":
    main()
