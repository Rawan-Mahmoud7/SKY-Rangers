#!/usr/bin/env python3
"""
188-Country Flag Classifier — ConvNeXt V2 Tiny + ArcFace
Google Colab T4 training script.

Dataset: 188k synthetic aerial flag crops (1000/country)
Layout:  dataset/{train,val,test}/<Country>/img_XXXXXX.jpg

Usage:
    !pip install timm pytorch-metric-learning torchmetrics
    !python train.py --dataset /content/drive/MyDrive/dataset
"""

import os, json, time, random, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import timm
from pytorch_metric_learning.losses import ArcFaceLoss
from torchmetrics.classification import MulticlassAccuracy


# ── Model ───────────────────────────────────────────────────────────────

class FlagClassifier(nn.Module):
    """RepViT-M1 (384-d) → BN → Linear(512) → BN"""

    def __init__(self, backbone_name, embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Helpers ─────────────────────────────────────────────────────────────

def get_logits(criterion, embeddings):
    """Classification logits from ArcFace weight matrix (for accuracy)."""
    W = F.normalize(criterion.W, dim=0)          # (embed, classes)
    E = F.normalize(embeddings, dim=1)            # (batch, embed)
    return criterion.scale * (E @ W.to(E.dtype))  # (batch, classes)


def get_transforms(size, train=True):
    """ImageNet-normalised transforms. Mild aug (data is pre-augmented)."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(10, translate=(0.02, 0.02), scale=(0.95, 1.105)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(), norm,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])


# ── Train / Eval ────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, loader, optimizer, scheduler,
                    scaler, device, acc1, acc5, epoch, total_epochs):
    model.train()
    criterion.train()
    acc1.reset()
    acc5.reset()
    total_loss, count = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{total_epochs} [train]", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(criterion.parameters()), 1.0)
        
        # Only step scheduler if the optimizer step was not skipped by the scaler due to NaN/Inf grads
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        count += bs

        with torch.no_grad():
            logits = get_logits(criterion, embeddings.detach())
            acc1.update(logits, labels)
            acc5.update(logits, labels)

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / count, acc1.compute().item() * 100, acc5.compute().item() * 100



@torch.no_grad()
def evaluate(model, criterion, loader, device, num_classes):
    model.eval()
    criterion.eval()
    acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    total_loss, count = 0.0, 0

    for imgs, labels in tqdm(loader, desc="  [eval]", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)

        logits = get_logits(criterion, embeddings)
        acc1.update(logits, labels)
        acc5.update(logits, labels)
        total_loss += loss.item() * labels.size(0)
        count += labels.size(0)

    return total_loss / count, acc1.compute().item() * 100, acc5.compute().item() * 100


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train 188-flag classifier")
    p.add_argument("--dataset",    default="/content/drive/MyDrive/dataset")
    p.add_argument("--save-dir",   default="/content/drive/MyDrive/flag_model")
    p.add_argument("--epochs",     type=int,   default=36)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--img-size",   type=int,   default=224)
    p.add_argument("--patience",   type=int,   default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    # ── Data extraction (Colab Drive speedup) ──
    if args.dataset.endswith(".zip") and os.path.isfile(args.dataset):
        import zipfile
        dst = "/content/dataset_local" if os.path.exists("/content") else "./dataset_local"
        if not os.path.exists(dst):
            print(f"Extracting zip dataset {args.dataset} to local SSD ({dst}) for fast access...")
            with zipfile.ZipFile(args.dataset, 'r') as z:
                z.extractall(dst)
        args.dataset = os.path.join(dst, "dataset")
    elif os.path.isdir(args.dataset) and args.dataset.startswith("/content/drive/"):
        import shutil
        dst = "/content/dataset_local"
        if not os.path.exists(dst):
            print("Copying dataset to local SSD...")
            shutil.copytree(args.dataset, dst)
        args.dataset = os.path.join(dst, "dataset")
        print(f"Using dataset: {args.dataset}")
    elif os.path.isdir(args.dataset) and os.path.exists("/content"):
        zip_path = args.dataset.rstrip("/") + ".zip"
        if os.path.isfile(zip_path):
            import zipfile
            dst = "/content/dataset_local"
            if not os.path.exists(dst):
                print(f"Found dataset zip at {zip_path}. Extracting to local SSD ({dst}) for fast access...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(dst)
            args.dataset = dst

    # ── Data ──
    train_ds = datasets.ImageFolder(
        f"{args.dataset}/train", get_transforms(args.img_size, train=True))
    val_ds = datasets.ImageFolder(
        f"{args.dataset}/val", get_transforms(args.img_size, train=False))
    test_ds = datasets.ImageFolder(
        f"{args.dataset}/test", get_transforms(args.img_size, train=False))
    num_classes = len(train_ds.classes)

    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(
        val_ds, args.batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(
        test_ds, args.batch_size * 2, shuffle=False, **kw)

    print(f"Device: {device} | Classes: {num_classes} | "
          f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Model + Loss ──
    backbone = "repvit_m1"
    embed_dim = 512
    model = FlagClassifier(backbone, embed_dim).to(device)
    criterion = ArcFaceLoss(
        num_classes, embed_dim, margin=0.4, scale=64.0).to(device)

    print(f"Backbone: {backbone} | Params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Optimizer (backbone 0.1× LR — proven for fine-tuning pretrained) ──
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": list(model.head.parameters()) + list(criterion.parameters()),
         "lr": args.lr},
    ], weight_decay=0.05)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr * 0.1, args.lr],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=3 / args.epochs,          # 3-epoch warmup
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    train_acc1 = MulticlassAccuracy(num_classes, top_k=1).to(device)
    train_acc5 = MulticlassAccuracy(num_classes, top_k=5).to(device)

    # ── Resume from checkpoint if available ──
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_top1": [], "val_top1": [], "val_top5": [],
    }

    if os.path.isfile(checkpoint_path):
        print(f"\n► Resuming from checkpoint: {checkpoint_path}")
        ckpt_resume = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt_resume["model"])
        criterion.load_state_dict(ckpt_resume["criterion"])
        optimizer.load_state_dict(ckpt_resume["optimizer"])
        scaler.load_state_dict(ckpt_resume["scaler"])
        best_acc = ckpt_resume["best_acc"]
        patience_counter = ckpt_resume["patience_counter"]
        history = ckpt_resume["history"]
        start_epoch = ckpt_resume["epoch"] + 1  # resume from next epoch

        # Restore scheduler state for exact LR continuation
        scheduler.load_state_dict(ckpt_resume["scheduler"])
        print(f"  Loaded epoch {ckpt_resume['epoch']+1}/{args.epochs} | "
              f"best val acc: {best_acc:.2f}% | patience: {patience_counter}/{args.patience}")
    else:
        print(f"\n► Starting training from scratch ({args.epochs} epochs)")

    # ── Train ──
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        tr_loss, tr_t1, tr_t5 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            scaler, device, train_acc1, train_acc5, epoch, args.epochs)

        v_loss, v_t1, v_t5 = evaluate(
            model, criterion, val_loader, device, num_classes)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_top1"].append(tr_t1)
        history["val_top1"].append(v_t1)
        history["val_top5"].append(v_t5)

        star = ""
        if v_t1 > best_acc:
            best_acc = v_t1
            torch.save({
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "val_acc": v_t1,
            }, f"{args.save_dir}/best_model.pth")
            star = " ★"
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint after every epoch (for resume)
        torch.save({
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "patience_counter": patience_counter,
            "history": history,
        }, checkpoint_path)

        lr_now = optimizer.param_groups[-1]["lr"]
        print(f"E{epoch+1:02d} | train {tr_loss:.4f} / {tr_t1:.1f}% | "
              f"val {v_loss:.4f} / {v_t1:.1f}% / top5 {v_t5:.1f}%{star} | "
              f"lr {lr_now:.1e} | {time.time()-t0:.0f}s")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered: no improvement for {args.patience} epochs.")
            break

    # ── Test (load best checkpoint) ──
    ckpt = torch.load(f"{args.save_dir}/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    criterion.load_state_dict(ckpt["criterion"])
    t_loss, t_t1, t_t5 = evaluate(
        model, criterion, test_loader, device, num_classes)
    print(f"\n{'='*50}")
    print(f"TEST  top-1: {t_t1:.2f}%  top-5: {t_t5:.2f}%  loss: {t_loss:.4f}")
    print(f"{'='*50}")

    # ── Save class mapping + history ──
    idx = train_ds.class_to_idx
    with open(f"{args.save_dir}/class_mapping.json", "w") as f:
        json.dump({"class_to_idx": idx,
                   "idx_to_class": {v: k for k, v in idx.items()}}, f, indent=2)
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot curves ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_loss"], label="train")
        ax1.plot(history["val_loss"],   label="val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(history["train_top1"], label="train top-1")
        ax2.plot(history["val_top1"],   label="val top-1")
        ax2.plot(history["val_top5"],   label="val top-5", ls="--")
        ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend()
        fig.tight_layout()
        fig.savefig(f"{args.save_dir}/curves.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    print(f"✓ Done. Best val top-1: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
