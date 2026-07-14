#!/usr/bin/env python3
"""
Evaluation and Comparison Script: PyTorch vs. ONNX
Measures inference speed, Top-1/Top-5 accuracy, correct class ranking,
and outputs predictions CSV, confusion matrix, and per-class accuracy.
"""

import os
import argparse
import time
import json
import sys
import io

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError when printing emojis/special characters
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# Try importing ONNX and ONNX Runtime
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Warning: 'onnx' or 'onnxruntime' is not installed. ONNX export and evaluation will not work.")

# Try importing Matplotlib for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Confusion matrix plot will be skipped.")


# ── Model & Wrapper ──────────────────────────────────────────────────

class FlagClassifier(nn.Module):
    """RepViT-M1 (384-d) → BN → Linear(512) → BN"""

    def __init__(self, backbone_name, embed_dim=512, pretrained=False):
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


class FlagClassifierInference(nn.Module):
    """Unified Inference Wrapper combining Model + Normalized ArcFace Weights"""

    def __init__(self, model, W, scale=64.0):
        super().__init__()
        self.model = model
        # Register W as buffer (normalize along dim 0 as in ArcFace get_logits)
        self.register_buffer("W", F.normalize(W, dim=0))
        self.scale = scale

    def forward(self, x):
        embeddings = self.model(x)
        E = F.normalize(embeddings, dim=1)
        logits = self.scale * (E @ self.W)
        return F.softmax(logits, dim=1)


# ── Helpers ─────────────────────────────────────────────────────────────

def get_transforms(size):
    """Standard evaluation normalization transforms."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm
    ])


def export_to_onnx(pytorch_model, onnx_path, device):
    """Exports the unified PyTorch inference model to ONNX format."""
    print(f"\n[ONNX] Exporting PyTorch model to ONNX format at: {onnx_path}...")
    pytorch_model.eval()
    
    # Input dummy tensor (Batch size 1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Export the model
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Check model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[ONNX] Export and verification completed successfully!")


def plot_confusion_matrix(cm, classes, save_path):
    """Generates and saves a confusion matrix visualization."""
    if not HAS_MATPLOTLIB:
        return
        
    num_classes = len(classes)
    plt.figure(figsize=(max(10, num_classes // 3), max(10, num_classes // 3)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Only show labels if there are not too many
    if num_classes <= 50:
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
        plt.yticks(tick_marks, classes, fontsize=6)
        
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plots] Saved confusion matrix visualization to: {save_path}")


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate_models(pytorch_model, onnx_path, dataloader, device, idx_to_class):
    # Initialize ONNX Runtime Session
    providers = ['CPUExecutionProvider']
    if device.type == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    print(f"\n[ONNX] Starting ONNX Runtime session with providers: {providers}")
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    
    pytorch_model.eval()
    
    # Lists to store prediction data
    results = []
    
    num_classes = len(idx_to_class)
    pt_cm = np.zeros((num_classes, num_classes), dtype=int)
    on_cm = np.zeros((num_classes, num_classes), dtype=int)
    
    pt_latencies = []
    on_latencies = []
    
    pt_top1_correct = 0
    pt_top5_correct = 0
    on_top1_correct = 0
    on_top5_correct = 0
    
    # Rank distribution counters
    pt_rank_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, '>5': 0}
    on_rank_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, '>5': 0}
    
    total_samples = 0
    
    print("\nEvaluating on dataset...")
    # Iterate through individual samples to log per-image statistics
    # Use batch_size = 1 for accurate latency/inference metrics per image
    for imgs, labels in dataloader:
        # Get true class info
        label = labels.item()
        true_class_name = idx_to_class[label]
        
        # Get file path if available
        img_path = dataloader.dataset.imgs[total_samples][0]
        rel_img_path = os.path.relpath(img_path, start=os.path.dirname(img_path))
        
        total_samples += 1
        
        # ── PyTorch Evaluation ──
        imgs_pt = imgs.to(device)
        with torch.no_grad():
            t_start = time.perf_counter()
            pt_probs = pytorch_model(imgs_pt)
            t_end = time.perf_counter()
            pt_latencies.append((t_end - t_start) * 1000.0) # in ms
            
        pt_probs = pt_probs.squeeze(0).cpu().numpy()
        pt_pred_idx = np.argmax(pt_probs)
        pt_cm[label, pt_pred_idx] += 1
        
        # Sort indices by probability descending
        pt_sorted_indices = np.argsort(pt_probs)[::-1]
        pt_rank = np.where(pt_sorted_indices == label)[0][0] + 1
        
        if pt_rank == 1:
            pt_top1_correct += 1
        if pt_rank <= 5:
            pt_top5_correct += 1
            
        # Update PyTorch rank distribution
        if pt_rank <= 5:
            pt_rank_dist[pt_rank] += 1
        else:
            pt_rank_dist['>5'] += 1
            
        # ── ONNX Evaluation ──
        imgs_on = imgs.numpy()
        t_start = time.perf_counter()
        on_probs = ort_session.run(None, {input_name: imgs_on})[0]
        t_end = time.perf_counter()
        on_latencies.append((t_end - t_start) * 1000.0) # in ms
        
        on_probs = on_probs.squeeze(0)
        on_pred_idx = np.argmax(on_probs)
        on_cm[label, on_pred_idx] += 1
        
        on_sorted_indices = np.argsort(on_probs)[::-1]
        on_rank = np.where(on_sorted_indices == label)[0][0] + 1
        
        if on_rank == 1:
            on_top1_correct += 1
        if on_rank <= 5:
            on_top5_correct += 1
            
        # Update ONNX rank distribution
        if on_rank <= 5:
            on_rank_dist[on_rank] += 1
        else:
            on_rank_dist['>5'] += 1
            
        # Build the row dictionary
        row = {
            'image_path': img_path,
            'image_filename': rel_img_path,
            'true_class_name': true_class_name,
            'true_class_idx': label,
            'pytorch_predicted_name': idx_to_class[pt_pred_idx],
            'pytorch_predicted_idx': pt_pred_idx,
            'pytorch_confidence': float(pt_probs[pt_pred_idx]),
            'pytorch_correct': bool(pt_pred_idx == label),
            'pytorch_rank': int(pt_rank),
            'onnx_predicted_name': idx_to_class[on_pred_idx],
            'onnx_predicted_idx': on_pred_idx,
            'onnx_confidence': float(on_probs[on_pred_idx]),
            'onnx_correct': bool(on_pred_idx == label),
            'onnx_rank': int(on_rank)
        }
        
        # Add Top-5 predictions for PyTorch
        for r in range(1, 6):
            p_idx = pt_sorted_indices[r-1]
            row[f'pytorch_top{r}_name'] = idx_to_class[p_idx]
            row[f'pytorch_top{r}_idx'] = int(p_idx)
            row[f'pytorch_top{r}_conf'] = float(pt_probs[p_idx])
            
        # Add Top-5 predictions for ONNX
        for r in range(1, 6):
            o_idx = on_sorted_indices[r-1]
            row[f'onnx_top{r}_name'] = idx_to_class[o_idx]
            row[f'onnx_top{r}_idx'] = int(o_idx)
            row[f'onnx_top{r}_conf'] = float(on_probs[o_idx])
            
        results.append(row)
        
        if total_samples % 100 == 0:
            print(f"Processed {total_samples} images...")

    # Compute metric summaries
    pt_avg_latency = np.mean(pt_latencies)
    on_avg_latency = np.mean(on_latencies)
    
    pt_fps = 1000.0 / pt_avg_latency
    on_fps = 1000.0 / on_avg_latency
    
    pt_top1_acc = (pt_top1_correct / total_samples) * 100.0
    pt_top5_acc = (pt_top5_correct / total_samples) * 100.0
    on_top1_acc = (on_top1_correct / total_samples) * 100.0
    on_top5_acc = (on_top5_correct / total_samples) * 100.0
    
    metrics = {
        'pt_avg_latency': pt_avg_latency,
        'on_avg_latency': on_avg_latency,
        'pt_fps': pt_fps,
        'on_fps': on_fps,
        'pt_top1_acc': pt_top1_acc,
        'pt_top5_acc': pt_top5_acc,
        'on_top1_acc': on_top1_acc,
        'on_top5_acc': on_top5_acc,
        'total_samples': total_samples,
        'pt_cm': pt_cm,
        'on_cm': on_cm,
        'pt_rank_dist': pt_rank_dist,
        'on_rank_dist': on_rank_dist
    }
    
    return results, metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate and Compare PyTorch vs ONNX models")
    p.add_argument("--model-path", default="./flag_model/best_model.pth", help="Path to PyTorch model checkpoint (.pth)")
    p.add_argument("--class-mapping", default="", help="Path to class_mapping.json (optional)")
    p.add_argument("--dataset", required=True, help="Path to evaluation dataset directory")
    p.add_argument("--onnx-path", default="./flag_model/best_model.onnx", help="Path to save exported ONNX model")
    p.add_argument("--csv-output", default="./flag_model/evaluation_predictions.csv", help="Path to save output CSV predictions")
    p.add_argument("--backbone", default="repvit_m1_1", help="Timm backbone model name")
    p.add_argument("--scale", type=float, default=64.0, help="ArcFace scale parameter")
    args = p.parse_args()

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)

    # 2. Dataset Loading
    print(f"Loading evaluation dataset from: {args.dataset}")
    val_dataset = datasets.ImageFolder(args.dataset, get_transforms(224))
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    classes = val_dataset.classes
    num_classes = len(classes)
    
    # Resolve Class Mapping
    idx_to_class = {i: c for i, c in enumerate(classes)}
    if args.class_mapping and os.path.exists(args.class_mapping):
        print(f"Loading class mapping from: {args.class_mapping}")
        with open(args.class_mapping, 'r') as f:
            mapping_data = json.load(f)
            # Accept idx_to_class dictionary format
            if "idx_to_class" in mapping_data:
                loaded_map = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
                # Verify consistency
                if len(loaded_map) == num_classes:
                    idx_to_class = loaded_map
                    print("Class mapping successfully verified!")
                else:
                    print("Warning: Loaded mapping size differs from dataset classes. Using dataset alphabetical indices.")

    # 3. Model Loading
    print(f"Loading model checkpoint from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu')
    
    # Detect shapes
    ckpt_num_classes = ckpt["criterion"]["W"].shape[1]
    ckpt_embed_dim = ckpt["criterion"]["W"].shape[0]
    print(f"Detected checkpoint weights: classes={ckpt_num_classes}, embedding dimensions={ckpt_embed_dim}")
    
    if ckpt_num_classes != num_classes:
        print(f"Warning: Checkpoint weights dimension ({ckpt_num_classes}) does not match dataset class count ({num_classes}).")

    # Auto-detect backbone features from checkpoint to prevent size mismatches
    if "head.0.weight" in ckpt["model"]:
        d_feat = ckpt["model"]["head.0.weight"].shape[0]
        if d_feat == 384 and args.backbone != "repvit_m1":
            print(f"Auto-detecting backbone: shape matches 'repvit_m1' (384 features). Overriding backbone setting.")
            args.backbone = "repvit_m1"
        elif d_feat == 448 and args.backbone != "repvit_m1_0":
            print(f"Auto-detecting backbone: shape matches 'repvit_m1_0' (448 features). Overriding backbone setting.")
            args.backbone = "repvit_m1_0"
        elif d_feat == 512 and args.backbone != "repvit_m1_1":
            print(f"Auto-detecting backbone: shape matches 'repvit_m1_1' (512 features). Overriding backbone setting.")
            args.backbone = "repvit_m1_1"

    # Instantiate PyTorch classifier
    base_model = FlagClassifier(args.backbone, embed_dim=ckpt_embed_dim, pretrained=False)
    base_model.load_state_dict(ckpt["model"])
    
    # Apply RepViT structural reparameterization
    print("Reparameterizing backbone (fusing branches)...")
    base_model.backbone.fuse()
    
    # Wrap with Unified Inference Model
    W_weights = ckpt["criterion"]["W"]
    pytorch_inference_model = FlagClassifierInference(base_model, W_weights, scale=args.scale).to(device)
    pytorch_inference_model.eval()

    # 4. Export to ONNX
    export_to_onnx(pytorch_inference_model, args.onnx_path, device)

    # 5. Evaluate both models
    print("\n" + "=" * 50)
    print("                     STARTING EVALUATION")
    print("=" * 50)
    
    results, metrics = evaluate_models(
        pytorch_inference_model, 
        args.onnx_path, 
        dataloader, 
        device, 
        idx_to_class
    )
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.csv_output, index=False)
    print(f"\n[Logging] Detailed predictions saved to CSV: {args.csv_output}")
    
    # Save directory resolution
    save_dir = os.path.dirname(args.csv_output)
    
    # Per-Class Accuracy Report & Export
    print("\n" + "=" * 50)
    print("                 PER-CLASS ACCURACY REPORT")
    print("=" * 50)
    print(f"{'Class Name':<25} | {'PyTorch Top-1 (%)':<18} | {'ONNX Top-1 (%)':<15}")
    print("-" * 50)
    
    pt_cm = metrics['pt_cm']
    on_cm = metrics['on_cm']
    per_class_data = []
    
    for i, class_name in enumerate(classes):
        pt_total = np.sum(pt_cm[i, :])
        pt_correct = pt_cm[i, i]
        on_total = np.sum(on_cm[i, :])
        on_correct = on_cm[i, i]
        
        pt_acc = (pt_correct / pt_total * 100.0) if pt_total > 0 else 0.0
        on_acc = (on_correct / on_total * 100.0) if on_total > 0 else 0.0
        
        per_class_data.append({
            'class_name': class_name,
            'pytorch_total': int(pt_total),
            'pytorch_correct': int(pt_correct),
            'pytorch_accuracy_pct': float(pt_acc),
            'onnx_total': int(on_total),
            'onnx_correct': int(on_correct),
            'onnx_accuracy_pct': float(on_acc)
        })
        
        # Print first 20 classes or classes with errors to save space, but calculate for all
        if i < 20 or pt_acc < 100.0 or on_acc < 100.0:
            print(f"{class_name:<25} | {pt_acc:>18.2f} | {on_acc:>15.2f}")
            
    if num_classes > 20:
        print(f"... and {num_classes - 20} more classes (fully logged to CSV predictions)")
        
    df_class = pd.DataFrame(per_class_data)
    per_class_path = os.path.join(save_dir, "per_class_accuracy.csv")
    df_class.to_csv(per_class_path, index=False)
    print(f"[Logging] Per-class accuracy exported to CSV: {per_class_path}")

    # ── Rank Distribution Summary & Export ──
    pt_dist = metrics['pt_rank_dist']
    on_dist = metrics['on_rank_dist']
    total_samples = metrics['total_samples']
    
    rank_dist_data = []
    for r in [1, 2, 3, 4, 5, '>5']:
        pt_cnt = pt_dist[r]
        pt_pct = (pt_cnt / total_samples) * 100.0
        on_cnt = on_dist[r]
        on_pct = (on_cnt / total_samples) * 100.0
        rank_dist_data.append({
            'Rank': f"Rank {r}" if isinstance(r, int) else r,
            'pytorch_count': pt_cnt,
            'pytorch_percentage_pct': pt_pct,
            'onnx_count': on_cnt,
            'onnx_percentage_pct': on_pct
        })
        
    df_rank = pd.DataFrame(rank_dist_data)
    rank_dist_path = os.path.join(save_dir, "rank_distribution.csv")
    df_rank.to_csv(rank_dist_path, index=False)
    print(f"[Logging] Rank distribution summary saved to CSV: {rank_dist_path}")
    
    print("\n" + "=" * 65)
    print("                    RANK DISTRIBUTION SUMMARY")
    print("=" * 65)
    print(f"{'Rank':<10} | {'PyTorch Count':<15} | {'PyTorch (%)':<12} | {'ONNX Count':<12} | {'ONNX (%)':<10}")
    print("-" * 65)
    for row in rank_dist_data:
        print(f"{row['Rank']:<10} | {row['pytorch_count']:>15} | {row['pytorch_percentage_pct']:>11.2f}% | {row['onnx_count']:>12} | {row['onnx_percentage_pct']:>9.2f}%")
    print("=" * 65)

    # 6. Plot Confusion Matrices
    plot_confusion_matrix(pt_cm, classes, os.path.join(save_dir, "pytorch_confusion_matrix.png"))
    plot_confusion_matrix(on_cm, classes, os.path.join(save_dir, "onnx_confusion_matrix.png"))

    # 7. Final Summary
    print("\n" + "=" * 65)
    print("                   FINAL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<25} | {'PyTorch Model':<18} | {'ONNX Runtime':<15}")
    print("-" * 65)
    print(f"{'Top-1 Accuracy':<25} | {metrics['pt_top1_acc']:>16.2f}% | {metrics['on_top1_acc']:>13.2f}%")
    print(f"{'Top-5 Accuracy':<25} | {metrics['pt_top5_acc']:>16.2f}% | {metrics['on_top5_acc']:>13.2f}%")
    print(f"{'Average Latency':<25} | {metrics['pt_avg_latency']:>14.2f} ms | {metrics['on_avg_latency']:>11.2f} ms")
    print(f"{'Inference Speed (FPS)':<25} | {metrics['pt_fps']:>14.2f} FPS| {metrics['on_fps']:>11.2f} FPS")
    print("-" * 65)
    speedup = metrics['pt_avg_latency'] / metrics['on_avg_latency']
    print(f"ONNX Speedup Factor: {speedup:.2f}x (lower latency is better)")
    print(f"Total samples evaluated: {metrics['total_samples']}")
    print("=" * 65)


if __name__ == '__main__':
    main()
