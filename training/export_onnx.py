#!/usr/bin/env python3
"""
ONNX Export Script for Flag Classifier
Loads best_model.pth, fuses RepViT backbone structures,
integrates the ArcFace weight matrix for unified classification inference,
and exports the final model to ONNX format.
$env:PYTHONIOENCODING="utf-8"
python export_onnx.py --model-path ./flag_model2/best_model.pth --onnx-path ./flag_model2/best_model.onnx

"""

import os
import argparse
import sys
import io

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError when printing emojis/special characters
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-path", default="./flag_model2/best_model.pth", help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--onnx-path", default="./flag_model2/best_model.onnx", help="Path to save output ONNX model")
    parser.add_argument("--backbone", default="repvit_m1", help="Timm backbone name")
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace scale parameter")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)

    # 1. Load checkpoint
    print(f"Loading checkpoint from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.model_path}")
    
    ckpt = torch.load(args.model_path, map_location='cpu')

    # 2. Detect dimensions
    W_weights = ckpt["criterion"]["W"]
    ckpt_num_classes = W_weights.shape[1]
    ckpt_embed_dim = W_weights.shape[0]
    print(f"Detected checkpoint: classes={ckpt_num_classes}, embed_dim={ckpt_embed_dim}")

    # 3. Instantiate model
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

    print(f"Instantiating model with backbone '{args.backbone}'...")
    base_model = FlagClassifier(args.backbone, embed_dim=ckpt_embed_dim, pretrained=False)
    base_model.load_state_dict(ckpt["model"])
    base_model.to(device)

    # 4. Apply RepViT structural reparameterization
    if hasattr(base_model.backbone, 'fuse'):
        print("Fusing backbone layers (RepViT structural reparameterization)...")
        base_model.backbone.fuse()
    else:
        print("Backbone does not support fusing or is already fused.")

    # 5. Wrap with Inference model
    print("Wrapping model with unified inference layer (Softmax probabilities)...")
    pytorch_inference_model = FlagClassifierInference(base_model, W_weights.to(device), scale=args.scale)
    pytorch_inference_model.eval()

    # 6. Export to ONNX
    print(f"Exporting to ONNX at: {args.onnx_path}")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    torch.onnx.export(
        pytorch_inference_model,
        dummy_input,
        args.onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 7. Verification and Consolidation
    try:
        import onnx
        print("Loading and verifying ONNX model structure...")
        onnx_model = onnx.load(args.onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model successfully verified!")

        # Consolidate external data files if they exist (common in some PyTorch 2.x versions)
        data_file_path = args.onnx_path + ".data"
        if os.path.exists(data_file_path):
            print("Consolidating model weights into a single self-contained ONNX file...")
            onnx.save(onnx_model, args.onnx_path)
            # Remove the external data file
            os.remove(data_file_path)
            print("✓ Removed external data file. Weights are now fully embedded in the ONNX file.")
    except ImportError:
        print("Warning: 'onnx' library not found. Verification/Consolidation skipped.")
    
    print("\nDone! ONNX model exported successfully.")


if __name__ == "__main__":
    main()
