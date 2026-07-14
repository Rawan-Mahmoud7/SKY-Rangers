"""
=============================================================================
Flag Classifier — ONNX Runtime Dual-Output Inference
=============================================================================
Loads the RepViT + ArcFace classifier exported as a dual-output ONNX model.

Outputs per forward pass:
    1. probabilities — (batch, 188) softmax over countries
    2. embeddings    — (batch, 512) L2-normalised embeddings

The embeddings are used for Egypt verification; the probabilities provide
Top-K country predictions with confidence scores.

Usage:
    >>> classifier = FlagClassifier(cfg)
    >>> result = classifier.classify(bgr_crop)   # ClassifierResult
    >>> results = classifier.classify_batch([crop1, crop2])
=============================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


# =========================================================================
# Classifier Result
# =========================================================================

@dataclass
class ClassifierResult:
    """
    Result from a single classifier inference.

    Attributes
    ----------
    top_k_labels : list of str
        Country names in descending confidence order.
    top_k_confidences : list of float
        Corresponding confidence scores.
    embedding : np.ndarray
        512-D L2-normalised embedding for similarity matching.
    """
    top_k_labels: List[str] = field(default_factory=list)
    top_k_confidences: List[float] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))


# =========================================================================
# Flag Classifier
# =========================================================================

class FlagClassifier:
    """
    ONNX Runtime flag classifier with dual outputs (probabilities + embeddings).

    Preprocessing matches the training pipeline:
        - Resize to 224×224
        - Normalise with ImageNet mean/std
        - CHW float32 format
    """

    def __init__(self, cfg: dict):
        cls_cfg = cfg.get("classifier", {})
        paths = cfg.get("paths", {})

        self.input_size = cls_cfg.get("input_size", 224)
        self.top_k = cls_cfg.get("top_k", 5)
        self.mean = np.array(
            cls_cfg.get("imagenet_mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        ).reshape(3, 1, 1)
        self.std = np.array(
            cls_cfg.get("imagenet_std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        ).reshape(3, 1, 1)

        # Load class mapping
        mapping_path = paths.get("class_mapping", "class_mapping.json")
        self.idx_to_class = self._load_class_mapping(mapping_path)
        self.num_classes = len(self.idx_to_class)
        logger.info(f"Class mapping loaded: {self.num_classes} classes")

        # Load ONNX model
        model_path = paths.get("classifier_onnx", "best_model.onnx")
        logger.info(f"Loading classifier ONNX model: {model_path}")

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(
            f"Classifier loaded: input_size={self.input_size}, "
            f"outputs={self.output_names}, top_k={self.top_k}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, bgr_crop: np.ndarray) -> ClassifierResult:
        """
        Classify a single BGR crop.

        Parameters
        ----------
        bgr_crop : np.ndarray
            BGR image crop of a detected flag.

        Returns
        -------
        ClassifierResult with top-K labels, confidences, and 512-D embedding.
        """
        input_tensor = self._preprocess(bgr_crop)  # (1, 3, 224, 224)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self._parse_outputs(outputs)

    def classify_batch(self, bgr_crops: List[np.ndarray]) -> List[ClassifierResult]:
        """
        Classify a batch of BGR crops.

        Parameters
        ----------
        bgr_crops : list of np.ndarray
            BGR image crops.

        Returns
        -------
        List of ClassifierResult, one per crop.
        """
        if not bgr_crops:
            return []

        # Stack into batch
        tensors = [self._preprocess(crop) for crop in bgr_crops]
        batch = np.concatenate(tensors, axis=0)  # (N, 3, 224, 224)

        outputs = self.session.run(self.output_names, {self.input_name: batch})

        results = []
        # Handle both single and batch outputs
        probs = outputs[0]  # (N, num_classes)
        embeddings = outputs[1] if len(outputs) > 1 else None  # (N, 512)

        for i in range(len(bgr_crops)):
            result = self._parse_single(
                probs[i],
                embeddings[i] if embeddings is not None else None,
            )
            results.append(result)

        return results

    def extract_embedding(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Extract only the 512-D embedding (for database building).

        Returns
        -------
        np.ndarray of shape (512,), L2-normalised.
        """
        result = self.classify(bgr_crop)
        return result.embedding

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """BGR crop → (1, 3, 224, 224) float32 tensor, ImageNet-normalised."""
        # Resize to input_size
        resized = cv2.resize(
            bgr_image, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # BGR → RGB
        rgb = resized[:, :, ::-1]
        # HWC → CHW, normalise to [0, 1]
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        # ImageNet normalisation
        chw = (chw - self.mean) / self.std
        # Add batch dimension
        return np.expand_dims(chw, axis=0)

    def _parse_outputs(self, outputs: list) -> ClassifierResult:
        """Parse ONNX outputs into ClassifierResult."""
        probs = outputs[0][0]  # (num_classes,) — remove batch dim
        embedding = outputs[1][0] if len(outputs) > 1 else np.zeros(512, dtype=np.float32)
        return self._parse_single(probs, embedding)

    def _parse_single(
        self, probs: np.ndarray, embedding: Optional[np.ndarray]
    ) -> ClassifierResult:
        """Parse a single prediction into ClassifierResult."""
        # Top-K indices
        k = min(self.top_k, len(probs))
        top_k_idx = np.argpartition(probs, -k)[-k:]
        # Sort by confidence descending
        top_k_idx = top_k_idx[np.argsort(probs[top_k_idx])[::-1]]

        top_k_labels = [
            self.idx_to_class.get(int(idx), f"class_{idx}")
            for idx in top_k_idx
        ]
        top_k_confs = [float(probs[idx]) for idx in top_k_idx]

        # Embedding (ensure it exists)
        if embedding is None:
            embedding = np.zeros(512, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)

        return ClassifierResult(
            top_k_labels=top_k_labels,
            top_k_confidences=top_k_confs,
            embedding=embedding,
        )

    @staticmethod
    def _load_class_mapping(path: str) -> Dict[int, str]:
        """Load class_mapping.json → {idx: class_name}."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both formats: {"idx_to_class": {...}} and {"class_to_idx": {...}}
            if "idx_to_class" in data:
                return {int(k): v for k, v in data["idx_to_class"].items()}
            elif "class_to_idx" in data:
                return {v: k for k, v in data["class_to_idx"].items()}
            else:
                logger.error(f"Unknown class_mapping format in {path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load class mapping from {path}: {e}")
            return {}
