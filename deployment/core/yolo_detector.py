"""
=============================================================================
YOLO Detector — ONNX Runtime Inference
=============================================================================
Loads a YOLO ONNX model and runs detection with pure NumPy NMS.
No PyTorch or ultralytics dependency at runtime.

Supports YOLOv8/v11 ONNX format exported via ultralytics model.export().

Usage:
    >>> detector = YOLODetector(cfg)
    >>> detections = detector.detect(frame)  # List[Detection]
=============================================================================
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


# =========================================================================
# Detection Dataclass
# =========================================================================

@dataclass
class Detection:
    """A single YOLO detection in one frame."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    confidence: float
    class_id: int = 0
    frame_idx: int = 0


# =========================================================================
# YOLO Detector
# =========================================================================

class YOLODetector:
    """
    ONNX Runtime YOLO detector with letterbox preprocessing and NumPy NMS.

    Parameters are loaded from the config dict under the 'yolo' and 'paths' keys.
    """

    def __init__(self, cfg: dict):
        yolo_cfg = cfg.get("yolo", {})
        paths = cfg.get("paths", {})

        self.conf_threshold = yolo_cfg.get("confidence_threshold", 0.7)
        self.iou_threshold = yolo_cfg.get("iou_threshold", 0.45)
        self.input_size = yolo_cfg.get("input_size", 640)
        self.max_detections = yolo_cfg.get("max_detections", 3)

        # Load ONNX model
        model_path = paths.get("yolo_onnx", "best.onnx")
        logger.info(f"Loading YOLO ONNX model: {model_path}")

        # Use CPU on Raspberry Pi; allow CUDA if available
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Determine model input shape
        input_shape = self.session.get_inputs()[0].shape
        if input_shape and len(input_shape) == 4:
            self._model_h = input_shape[2] if isinstance(input_shape[2], int) else self.input_size
            self._model_w = input_shape[3] if isinstance(input_shape[3], int) else self.input_size
        else:
            self._model_h = self.input_size
            self._model_w = self.input_size

        logger.info(
            f"YOLO loaded: input={self._model_w}x{self._model_h}, "
            f"conf={self.conf_threshold}, iou={self.iou_threshold}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """
        Run YOLO detection on a BGR frame.

        Returns a list of Detection objects, sorted by confidence (descending),
        capped at max_detections.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: letterbox + normalize
        input_tensor, ratio, pad_w, pad_h = self._preprocess(frame)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        raw_output = outputs[0]  # (1, num_preds, 4+num_classes) or (1, 4+num_classes, num_preds)

        # Parse detections
        detections = self._postprocess(
            raw_output, ratio, pad_w, pad_h, orig_w, orig_h, frame_idx
        )

        # Sort by confidence, take top-N
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Letterbox resize + normalize to [0, 1] float32.

        Returns (input_tensor, ratio, pad_w, pad_h).
        """
        orig_h, orig_w = frame.shape[:2]

        # Compute scale to fit within model input
        ratio = min(self._model_w / orig_w, self._model_h / orig_h)
        new_w = int(round(orig_w * ratio))
        new_h = int(round(orig_h * ratio))

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to model input size (center padding)
        pad_w = (self._model_w - new_w) / 2.0
        pad_h = (self._model_h - new_h) / 2.0
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Ensure exact size (rounding edge case)
        if padded.shape[0] != self._model_h or padded.shape[1] != self._model_w:
            padded = cv2.resize(padded, (self._model_w, self._model_h))

        # BGR → RGB, HWC → CHW, normalize to [0, 1], add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # (1, 3, H, W)

        return blob, ratio, pad_w, pad_h

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw_output: np.ndarray,
        ratio: float,
        pad_w: float,
        pad_h: float,
        orig_w: int,
        orig_h: int,
        frame_idx: int,
    ) -> List[Detection]:
        """
        Parse YOLO output, apply NMS, scale boxes back to original frame.

        Handles both YOLOv8 output formats:
          - (1, num_preds, 4+num_classes)  — row-major
          - (1, 4+num_classes, num_preds)  — column-major (needs transpose)
        """
        output = raw_output[0]  # Remove batch dim → (num_preds, 4+C) or (4+C, num_preds)

        # Auto-detect format: if dim 1 > dim 0, it's column-major
        if output.shape[0] < output.shape[1]:
            output = output.T  # → (num_preds, 4+C)

        # Split into boxes and class scores
        boxes_xywh = output[:, :4]       # (N, 4) — center_x, center_y, w, h
        class_scores = output[:, 4:]     # (N, num_classes)

        # Best class per prediction
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter by confidence
        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Scale back to original image coordinates
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio

        # Clip to frame bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        # NMS
        keep = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        detections = []
        for i in keep:
            det = Detection(
                bbox=(
                    int(round(boxes_xyxy[i, 0])),
                    int(round(boxes_xyxy[i, 1])),
                    int(round(boxes_xyxy[i, 2])),
                    int(round(boxes_xyxy[i, 3])),
                ),
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
                frame_idx=frame_idx,
            )
            detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Pure NumPy NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        """
        Non-Maximum Suppression in pure NumPy.

        Parameters
        ----------
        boxes : (N, 4) array of [x1, y1, x2, y2]
        scores : (N,) array of confidence scores
        iou_threshold : float

        Returns
        -------
        List[int] — indices of kept boxes
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            # IoU of current best vs remaining
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]  # +1 because we skipped index 0

        return keep
