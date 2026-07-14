"""
=============================================================================
Main Real-Time Pipeline Orchestrator — Event-Driven
=============================================================================
Connects the camera, YOLO, Tracker, Crop Extractor, Classifier, Temporal Fusion,
Decision Engine, and Visualizer into a single cohesive frame loop.

This module is designed to be imported by a runner script (run_pipeline.py)
that provides the actual frame loop. It runs only ONNX models via ONNX Runtime
and outputs structured VisionEvents.
=============================================================================
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .events import EventBus, FlagDetectedEvent, CountryConfirmedEvent, TrackLostEvent, VisionEvent
from .yolo_detector import Detection, YOLODetector
from .classifier import ClassifierResult, FlagClassifier
from .tracker import Track, TrackState, TargetTracker
from .temporal_fusion import TemporalFusion, FusionResult
from .egypt_verifier import EgyptVerifier
from .decision_engine import DecisionEngine
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    Orchestrates the real-time event-driven target identification pipeline.

    Loads the YOLO detector, Flag classifier, Tracker, Temporal Fusion,
    Egypt Embedding Verifier, Decision Engine, and Visualizer.
    """

    def __init__(self, cfg: dict, frame_size: Tuple[int, int] = (1280, 720)):
        self.cfg = cfg
        self.frame_size = frame_size

        logger.info("Initialising Event-Driven Pipeline Components...")
        # 1. Event Bus
        self.event_bus = EventBus()

        # 2. YOLO Detector
        self.detector = YOLODetector(cfg)

        # 3. Classifier
        self.classifier = FlagClassifier(cfg)

        # 4. Tracker (OC-SORT)
        self.tracker = TargetTracker(cfg)

        # 5. Temporal Fusion
        self.fusion = TemporalFusion(cfg)

        # 6. Egypt Verifier
        self.egypt_verifier = EgyptVerifier(cfg)

        # 7. Decision Engine
        self.decision_engine = DecisionEngine(cfg)

        # 8. Visualizer
        self.visualizer = Visualizer(cfg, frame_size=frame_size)

        # Pipeline state
        self.frame_idx = 0
        self.start_time = time.time()
        self.events_log: List[dict] = []

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[VisionEvent]]:
        """
        Run the full pipeline on a single frame.

        Returns
        -------
        annotated_frame : np.ndarray
        events : List[VisionEvent]
            List of vision events emitted during this frame.
        """
        self.frame_idx += 1
        t0 = time.perf_counter()
        events: List[VisionEvent] = []

        # 1. YOLO Detection (runs on every frame)
        detections = self.detector.detect(frame, self.frame_idx)

        # 2. Tracker Update
        # Tracker returns: active_tracks, new_track_ids, lost_track_ids
        active_tracks, new_track_ids, lost_track_ids = self.tracker.update(detections, self.frame_idx)

        # 3. Emit FLAG_DETECTED events for new tracks
        for tid in new_track_ids:
            track = self.tracker.get_track(tid)
            if track:
                ev = FlagDetectedEvent(
                    track_id=tid,
                    frame_idx=self.frame_idx,
                    bbox=track.bbox,
                    yolo_confidence=track.yolo_conf
                )
                events.append(ev)
                self.event_bus.emit(ev)

        # 4. Emit TRACK_LOST events for expired tracks
        for tid in lost_track_ids:
            # We don't have the track object in the tracker anymore, but we can emit a lost event
            ev = TrackLostEvent(
                track_id=tid,
                frame_idx=self.frame_idx
            )
            events.append(ev)
            self.event_bus.emit(ev)

        # 5. Crop and classify for active, unconfirmed tracks
        h, w = frame.shape[:2]
        for track in active_tracks:
            # If already confirmed, don't classify it again
            if track.confirmed_country is not None:
                continue

            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                
                # Classify (returns probabilities & 512-D embedding)
                cls_result = self.classifier.classify(crop)

                # Temporal Fusion
                fusion_result = self.fusion.update(track, cls_result)

                # Decision Engine
                decision_event = self.decision_engine.evaluate(
                    track, cls_result, fusion_result, self.egypt_verifier, self.frame_idx
                )

                if decision_event:
                    events.append(decision_event)
                    self.event_bus.emit(decision_event)

        # 6. Visualization
        fps = 1.0 / (time.perf_counter() - t0 + 1e-6)
        annotated_frame = self.visualizer.draw(
            frame, active_tracks, events, self.frame_idx, fps
        )

        # Accumulate events for JSON log
        for ev in events:
            ev_dict = {
                "frame": ev.frame_idx,
                "track": ev.track_id,
                "type": ev.event_type,
                "time": ev.timestamp
            }
            if isinstance(ev, FlagDetectedEvent):
                ev_dict["bbox"] = ev.bbox
                ev_dict["yolo_conf"] = ev.yolo_confidence
            elif isinstance(ev, CountryConfirmedEvent):
                ev_dict["country"] = ev.country
                ev_dict["confidence"] = ev.confidence
                ev_dict["egypt_verified"] = ev.egypt_verified
            self.events_log.append(ev_dict)

        return annotated_frame, events

    def shutdown(self) -> None:
        """Cleanup visualizer and save event log."""
        self.visualizer.release()
        
        # Save event log
        log_path = "./vision_events.json"
        try:
            with open(log_path, "w") as f:
                json.dump(self.events_log, f, indent=2)
            logger.info(f"Pipeline shutdown complete. Events saved to {log_path}.")
        except Exception as e:
            logger.error(f"Failed to save event log to {log_path}: {e}")
