"""
=============================================================================
Run UAV Real-Time Event-Driven Vision Pipeline
=============================================================================
Usage:
    python run_pipeline.py [--config config.yaml] [--source video.mp4]
=============================================================================
"""

import argparse
import logging
import os
import sys
import time

# Workaround for OpenMP error "OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import yaml

from core.pipeline import VisionPipeline
from core.events import FlagDetectedEvent, CountryConfirmedEvent, TrackLostEvent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("pipeline")


# =========================================================================
# Flight Controller Dummy Callback Subscriptions
# =========================================================================

def on_flag_detected(event: FlagDetectedEvent):
    """Callback triggered immediately when a new tracked flag is detected."""
    logger.info(
        f"[FLIGHT CONTROL EVENT] >>> Flag Detected! "
        f"Track#{event.track_id} | bbox={event.bbox} | yolo_conf={event.yolo_confidence:.2f}"
    )


def on_country_confirmed(event: CountryConfirmedEvent):
    """Callback triggered when the pipeline confirms a flag's identity."""
    verified_str = " (Verified via Embedding Sim)" if event.egypt_verified else ""
    logger.info(
        f"[FLIGHT CONTROL EVENT] >>> IDENTITY CONFIRMED! "
        f"Track#{event.track_id} = {event.country} (conf={event.confidence:.1%}){verified_str}"
    )


def on_track_lost(event: TrackLostEvent):
    """Callback triggered when a track is lost/expired."""
    logger.info(
        f"[FLIGHT CONTROL EVENT] >>> Track Lost! "
        f"Track#{event.track_id}"
    )


# =========================================================================
# Main Run Loop
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run UAV Event-Driven Vision Pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", help="Path to video file or camera index (e.g., 0)")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {args.config}: {e}")
        sys.exit(1)

    # Setup paths
    source = args.source if args.source is not None else cfg.get("paths", {}).get("camera_source", 0)
    
    # Resolve video source
    try:
        source_idx = int(source)
        cap = cv2.VideoCapture(source_idx)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video source opened: {source} ({width}x{height})")

    # Initialise Pipeline
    try:
        pipeline = VisionPipeline(cfg, frame_size=(width, height))
    except Exception as e:
        logger.error(f"Failed to initialize VisionPipeline: {e}")
        logger.error("Please ensure you have run 'python run_export_models.py' to generate ONNX models.")
        cap.release()
        sys.exit(1)
    
    # Subscribe flight controller simulation callbacks to the event bus
    pipeline.event_bus.subscribe("FLAG_DETECTED", on_flag_detected)
    pipeline.event_bus.subscribe("COUNTRY_CONFIRMED", on_country_confirmed)
    pipeline.event_bus.subscribe("TRACK_LOST", on_track_lost)

    logger.info("Pipeline starting. Press Ctrl+C to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            # Run pipeline for one frame (handles YOLO, Tracker, Classifier, Decision, Events)
            annotated_frame, events = pipeline.process_frame(frame)
            
            # If a user wants to view it live, they can uncomment:
            # cv2.imshow("UAV Vision", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        pipeline.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
