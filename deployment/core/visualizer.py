"""
=============================================================================
Visual Debugger — Headless OpenCV Overlay
=============================================================================
Draws track bounding boxes, classification confidences, event logs, and HUD
overlays onto video frames using OpenCV primitives.
=============================================================================
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .events import FlagDetectedEvent, CountryConfirmedEvent, TrackLostEvent, VisionEvent
from .tracker import Track, TrackState

logger = logging.getLogger(__name__)

# BGR color defaults
_DEFAULT_COLORS = {
    TrackState.TRACKING: (0, 255, 255),       # Yellow
    TrackState.CONFIRMED: (0, 255, 0),        # Green
    TrackState.LOST: (128, 128, 128),         # Gray
}


class Visualizer:
    """
    Draws debug overlays and writes output video.
    """

    def __init__(self, cfg: dict, frame_size: Tuple[int, int] = (1280, 720)):
        vis_cfg = cfg.get("visualizer", {})
        self.enabled = vis_cfg.get("enabled", True)
        self.font_scale = vis_cfg.get("font_scale", 0.5)
        self.thickness = vis_cfg.get("line_thickness", 2)
        self.show_hud = vis_cfg.get("show_hud", True)
        self.show_log = vis_cfg.get("show_event_log", True)
        self.log_max = vis_cfg.get("event_log_max", 5)

        # State color mapping
        raw_colors = vis_cfg.get("colors", {})
        self.colors: Dict[TrackState, Tuple[int, int, int]] = {}
        for state in TrackState:
            key = state.value.lower()
            if key in raw_colors:
                self.colors[state] = tuple(raw_colors[key])
            else:
                self.colors[state] = _DEFAULT_COLORS.get(state, (255, 255, 255))

        # Video writer
        self.writer: Optional[cv2.VideoWriter] = None
        output_path = vis_cfg.get("output_video", "./debug_output.mp4")
        fps = vis_cfg.get("video_fps", 15)
        if self.enabled and output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = frame_size
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            logger.info(f"Visualizer writing to {output_path} @ {fps} FPS")

        # Event log ring buffer
        self._log_entries: deque = deque(maxlen=50)

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        events: List[VisionEvent],
        frame_idx: int,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Draw all overlays onto frame (in-place) and write to video.
        """
        if not self.enabled:
            return frame

        # Record new events for log
        for ev in events:
            self._log_entries.append(ev)

        # Draw per-track overlays
        for track in tracks:
            self._draw_track(frame, track)

        # Draw global HUD
        if self.show_hud:
            self._draw_hud(frame, tracks, frame_idx, fps)

        # Draw event log
        if self.show_log:
            self._draw_event_log(frame)

        # Write frame
        if self.writer is not None:
            self.writer.write(frame)

        return frame

    def release(self) -> None:
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            logger.info("Visualizer video writer released.")

    def _draw_track(self, frame: np.ndarray, track: Track) -> None:
        """Draw bbox + info panel for one track."""
        x1, y1, x2, y2 = track.bbox
        color = self.colors.get(track.state, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Prepare details
        lines = [
            f"Track#{track.track_id} | {track.state.value}",
            f"YOLO:{track.yolo_conf:.2f} | Size:{x2-x1}x{y2-y1}"
        ]

        if track.confirmed_country is not None:
            lines.append(f"Identity: {track.confirmed_country}")
            lines.append(f"Conf: {track.confirmed_confidence:.1%} {'(Verified)' if track.egypt_verified else ''}")
        else:
            if track.best_country:
                lines.append(f"Guess: {track.best_country} ({track.vote_fraction:.0%} votes)")
                lines.append(f"Fused Conf: {track.best_confidence:.3f}")
            else:
                lines.append("Classifying...")

            # Streak indicator
            if track.streak_high > 0:
                lines.append(f"Streak High: {track.streak_high}")
            elif track.streak_low > 0:
                lines.append(f"Streak Low: {track.streak_low}")

        # Draw label background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.font_scale
        th = 1
        padding = 4
        line_h = int(20 * fs / 0.5)

        # Position label above or below bbox depending on vertical clearance
        label_y_start = y1 - len(lines) * line_h - padding
        if label_y_start < 0:
            label_y_start = y2 + padding

        # Measure text width
        max_text_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, fs, th)
            max_text_w = max(max_text_w, tw)

        bg_x1 = x1
        bg_y1 = label_y_start
        bg_x2 = x1 + max_text_w + 2 * padding
        bg_y2 = label_y_start + len(lines) * line_h + padding

        # semi-transparent background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text lines
        for i, line in enumerate(lines):
            ty = label_y_start + (i + 1) * line_h
            cv2.putText(frame, line, (x1 + padding, ty), font, fs, color, th, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_idx: int,
        fps: float,
    ) -> None:
        """Draw the top-of-frame HUD status bar."""
        h, w = frame.shape[:2]

        bar_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        state_counts: Dict[str, int] = {}
        for t in tracks:
            s = t.state.value
            state_counts[s] = state_counts.get(s, 0) + 1

        summary_parts = [f"{count}x{state}" for state, count in state_counts.items()]
        summary = ", ".join(summary_parts) if summary_parts else "No tracked flags"

        text = f"Frame:{frame_idx} | FPS:{fps:.1f} | Active Tracks:{len(tracks)} | [{summary}]"
        cv2.putText(
            frame, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_event_log(self, frame: np.ndarray) -> None:
        """Draw recent events log at the bottom of the frame."""
        h, w = frame.shape[:2]
        entries = list(self._log_entries)[-self.log_max:]

        if not entries:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.4
        line_h = 18
        padding = 6

        # Background
        log_h = len(entries) * line_h + 2 * padding
        bg_y1 = h - log_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bg_y1), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, ev in enumerate(entries):
            if isinstance(ev, FlagDetectedEvent):
                color = (0, 255, 255)  # Yellow
                text = f"[F{ev.frame_idx}] Event: FLAG_DETECTED | Track#{ev.track_id} (conf={ev.yolo_confidence:.2f})"
            elif isinstance(ev, CountryConfirmedEvent):
                color = (0, 255, 0)    # Green
                text = f"[F{ev.frame_idx}] Event: COUNTRY_CONFIRMED | Track#{ev.track_id} = {ev.country} ({ev.confidence:.1%}) {'[EGYPT VERIFIED]' if ev.egypt_verified else ''}"
            elif isinstance(ev, TrackLostEvent):
                color = (128, 128, 128) # Gray
                text = f"[F{ev.frame_idx}] Event: TRACK_LOST | Track#{ev.track_id}"
            else:
                color = (255, 255, 255)
                text = f"[F{ev.frame_idx}] Event: {ev.event_type} | Track#{ev.track_id}"

            ty = bg_y1 + padding + (i + 1) * line_h
            cv2.putText(frame, text, (10, ty), font, fs, color, 1, cv2.LINE_AA)
