"""
=============================================================================
OC-SORT Target Tracker — Cleaned for Event-Driven Pipeline
=============================================================================
Observation-Centric SORT multi-object tracker with three key innovations:

    ORU (Observation-Centric Re-Update)
        Corrects Kalman drift after detection gaps using real observations.

    OCM (Observation-Centric Momentum)
        Uses observation-derived velocity for directional cost modulation.

    OCR (Observation-Centric Recovery)
        Second-chance matching using last real observation instead of
        drifted Kalman prediction.

Changes from previous version:
    - Removed DINOv2 fields (sim_wk, margin, vote_frac, best_target, etc.)
    - Simplified track state: TRACKING / CONFIRMED / LOST
    - Added `is_new` flag for FLAG_DETECTED event emission
    - Added `confirmed_country` for classifier locking
    - update() returns (active_tracks, new_track_ids, lost_track_ids)

Public API:
    Detection, Track, TrackState
    TargetTracker.update(detections, frame_idx) → (active, new_ids, lost_ids)
    TargetTracker.get_track(track_id) → Track
    TargetTracker.get_all_tracks() → List[Track]
=============================================================================
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .yolo_detector import Detection

logger = logging.getLogger(__name__)


# =========================================================================
# Track State
# =========================================================================

class TrackState(Enum):
    """Vision-only track states (no flight states)."""
    TRACKING = "TRACKING"       # Active, being classified
    CONFIRMED = "CONFIRMED"     # Country confirmed, classifier locked
    LOST = "LOST"               # Track expired


# =========================================================================
# Track
# =========================================================================

@dataclass
class Track:
    """
    Persistent track for a single detection across frames.

    Contains temporal history needed by the decision engine.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]         # Latest bbox (x1, y1, x2, y2)
    yolo_conf: float = 0.0                   # Latest YOLO confidence
    state: TrackState = TrackState.TRACKING

    # --- Classifier result (set by decision engine) ---
    confirmed_country: Optional[str] = None  # Set once → stops classifier
    confirmed_confidence: float = 0.0
    egypt_verified: bool = False             # True if confirmed via embedding

    # --- Temporal fusion state (managed by TemporalFusion) ---
    # Stored here so state follows the track's lifecycle
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    c_temporal: float = 0.0                  # EMA-smoothed confidence
    streak_high: int = 0
    streak_low: int = 0
    best_country: str = ""                   # Current best guess from fusion
    best_confidence: float = 0.0             # Current best confidence from fusion
    vote_fraction: float = 0.0               # How consistently top country is picked

    # --- Lifecycle ---
    is_new: bool = True                      # True on creation frame only
    frames_since_update: int = 0
    total_frames: int = 0
    frame_created: int = 0


# =========================================================================
# Internal: Kalman Filter
# =========================================================================

def _bbox_to_z(bbox) -> np.ndarray:
    """Convert [x1, y1, x2, y2] → [cx, cy, area, aspect_ratio]."""
    bbox = np.asarray(bbox, dtype=np.float64).flatten()
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def _z_to_bbox(z) -> np.ndarray:
    """Convert [cx, cy, area, aspect_ratio] → [x1, y1, x2, y2]."""
    z = z.flatten()
    cx, cy = z[0], z[1]
    s = max(z[2], 1.0)
    r = max(z[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _KalmanBoxTracker:
    """
    Internal Kalman filter for a single tracked object.

    State: [cx, cy, s, r, vx, vy, vs]
    Measurement: [cx, cy, s, r]
    """

    def __init__(self, bbox: np.ndarray, delta_t: int = 3):
        z = _bbox_to_z(bbox)

        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = z

        self.P = np.eye(7, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0

        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7), dtype=np.float64)
        np.fill_diagonal(self.H[:4, :4], 1.0)

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        self.R = np.eye(4, dtype=np.float64)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 0

        self.observations: Dict[int, np.ndarray] = {0: bbox.copy()}
        self.last_observation: np.ndarray = bbox.copy()
        self._max_obs = 50

        self.velocity = np.zeros(2, dtype=np.float64)
        self.delta_t = delta_t

    def predict(self) -> np.ndarray:
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: np.ndarray) -> None:
        self.observations[self.age] = bbox.copy()
        self.last_observation = bbox.copy()
        self._prune_observations()
        self._update_velocity()

        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def re_update(self, bbox: np.ndarray) -> None:
        """ORU: correct Kalman drift after a detection gap."""
        if not self.observations:
            self.update(bbox)
            return

        last_obs_age = max(self.observations.keys())
        last_obs = self.observations[last_obs_age]
        gap = self.age - last_obs_age

        if gap <= 1:
            self.update(bbox)
            return

        z_last = _bbox_to_z(last_obs)
        z_new = _bbox_to_z(bbox)

        self.x[:4] = z_last
        self.x[4, 0] = (z_new[0, 0] - z_last[0, 0]) / gap
        self.x[5, 0] = (z_new[1, 0] - z_last[1, 0]) / gap
        self.x[6, 0] = (z_new[2, 0] - z_last[2, 0]) / gap

        self.P = np.eye(7, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0
        self.update(bbox)

    def _update_velocity(self) -> None:
        """OCM: velocity from real observations."""
        ages = sorted(self.observations.keys())
        if len(ages) < 2:
            self.velocity = np.zeros(2, dtype=np.float64)
            return

        current_age = ages[-1]
        target_age = current_age - self.delta_t
        prev_age = None
        for a in reversed(ages[:-1]):
            if a <= target_age:
                prev_age = a
                break
        if prev_age is None:
            prev_age = ages[-2]

        prev_obs = self.observations[prev_age]
        curr_obs = self.observations[current_age]
        dt = max(current_age - prev_age, 1)

        prev_cx = (prev_obs[0] + prev_obs[2]) / 2.0
        prev_cy = (prev_obs[1] + prev_obs[3]) / 2.0
        curr_cx = (curr_obs[0] + curr_obs[2]) / 2.0
        curr_cy = (curr_obs[1] + curr_obs[3]) / 2.0

        self.velocity = np.array([
            (curr_cx - prev_cx) / dt,
            (curr_cy - prev_cy) / dt,
        ], dtype=np.float64)

    def get_state(self) -> np.ndarray:
        return _z_to_bbox(self.x[:4])

    def _prune_observations(self) -> None:
        if len(self.observations) > self._max_obs:
            ages = sorted(self.observations.keys())
            for a in ages[: len(ages) - self._max_obs]:
                del self.observations[a]


# =========================================================================
# OC-SORT Target Tracker
# =========================================================================

class TargetTracker:
    """
    OC-SORT multi-object tracker.

    update() returns (active_tracks, new_track_ids, lost_track_ids) so the
    pipeline can emit FLAG_DETECTED and TRACK_LOST events.
    """

    def __init__(self, cfg: dict):
        tracker_cfg = cfg.get("tracker", {})

        self.iou_threshold: float = tracker_cfg.get("iou_threshold", 0.3)
        self.max_age: int = tracker_cfg.get("max_age", 30)
        self.min_hits: int = tracker_cfg.get("min_hits", 1)

        self.delta_t: int = tracker_cfg.get("delta_t", 3)
        self.inertia: float = tracker_cfg.get("inertia", 0.2)
        self.use_oru: bool = tracker_cfg.get("use_oru", True)
        self.use_ocm: bool = tracker_cfg.get("use_ocm", True)
        self.use_ocr: bool = tracker_cfg.get("use_ocr", True)
        self.ocr_iou_threshold: float = tracker_cfg.get("ocr_iou_threshold", 0.2)
        self.max_center_dist: float = tracker_cfg.get("max_center_dist", 150.0)

        self.tracks: Dict[int, Track] = {}
        self._kf_trackers: Dict[int, _KalmanBoxTracker] = {}
        self._next_id: int = 0

        logger.info(
            f"OC-SORT tracker initialised: iou={self.iou_threshold}, "
            f"max_age={self.max_age}, ORU={self.use_oru}, OCM={self.use_ocm}, OCR={self.use_ocr}"
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Detection], frame_idx: int
    ) -> Tuple[List[Track], List[int], List[int]]:
        """
        Match new detections to existing tracks.

        Returns
        -------
        active_tracks : List[Track]
            Tracks updated this frame (and meeting min_hits).
        new_track_ids : List[int]
            IDs of tracks created this frame (→ FLAG_DETECTED).
        lost_track_ids : List[int]
            IDs of tracks expired this frame (→ TRACK_LOST).
        """
        new_track_ids: List[int] = []
        lost_track_ids: List[int] = []

        # Clear is_new flags from previous frame
        for track in self.tracks.values():
            track.is_new = False

        # Step 1: Predict all Kalman trackers
        track_ids = list(self._kf_trackers.keys())
        predicted_bboxes: Dict[int, np.ndarray] = {}
        for tid in track_ids:
            predicted_bboxes[tid] = self._kf_trackers[tid].predict()

        # Step 2: Primary association (Hungarian + IoU)
        det_bboxes = [np.array(d.bbox, dtype=np.float64) for d in detections]

        matched, unmatched_det_idxs, unmatched_trk_idxs = (
            self._primary_association(track_ids, predicted_bboxes, det_bboxes)
        )

        # Step 3: OCR recovery
        if self.use_ocr and unmatched_trk_idxs and unmatched_det_idxs:
            ocr_matched, still_unmatched_dets, still_unmatched_trks = (
                self._ocr_association(
                    track_ids, unmatched_trk_idxs, det_bboxes, unmatched_det_idxs
                )
            )
            matched.extend(ocr_matched)
            unmatched_det_idxs = still_unmatched_dets
            unmatched_trk_idxs = still_unmatched_trks

        # Step 4: Update matched tracks
        for trk_idx, det_idx in matched:
            tid = track_ids[trk_idx]
            det = detections[det_idx]
            det_bbox = np.array(det.bbox, dtype=np.float64)
            kft = self._kf_trackers[tid]

            if self.use_oru and kft.time_since_update > 1:
                kft.re_update(det_bbox)
            else:
                kft.update(det_bbox)

            track = self.tracks[tid]
            kf_bbox = kft.get_state()
            track.bbox = self._clip_bbox(kf_bbox)
            track.yolo_conf = det.confidence
            track.frames_since_update = 0
            track.total_frames += 1

        # Step 5: Create new tracks
        for det_idx in unmatched_det_idxs:
            det = detections[det_idx]
            det_bbox = np.array(det.bbox, dtype=np.float64)
            tid = self._create_track(det, det_bbox, frame_idx)
            new_track_ids.append(tid)

        # Step 6: Age unmatched tracks + expire
        expired = []
        for trk_idx in unmatched_trk_idxs:
            tid = track_ids[trk_idx]
            self.tracks[tid].frames_since_update += 1
            self._kf_trackers[tid].hit_streak = 0

            if self.tracks[tid].frames_since_update > self.max_age:
                expired.append(tid)

        for tid in expired:
            self.tracks[tid].state = TrackState.LOST
            lost_track_ids.append(tid)
            logger.debug(f"Track #{tid} expired (age > {self.max_age})")
            del self.tracks[tid]
            del self._kf_trackers[tid]

        # Return active tracks
        active = [
            t for t in self.tracks.values()
            if t.frames_since_update == 0
            and self._kf_trackers[t.track_id].hits >= self.min_hits
        ]

        return active, new_track_ids, lost_track_ids

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[Track]:
        return list(self.tracks.values())

    def remove_track(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)
        self._kf_trackers.pop(track_id, None)

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _primary_association(
        self,
        track_ids: List[int],
        predicted_bboxes: Dict[int, np.ndarray],
        det_bboxes: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not det_bboxes:
            return ([], list(range(len(det_bboxes))), list(range(len(track_ids))))

        pred_boxes = [predicted_bboxes[tid] for tid in track_ids]
        iou_matrix = self._compute_iou_matrix(pred_boxes, det_bboxes)

        if self.use_ocm:
            iou_matrix = self._apply_ocm(iou_matrix, track_ids, pred_boxes, det_bboxes)

        cost_matrix = 1.0 - iou_matrix

        pred_centers = np.array([
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            for box in pred_boxes
        ])
        det_centers = np.array([
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            for box in det_bboxes
        ])
        dist_matrix = np.linalg.norm(
            pred_centers[:, None, :] - det_centers[None, :, :], axis=-1
        )

        for i in range(len(track_ids)):
            for j in range(len(det_bboxes)):
                if iou_matrix[i, j] < self.iou_threshold:
                    if dist_matrix[i, j] <= self.max_center_dist:
                        cost_matrix[i, j] = 1.0 + (dist_matrix[i, j] / self.max_center_dist)
                    else:
                        cost_matrix[i, j] = 1.0 + 1e-5

        row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(len(det_bboxes)))
        unmatched_trks = set(range(len(track_ids)))

        for r, c in zip(row_idxs, col_idxs):
            if iou_matrix[r, c] >= self.iou_threshold or dist_matrix[r, c] <= self.max_center_dist:
                matched.append((r, c))
                unmatched_dets.discard(c)
                unmatched_trks.discard(r)

        return matched, list(unmatched_dets), list(unmatched_trks)

    def _ocr_association(
        self,
        track_ids: List[int],
        unmatched_trk_idxs: List[int],
        det_bboxes: List[np.ndarray],
        unmatched_det_idxs: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        last_obs_boxes = [
            self._kf_trackers[track_ids[i]].last_observation
            for i in unmatched_trk_idxs
        ]
        rem_det_boxes = [det_bboxes[j] for j in unmatched_det_idxs]

        iou_matrix = self._compute_iou_matrix(last_obs_boxes, rem_det_boxes)
        cost_matrix = 1.0 - iou_matrix

        pred_centers = np.array([
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            for box in last_obs_boxes
        ])
        det_centers = np.array([
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            for box in rem_det_boxes
        ])

        if pred_centers.size > 0 and det_centers.size > 0:
            dist_matrix = np.linalg.norm(
                pred_centers[:, None, :] - det_centers[None, :, :], axis=-1
            )
        else:
            dist_matrix = np.empty((0, 0))

        for i in range(len(unmatched_trk_idxs)):
            for j in range(len(unmatched_det_idxs)):
                if iou_matrix[i, j] < self.ocr_iou_threshold:
                    if dist_matrix.size > 0 and dist_matrix[i, j] <= self.max_center_dist:
                        cost_matrix[i, j] = 1.0 + (dist_matrix[i, j] / self.max_center_dist)
                    else:
                        cost_matrix[i, j] = 1.0 + 1e-5

        if cost_matrix.size == 0:
            return [], list(unmatched_det_idxs), list(unmatched_trk_idxs)

        row_idxs, col_idxs = linear_sum_assignment(cost_matrix)

        matched = []
        still_unmatched_dets = set(unmatched_det_idxs)
        still_unmatched_trks = set(unmatched_trk_idxs)

        for r, c in zip(row_idxs, col_idxs):
            if iou_matrix[r, c] >= self.ocr_iou_threshold or (
                dist_matrix.size > 0 and dist_matrix[r, c] <= self.max_center_dist
            ):
                orig_trk_idx = unmatched_trk_idxs[r]
                orig_det_idx = unmatched_det_idxs[c]
                matched.append((orig_trk_idx, orig_det_idx))
                still_unmatched_dets.discard(orig_det_idx)
                still_unmatched_trks.discard(orig_trk_idx)

        return matched, list(still_unmatched_dets), list(still_unmatched_trks)

    def _apply_ocm(
        self,
        iou_matrix: np.ndarray,
        track_ids: List[int],
        pred_boxes: List[np.ndarray],
        det_bboxes: List[np.ndarray],
    ) -> np.ndarray:
        iou_ocm = iou_matrix.copy()
        for i, tid in enumerate(track_ids):
            kft = self._kf_trackers[tid]
            vel_norm = np.linalg.norm(kft.velocity)
            if vel_norm < 1e-4:
                continue

            pred_cx = (pred_boxes[i][0] + pred_boxes[i][2]) / 2.0
            pred_cy = (pred_boxes[i][1] + pred_boxes[i][3]) / 2.0

            for j in range(len(det_bboxes)):
                det_cx = (det_bboxes[j][0] + det_bboxes[j][2]) / 2.0
                det_cy = (det_bboxes[j][1] + det_bboxes[j][3]) / 2.0
                diff = np.array([det_cx - pred_cx, det_cy - pred_cy])
                diff_norm = np.linalg.norm(diff)
                if diff_norm < 1e-4:
                    continue
                cos_sim = np.dot(kft.velocity, diff) / (vel_norm * diff_norm)
                iou_ocm[i, j] *= (1.0 + self.inertia * cos_sim)

        np.clip(iou_ocm, 0.0, 1.0, out=iou_ocm)
        return iou_ocm

    # ------------------------------------------------------------------
    # Track creation
    # ------------------------------------------------------------------

    def _create_track(
        self, det: Detection, det_bbox: np.ndarray, frame_idx: int
    ) -> int:
        tid = self._next_id
        self._next_id += 1

        kft = _KalmanBoxTracker(det_bbox, delta_t=self.delta_t)
        self._kf_trackers[tid] = kft

        new_track = Track(
            track_id=tid,
            bbox=det.bbox,
            yolo_conf=det.confidence,
            state=TrackState.TRACKING,
            frame_created=frame_idx,
            is_new=True,
        )
        self.tracks[tid] = new_track
        return tid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou_matrix(
        bboxes_a: List[np.ndarray], bboxes_b: List[np.ndarray]
    ) -> np.ndarray:
        a = np.array(bboxes_a, dtype=np.float64).reshape(-1, 4)
        b = np.array(bboxes_b, dtype=np.float64).reshape(-1, 4)

        x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / (union + 1e-8)

    @staticmethod
    def _clip_bbox(bbox: np.ndarray) -> Tuple[int, int, int, int]:
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = max(x1 + 1, int(round(bbox[2])))
        y2 = max(y1 + 1, int(round(bbox[3])))
        return (x1, y1, x2, y2)
