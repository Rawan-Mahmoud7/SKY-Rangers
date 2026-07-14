"""
=============================================================================
Temporal Fusion — Per-Track Confidence Accumulation
=============================================================================
Maintains temporal state for each track: EMA-smoothed confidence, prediction
voting across frames, streak counting, and stability metrics.

Key design: NEVER decide from a single frame. All outputs represent
accumulated evidence over multiple frames.

Usage:
    >>> fusion = TemporalFusion(cfg)
    >>> result = fusion.update(track, classifier_result)
    >>> if result.is_stable and result.streak_high >= confirm_streak:
    ...     # Ready to confirm
=============================================================================
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .classifier import ClassifierResult

logger = logging.getLogger(__name__)


# =========================================================================
# Fusion Result
# =========================================================================

@dataclass
class FusionResult:
    """
    Temporal fusion output for a single track in a single frame.

    Attributes
    ----------
    best_country : str
        Country with highest accumulated vote weight.
    fused_confidence : float
        EMA-smoothed, stability-weighted confidence.
    vote_fraction : float
        Fraction of recent frames where best_country was the top prediction.
    streak_high : int
        Consecutive high-confidence frames for the same country.
    streak_low : int
        Consecutive low-confidence frames.
    is_stable : bool
        True when enough frames have been accumulated and variance is low.
    total_frames : int
        How many classifier frames this track has accumulated.
    top_country_this_frame : str
        The classifier's top prediction this specific frame (before fusion).
    top_confidence_this_frame : float
        The classifier's top confidence this specific frame.
    """
    best_country: str = ""
    fused_confidence: float = 0.0
    vote_fraction: float = 0.0
    streak_high: int = 0
    streak_low: int = 0
    is_stable: bool = False
    total_frames: int = 0
    top_country_this_frame: str = ""
    top_confidence_this_frame: float = 0.0


# =========================================================================
# Per-Track State (internal, stored on Track object)
# =========================================================================
# The temporal state is stored directly on the Track dataclass fields:
#   track.confidence_history   — deque of recent confidences
#   track.c_temporal           — EMA-smoothed confidence
#   track.streak_high          — consecutive high-confidence same-country frames
#   track.streak_low           — consecutive low-confidence frames
#   track.best_country         — current best guess
#   track.best_confidence      — current best confidence
#   track.vote_fraction        — vote consistency


# =========================================================================
# Temporal Fusion
# =========================================================================

class TemporalFusion:
    """
    Accumulates classifier results over time for each track.

    Does NOT make decisions — only computes evidence metrics.
    The DecisionEngine uses these metrics to decide confirmation.
    """

    def __init__(self, cfg: dict):
        tf_cfg = cfg.get("temporal_fusion", {})
        dec_cfg = cfg.get("decision", {})

        self.ema_alpha = tf_cfg.get("ema_alpha", 0.3)
        self.history_length = tf_cfg.get("history_length", 10)
        self.min_frames = tf_cfg.get("min_frames_for_decision", 3)
        self.high_conf_thresh = dec_cfg.get("high_confidence_threshold", 0.60)
        self.low_conf_thresh = dec_cfg.get("low_confidence_threshold", 0.30)

        # Per-track vote history: track_id → deque of (country, confidence)
        self._vote_history: Dict[int, deque] = {}

        logger.info(
            f"TemporalFusion: ema={self.ema_alpha}, history={self.history_length}, "
            f"min_frames={self.min_frames}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, track, classifier_result: ClassifierResult) -> FusionResult:
        """
        Update temporal state for a track with a new classifier result.

        Parameters
        ----------
        track : Track
            The track object (state is modified in-place).
        classifier_result : ClassifierResult
            Latest classifier output for this track's crop.

        Returns
        -------
        FusionResult with accumulated evidence metrics.
        """
        tid = track.track_id

        # Ensure vote history exists
        if tid not in self._vote_history:
            self._vote_history[tid] = deque(maxlen=self.history_length)

        # This frame's top prediction
        if classifier_result.top_k_labels:
            top_country = classifier_result.top_k_labels[0]
            top_conf = classifier_result.top_k_confidences[0]
        else:
            top_country = ""
            top_conf = 0.0

        # Record vote
        self._vote_history[tid].append((top_country, top_conf))

        # --- EMA confidence ---
        track.confidence_history.append(top_conf)

        if len(track.confidence_history) <= 1:
            track.c_temporal = top_conf
        else:
            track.c_temporal = (
                (1 - self.ema_alpha) * track.c_temporal
                + self.ema_alpha * top_conf
            )

        # --- Stability (1 - stddev of recent confidences) ---
        if len(track.confidence_history) >= 3:
            recent = list(track.confidence_history)[-self.history_length:]
            stability = max(0.0, 1.0 - float(np.std(recent)))
        else:
            stability = 1.0

        fused_confidence = track.c_temporal * stability

        # --- Vote counting ---
        votes = self._vote_history[tid]
        if votes:
            country_counts = Counter(country for country, _ in votes)
            best_country = country_counts.most_common(1)[0][0]
            vote_fraction = country_counts[best_country] / len(votes)
        else:
            best_country = ""
            vote_fraction = 0.0

        # --- Streak management ---
        if top_conf >= self.high_conf_thresh and top_country == best_country:
            track.streak_high += 1
            track.streak_low = 0
        elif top_conf < self.low_conf_thresh:
            track.streak_low += 1
            track.streak_high = 0
        else:
            # Mid-zone: don't reset, don't increment
            pass

        # Update track state
        track.best_country = best_country
        track.best_confidence = fused_confidence
        track.vote_fraction = vote_fraction

        total_frames = len(self._vote_history[tid])
        is_stable = total_frames >= self.min_frames and stability > 0.5

        return FusionResult(
            best_country=best_country,
            fused_confidence=fused_confidence,
            vote_fraction=vote_fraction,
            streak_high=track.streak_high,
            streak_low=track.streak_low,
            is_stable=is_stable,
            total_frames=total_frames,
            top_country_this_frame=top_country,
            top_confidence_this_frame=top_conf,
        )

    def remove_track(self, track_id: int) -> None:
        """Clean up state when a track is lost."""
        self._vote_history.pop(track_id, None)
