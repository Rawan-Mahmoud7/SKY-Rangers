"""
=============================================================================
Decision Engine — Event-Emitting, Egypt-Aware
=============================================================================
Evaluates temporal fusion results and Egypt verification to decide when to
confirm a country for a track. Emits CountryConfirmedEvent when all criteria
are met.

Core guarantees:
    1. NEVER confirm from a single frame — requires confirm_streak consecutive
       high-confidence frames.
    2. Once confirmed, the track is LOCKED — classifier never runs again.
    3. Egypt gets priority verification via embedding similarity.

Decision flow per track per frame:
    1. Already confirmed? → skip
    2. Egypt in top-3? → run embedding verification
       a. Similarity passes + temporal stability → confirm Egypt
       b. Fails → keep classifier result
    3. Non-Egypt: check standard confirmation criteria
       a. fused_confidence >= threshold
       b. streak_high >= confirm_streak
       c. vote_fraction >= vote_threshold
       → Confirm that country
    4. Otherwise → continue accumulating (no event)
=============================================================================
"""

import logging
import time
from typing import Optional

from .classifier import ClassifierResult
from .egypt_verifier import EgyptVerifier
from .events import CountryConfirmedEvent, VisionEvent
from .temporal_fusion import FusionResult
from .tracker import Track, TrackState

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Makes confirmation decisions based on temporal fusion and Egypt verification.

    Only emits CountryConfirmedEvent — never issues navigation commands.
    """

    def __init__(self, cfg: dict):
        dec_cfg = cfg.get("decision", {})
        ev_cfg = cfg.get("egypt_verification", {})

        self.confirm_threshold = dec_cfg.get("confirm_threshold", 0.70)
        self.confirm_streak = dec_cfg.get("confirm_streak", 5)
        self.high_conf_thresh = dec_cfg.get("high_confidence_threshold", 0.60)
        self.vote_threshold = dec_cfg.get("non_egypt_confirm_vote_fraction", 0.7)

        self.egypt_enabled = ev_cfg.get("enabled", True)
        self.egypt_class_name = ev_cfg.get("egypt_class_name", "Egypt")

        logger.info(
            f"DecisionEngine: confirm_thresh={self.confirm_threshold}, "
            f"confirm_streak={self.confirm_streak}, "
            f"vote_thresh={self.vote_threshold}, "
            f"egypt_enabled={self.egypt_enabled}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        track: Track,
        classifier_result: ClassifierResult,
        fusion_result: FusionResult,
        egypt_verifier: EgyptVerifier,
        frame_idx: int,
    ) -> Optional[VisionEvent]:
        """
        Evaluate a track and decide whether to confirm.

        Parameters
        ----------
        track : Track
            The active track.
        classifier_result : ClassifierResult
            Latest classifier output (top-K + embedding).
        fusion_result : FusionResult
            Temporal fusion metrics.
        egypt_verifier : EgyptVerifier
            Egypt embedding verifier instance.
        frame_idx : int
            Current frame index.

        Returns
        -------
        Optional[CountryConfirmedEvent]
            Event if confirmation criteria are met, None otherwise.
        """
        # Already confirmed — should not reach here, but guard anyway
        if track.confirmed_country is not None:
            return None

        # --- Egypt verification path ---
        if self.egypt_enabled and self._egypt_in_top_k(
            classifier_result, k=3
        ):
            egypt_result = egypt_verifier.verify(classifier_result.embedding)

            if egypt_result.is_egypt and fusion_result.is_stable:
                # Egypt embedding verification passed + temporal stability
                # Check if we have minimum frames
                if fusion_result.total_frames >= self.confirm_streak:
                    return self._confirm_track(
                        track, self.egypt_class_name,
                        egypt_result.similarity_score,
                        egypt_verified=True,
                        frame_idx=frame_idx,
                    )

                # Not enough frames yet — continue accumulating
                logger.debug(
                    f"Track#{track.track_id}: Egypt verified (sim={egypt_result.similarity_score:.3f}) "
                    f"but need more frames ({fusion_result.total_frames}/{self.confirm_streak})"
                )
                return None

            # Egypt similarity failed → fall through to standard path
            if not egypt_result.is_egypt:
                logger.debug(
                    f"Track#{track.track_id}: Egypt in top-3 but similarity "
                    f"failed ({egypt_result.similarity_score:.3f} < {egypt_result.threshold})"
                )

        # --- Standard confirmation path (any country including Egypt without embedding) ---
        if (
            fusion_result.is_stable
            and fusion_result.fused_confidence >= self.confirm_threshold
            and fusion_result.streak_high >= self.confirm_streak
            and fusion_result.vote_fraction >= self.vote_threshold
        ):
            return self._confirm_track(
                track, fusion_result.best_country,
                fusion_result.fused_confidence,
                egypt_verified=False,
                frame_idx=frame_idx,
            )

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _egypt_in_top_k(
        self, result: ClassifierResult, k: int = 3
    ) -> bool:
        """Check if Egypt is in the top-K predictions."""
        top_labels = result.top_k_labels[:k]
        return self.egypt_class_name in top_labels

    def _confirm_track(
        self,
        track: Track,
        country: str,
        confidence: float,
        egypt_verified: bool,
        frame_idx: int,
    ) -> CountryConfirmedEvent:
        """Lock the track and create a confirmation event."""
        track.confirmed_country = country
        track.confirmed_confidence = confidence
        track.egypt_verified = egypt_verified
        track.state = TrackState.CONFIRMED

        logger.info(
            f"[F{frame_idx}] Track#{track.track_id} CONFIRMED: "
            f"{country} (conf={confidence:.3f}, egypt_verified={egypt_verified})"
        )

        return CountryConfirmedEvent(
            track_id=track.track_id,
            frame_idx=frame_idx,
            timestamp=time.time(),
            country=country,
            confidence=confidence,
            egypt_verified=egypt_verified,
            bbox=track.bbox,
        )
