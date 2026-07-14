"""
=============================================================================
Vision Events — Event Dataclasses + Pub/Sub Bus
=============================================================================
Defines all events emitted by the vision pipeline to the flight controller.
The pipeline is a passive observer — it never issues navigation commands.

Event types:
    FLAG_DETECTED      — YOLO first detects a new tracked object
    COUNTRY_CONFIRMED  — Temporal fusion confirms a country identity
    TRACK_LOST         — Track expired (no detection for max_age frames)

Usage:
    >>> bus = EventBus()
    >>> bus.subscribe("FLAG_DETECTED", lambda e: print(e))
    >>> bus.emit(FlagDetectedEvent(...))
=============================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Event Dataclasses
# =========================================================================

@dataclass
class VisionEvent:
    """Base event emitted by the vision pipeline."""
    event_type: str
    track_id: int
    frame_idx: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class FlagDetectedEvent(VisionEvent):
    """
    Emitted immediately when a NEW track is created from a YOLO detection.

    The flight controller should react to this (e.g., hover, slow down)
    while the recognition pipeline works on identifying the flag.
    """
    event_type: str = field(default="FLAG_DETECTED", init=False)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    yolo_confidence: float = 0.0


@dataclass
class CountryConfirmedEvent(VisionEvent):
    """
    Emitted when temporal fusion + decision engine confirm a country.

    This only fires after multiple consistent frames exceed the
    confirmation thresholds. Once emitted, the track is locked and
    the classifier will NOT run again for this track ID.
    """
    event_type: str = field(default="COUNTRY_CONFIRMED", init=False)
    country: str = ""
    confidence: float = 0.0
    egypt_verified: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class TrackLostEvent(VisionEvent):
    """
    Emitted when a track expires (no detection for max_age frames).

    If the track was never confirmed, last_country_guess contains
    the best guess at the time of expiry (may be None).
    """
    event_type: str = field(default="TRACK_LOST", init=False)
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_country_guess: Optional[str] = None


# =========================================================================
# Event Bus — Simple Pub/Sub
# =========================================================================

class EventBus:
    """
    Lightweight pub/sub for vision events.

    The flight controller registers callbacks for event types it cares about.
    Events are dispatched synchronously (no threading) to keep latency
    predictable on the Raspberry Pi.

    Usage
    -----
    >>> bus = EventBus()
    >>> bus.subscribe("FLAG_DETECTED", my_handler)
    >>> bus.subscribe("COUNTRY_CONFIRMED", my_handler)
    >>> bus.emit(some_event)
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Register a callback for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"EventBus: subscribed {handler.__name__} to {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Remove a callback for a specific event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def emit(self, event: VisionEvent) -> None:
        """
        Dispatch an event to all registered handlers.

        Handlers are called synchronously in registration order.
        Exceptions in handlers are caught and logged (never crash the pipeline).
        """
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"EventBus: handler {handler.__name__} raised {e} "
                    f"for {event.event_type}",
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
