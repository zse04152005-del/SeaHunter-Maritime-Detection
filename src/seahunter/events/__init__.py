"""Risk-event and geofence primitives."""

from .api import create_event_app
from .engine import DangerZoneEngine, DangerZoneEngineConfig, ZoneObservation, ZoneRule
from .evidence import EvidenceFrame, EvidenceRecorder, EvidenceRecorderConfig
from .feedback import HardSampleFeedbackWriter
from .geofence import Point, Polygon, point_in_polygon
from .persistence import FeedbackLabel, OperatorFeedback, SQLiteEventStore
from .publication import EventHub, MQTTClient, MQTTEventPublisher, PublishedEvent
from .service import EventService, PublicationFailure
from .zones import DangerZone, ZoneCoordinateSpace, signed_distance_to_polygon, validate_simple_polygon

__all__ = [
    "DangerZone",
    "DangerZoneEngine",
    "DangerZoneEngineConfig",
    "EventHub",
    "EventService",
    "EvidenceFrame",
    "EvidenceRecorder",
    "EvidenceRecorderConfig",
    "FeedbackLabel",
    "HardSampleFeedbackWriter",
    "MQTTClient",
    "MQTTEventPublisher",
    "OperatorFeedback",
    "Point",
    "Polygon",
    "PublicationFailure",
    "PublishedEvent",
    "SQLiteEventStore",
    "ZoneCoordinateSpace",
    "ZoneObservation",
    "ZoneRule",
    "create_event_app",
    "point_in_polygon",
    "signed_distance_to_polygon",
    "validate_simple_polygon",
]
