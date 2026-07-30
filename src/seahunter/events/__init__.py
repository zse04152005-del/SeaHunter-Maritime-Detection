"""Risk-event and geofence primitives."""

from .engine import DangerZoneEngine, DangerZoneEngineConfig, ZoneObservation, ZoneRule
from .geofence import Point, Polygon, point_in_polygon
from .zones import DangerZone, ZoneCoordinateSpace, signed_distance_to_polygon, validate_simple_polygon

__all__ = [
    "DangerZone",
    "DangerZoneEngine",
    "DangerZoneEngineConfig",
    "Point",
    "Polygon",
    "ZoneCoordinateSpace",
    "ZoneObservation",
    "ZoneRule",
    "point_in_polygon",
    "signed_distance_to_polygon",
    "validate_simple_polygon",
]
