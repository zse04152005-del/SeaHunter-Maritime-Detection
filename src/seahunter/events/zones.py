"""Validated local/geodetic danger-zone definitions and metric geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, hypot, isfinite, radians

from .geofence import Point, Polygon, point_in_polygon

_EARTH_RADIUS_M = 6_378_137.0


class ZoneCoordinateSpace(str, Enum):
    LOCAL_ENU = "local_enu"
    GEODETIC = "geodetic"


@dataclass(frozen=True, slots=True)
class DangerZone:
    zone_id: str
    name: str
    polygon: Polygon
    coordinate_space: ZoneCoordinateSpace
    severity: float
    approach_distance_m: float = 50.0
    collision_ttc_seconds: float = 30.0
    expected_direction_en: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.zone_id.strip() or not self.name.strip():
            raise ValueError("zone_id and name must not be empty")
        validate_simple_polygon(self.polygon)
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("zone severity must be within [0, 1]")
        if self.approach_distance_m < 0.0 or self.collision_ttc_seconds <= 0.0:
            raise ValueError("zone approach distance and collision TTC must be valid")
        if self.coordinate_space is ZoneCoordinateSpace.GEODETIC:
            for longitude, latitude in self.polygon:
                if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
                    raise ValueError("geodetic zone points must be (longitude, latitude)")
        if self.expected_direction_en is not None:
            east, north = self.expected_direction_en
            if not isfinite(east) or not isfinite(north) or hypot(east, north) <= 1e-9:
                raise ValueError("expected_direction_en must be a finite non-zero vector")

    def local_polygon(self) -> Polygon:
        if self.coordinate_space is ZoneCoordinateSpace.LOCAL_ENU:
            return self.polygon
        origin_longitude = sum(point[0] for point in self.polygon) / len(self.polygon)
        origin_latitude = sum(point[1] for point in self.polygon) / len(self.polygon)
        return tuple(_geodetic_to_local(point, origin_longitude, origin_latitude) for point in self.polygon)

    def point_to_local(self, point: Point) -> Point:
        if self.coordinate_space is ZoneCoordinateSpace.LOCAL_ENU:
            return point
        longitude, latitude = point
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise ValueError("geodetic observation must be (longitude, latitude)")
        origin_longitude = sum(vertex[0] for vertex in self.polygon) / len(self.polygon)
        origin_latitude = sum(vertex[1] for vertex in self.polygon) / len(self.polygon)
        return _geodetic_to_local(point, origin_longitude, origin_latitude)


def validate_simple_polygon(polygon: Polygon) -> None:
    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three vertices")
    if not all(isfinite(value) for point in polygon for value in point):
        raise ValueError("polygon coordinates must be finite")
    if len(set(polygon)) != len(polygon):
        raise ValueError("polygon vertices must be unique")
    area_twice = sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )
    if abs(area_twice) <= 1e-9:
        raise ValueError("polygon area must be non-zero")
    for first in range(len(polygon)):
        first_start = polygon[first]
        first_end = polygon[(first + 1) % len(polygon)]
        for second in range(first + 1, len(polygon)):
            if second in {first, (first + 1) % len(polygon)} or first == (second + 1) % len(polygon):
                continue
            second_start = polygon[second]
            second_end = polygon[(second + 1) % len(polygon)]
            if _segments_intersect(first_start, first_end, second_start, second_end):
                raise ValueError("polygon must not self-intersect")


def signed_distance_to_polygon(point: Point, polygon: Polygon) -> float:
    """Return positive metric distance inside and negative distance outside."""

    validate_simple_polygon(polygon)
    distance = min(_point_segment_distance(point, polygon[index - 1], polygon[index]) for index in range(len(polygon)))
    return distance if point_in_polygon(point, polygon) else -distance


def _geodetic_to_local(point: Point, origin_longitude: float, origin_latitude: float) -> Point:
    longitude, latitude = point
    east = radians(longitude - origin_longitude) * _EARTH_RADIUS_M * cos(radians(origin_latitude))
    north = radians(latitude - origin_latitude) * _EARTH_RADIUS_M
    return (east, north)


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-18:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    return hypot(point[0] - (start[0] + projection * dx), point[1] - (start[1] + projection * dy))


def _segments_intersect(first_start: Point, first_end: Point, second_start: Point, second_end: Point) -> bool:
    def orientation(start: Point, end: Point, point: Point) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])

    first_a = orientation(first_start, first_end, second_start)
    first_b = orientation(first_start, first_end, second_end)
    second_a = orientation(second_start, second_end, first_start)
    second_b = orientation(second_start, second_end, first_end)
    return first_a * first_b < 0.0 and second_a * second_b < 0.0
