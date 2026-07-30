"""Deterministic two-dimensional geofence geometry."""

from __future__ import annotations

Point = tuple[float, float]
Polygon = tuple[Point, ...]


def _point_on_segment(point: Point, start: Point, end: Point, epsilon: float) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= epsilon


def point_in_polygon(point: Point, polygon: Polygon, *, include_boundary: bool = True, epsilon: float = 1e-9) -> bool:
    """Return whether a point is inside a simple polygon using ray casting."""

    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three vertices")

    inside = False
    px, py = point
    previous = polygon[-1]

    for current in polygon:
        if _point_on_segment(point, previous, current, epsilon):
            return include_boundary

        x1, y1 = previous
        x2, y2 = current
        crosses_y = (y1 > py) != (y2 > py)
        if crosses_y:
            x_intersection = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersection:
                inside = not inside
        previous = current

    return inside
