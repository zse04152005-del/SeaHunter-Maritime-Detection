from __future__ import annotations

import unittest

from seahunter.events import point_in_polygon


class GeofenceTests(unittest.TestCase):
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    def test_inside(self) -> None:
        self.assertTrue(point_in_polygon((5.0, 5.0), self.polygon))

    def test_outside(self) -> None:
        self.assertFalse(point_in_polygon((11.0, 5.0), self.polygon))

    def test_boundary_policy(self) -> None:
        self.assertTrue(point_in_polygon((0.0, 5.0), self.polygon))
        self.assertFalse(point_in_polygon((0.0, 5.0), self.polygon, include_boundary=False))

    def test_invalid_polygon(self) -> None:
        with self.assertRaises(ValueError):
            point_in_polygon((0.0, 0.0), ((0.0, 0.0), (1.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
