from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from seahunter.events import (
    DangerZone,
    DangerZoneEngine,
    DangerZoneEngineConfig,
    ZoneCoordinateSpace,
    ZoneObservation,
    ZoneRule,
    signed_distance_to_polygon,
    validate_simple_polygon,
)
from seahunter.schemas import EventStatus, MotionTrend

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def observation(
    seconds: float,
    point: tuple[float, float],
    *,
    velocity: tuple[float, float] = (0.0, 0.0),
    reliability: float = 0.9,
    ttc_seconds: float | None = None,
    trend: MotionTrend = MotionTrend.UNKNOWN,
) -> ZoneObservation:
    return ZoneObservation(
        track_id=7,
        captured_at=START + timedelta(seconds=seconds),
        point=point,
        velocity_en_m_s=velocity,
        reliability=reliability,
        class_id=1,
        ttc_seconds=ttc_seconds,
        trend=trend,
    )


def zone() -> DangerZone:
    return DangerZone(
        zone_id="restricted-1",
        name="Restricted water",
        polygon=((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)),
        coordinate_space=ZoneCoordinateSpace.LOCAL_ENU,
        severity=0.9,
        approach_distance_m=30.0,
        collision_ttc_seconds=15.0,
        expected_direction_en=(1.0, 0.0),
    )


class EventEngineTests(unittest.TestCase):
    def test_polygon_validation_distance_and_geodetic_projection(self) -> None:
        with self.assertRaises(ValueError):
            validate_simple_polygon(((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)))
        self.assertAlmostEqual(signed_distance_to_polygon((5.0, 5.0), zone().local_polygon()), 5.0)
        self.assertAlmostEqual(signed_distance_to_polygon((-2.0, 5.0), zone().local_polygon()), -2.0)
        geodetic = DangerZone(
            zone_id="geo",
            name="Geodetic",
            polygon=((120.0, 30.0), (120.001, 30.0), (120.001, 30.001), (120.0, 30.001)),
            coordinate_space=ZoneCoordinateSpace.GEODETIC,
            severity=0.5,
        )
        self.assertGreater(
            signed_distance_to_polygon(geodetic.point_to_local((120.0005, 30.0005)), geodetic.local_polygon()), 0.0
        )

    def test_single_frame_jitter_does_not_open_event(self) -> None:
        engine = DangerZoneEngine((zone(),))
        self.assertEqual(engine.update(observation(0.0, (10.0, 10.0))), [])
        self.assertEqual(engine.update(observation(0.5, (-5.0, 10.0))), [])
        self.assertEqual(engine.active_events(), ())

    def test_enter_acknowledge_dwell_exit_and_cooldown_lifecycle(self) -> None:
        engine = DangerZoneEngine(
            (zone(),),
            DangerZoneEngineConfig(
                minimum_enter_seconds=1.0,
                minimum_exit_seconds=1.0,
                minimum_dwell_seconds=2.0,
                cooldown_seconds=10.0,
            ),
        )
        engine.update(observation(0.0, (10.0, 10.0)))
        entered = engine.update(observation(1.0, (10.0, 10.0)))
        enter = next(event for event in entered if event.rule_id == ZoneRule.ENTER.value)
        self.assertEqual(enter.status, EventStatus.OPEN)

        acknowledged = engine.acknowledge(enter.event_id, START + timedelta(seconds=1.5))
        self.assertEqual(acknowledged.status, EventStatus.ACKNOWLEDGED)
        dwell_update = engine.update(observation(3.0, (10.0, 10.0)))
        self.assertEqual(
            next(event for event in dwell_update if event.rule_id == ZoneRule.ENTER.value).status,
            EventStatus.ACKNOWLEDGED,
        )
        self.assertEqual(
            next(event for event in dwell_update if event.rule_id == ZoneRule.DWELL.value).status,
            EventStatus.OPEN,
        )

        engine.update(observation(4.0, (-5.0, 10.0)))
        exited = engine.update(observation(5.0, (-5.0, 10.0)))
        self.assertEqual(
            next(event for event in exited if event.rule_id == ZoneRule.ENTER.value).status,
            EventStatus.CLOSED,
        )
        self.assertEqual(
            next(event for event in exited if event.rule_id == ZoneRule.EXIT.value).status,
            EventStatus.OPEN,
        )

        engine.update(observation(6.0, (10.0, 10.0)))
        reopened = engine.update(observation(7.0, (10.0, 10.0)))
        self.assertFalse(any(event.rule_id == ZoneRule.ENTER.value for event in reopened))

    def test_approach_reverse_collision_and_risk_score(self) -> None:
        engine = DangerZoneEngine(
            (zone(),),
            DangerZoneEngineConfig(
                minimum_enter_seconds=0.0,
                class_risk=((1, 1.0),),
            ),
        )
        engine.update(observation(0.0, (-20.0, 10.0), velocity=(5.0, 0.0)))
        approaching = engine.update(observation(1.0, (-10.0, 10.0), velocity=(5.0, 0.0)))
        approach = next(event for event in approaching if event.rule_id == ZoneRule.APPROACH.value)
        self.assertGreater(approach.risk_score, 0.7)

        inside = engine.update(
            observation(
                2.0,
                (10.0, 10.0),
                velocity=(-2.0, 0.0),
                ttc_seconds=5.0,
                trend=MotionTrend.APPROACHING,
            )
        )
        rules = {event.rule_id for event in inside if event.status is not EventStatus.CLOSED}
        self.assertIn(ZoneRule.REVERSE.value, rules)
        self.assertIn(ZoneRule.COLLISION.value, rules)

    def test_low_reliability_observation_does_not_close_active_event(self) -> None:
        engine = DangerZoneEngine((zone(),), DangerZoneEngineConfig(minimum_enter_seconds=0.0))
        opened = engine.update(observation(0.0, (10.0, 10.0)))
        self.assertTrue(any(event.rule_id == ZoneRule.ENTER.value for event in opened))
        self.assertEqual(engine.update(observation(1.0, (-10.0, 10.0), reliability=0.1)), [])
        self.assertTrue(any(event.rule_id == ZoneRule.ENTER.value for event in engine.active_events()))


if __name__ == "__main__":
    unittest.main()
