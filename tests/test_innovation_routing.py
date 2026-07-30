from __future__ import annotations

import unittest

from seahunter.data import WeatherCondition
from seahunter.innovation import (
    AblationMetrics,
    AblationRun,
    DegradationObservation,
    RouteProfile,
    WeatherRouter,
    WeatherRouterConfig,
    needs_sim2real,
    summarize_ablations,
)


def profiles(model_id: str = "robust-v1") -> tuple[RouteProfile, ...]:
    return tuple(RouteProfile(weather, model_id, f"preprocess-{weather.value}", 0.4) for weather in WeatherCondition)


class RoutingAndAblationInnovationTests(unittest.TestCase):
    def test_weather_route_switches_only_after_clip_stable_vote_and_dwell(self) -> None:
        router = WeatherRouter(
            WeatherRouterConfig(profiles(), voting_window=3, minimum_votes=2, minimum_dwell_frames=2)
        )
        first = router.update(DegradationObservation("clip-a", 0, WeatherCondition.FOG, 2, 0.9))
        second = router.update(DegradationObservation("clip-a", 1, WeatherCondition.FOG, 2, 0.9))
        switched = router.update(DegradationObservation("clip-a", 2, WeatherCondition.FOG, 2, 0.9))
        new_clip = router.update(DegradationObservation("clip-b", 0, WeatherCondition.FOG, 2, 0.9))

        self.assertFalse(first.switched)
        self.assertFalse(second.switched)
        self.assertTrue(switched.switched)
        self.assertEqual(switched.profile.weather, WeatherCondition.FOG)
        self.assertEqual(new_clip.profile.weather, WeatherCondition.NORMAL)

    def test_expert_models_require_explicit_cost_acceptance(self) -> None:
        expert_profiles = tuple(
            RouteProfile(weather, f"expert-{weather.value}", "none", 0.5) for weather in WeatherCondition
        )
        with self.assertRaisesRegex(ValueError, "single robust model"):
            WeatherRouterConfig(expert_profiles)
        self.assertTrue(WeatherRouterConfig(expert_profiles, allow_expert_models=True).allow_expert_models)

    def test_sim2real_is_requested_only_for_real_coverage_gap(self) -> None:
        complete = {weather: 100 for weather in WeatherCondition}
        incomplete = {WeatherCondition.NORMAL: 100}
        self.assertFalse(needs_sim2real(complete, minimum_real_samples=50))
        self.assertTrue(needs_sim2real(incomplete, minimum_real_samples=50))

    def test_ablation_requires_three_unique_seeds_and_reports_variation(self) -> None:
        runs = tuple(
            AblationRun(
                "selective-roi",
                seed,
                AblationMetrics(0.8 + seed * 0.01, 0.7, 1.0, 10.0, 20.0, 1024.0, 12.0 - seed),
                True,
            )
            for seed in (0, 1, 2)
        )
        summary = summarize_ablations(runs)[0]

        self.assertEqual(summary.seeds, (0, 1, 2))
        self.assertGreater(summary.metrics["precision"].standard_deviation, 0.0)
        self.assertTrue(summary.exportable)
        with self.assertRaisesRegex(ValueError, "requires 3"):
            summarize_ablations(runs[:2])


if __name__ == "__main__":
    unittest.main()
