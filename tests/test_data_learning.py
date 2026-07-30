from __future__ import annotations

import unittest

from seahunter.data import (
    ActiveLearningCandidate,
    ActiveLearningConfig,
    PseudoLabel,
    PseudoLabelFilterConfig,
    WeatherCondition,
    filter_pseudo_labels,
    select_active_learning_samples,
)


class DataLearningTests(unittest.TestCase):
    def test_active_learning_prioritizes_false_alarm_disagreement_and_fragmentation(self) -> None:
        candidates = (
            ActiveLearningCandidate("routine", "video-a", 0.1, 0.1, 0, False),
            ActiveLearningCandidate("false-alarm", "video-b", 0.8, 0.9, 4, True),
            ActiveLearningCandidate("fragmented", "video-c", 0.5, 0.4, 3, False),
        )
        selected = select_active_learning_samples(candidates, 2)

        self.assertEqual([item.sample_id for item in selected], ["false-alarm", "fragmented"])
        self.assertEqual(
            selected[0].reasons,
            ("low_confidence", "model_disagreement", "track_fragmentation", "false_alarm"),
        )

    def test_active_learning_enforces_per_video_cap(self) -> None:
        candidates = tuple(
            ActiveLearningCandidate(f"sample-{index}", "same-video", 1.0, 1.0, index, False) for index in range(3)
        )
        selected = select_active_learning_samples(candidates, 3, ActiveLearningConfig(maximum_per_video=1))
        self.assertEqual(len(selected), 1)

    def test_pseudo_labels_require_weather_threshold_agreement_time_and_tracklet(self) -> None:
        labels = tuple(
            PseudoLabel(
                sample_id=f"good-{index}",
                video_id="video-a",
                frame_index=index,
                track_id=4,
                class_name="boat",
                student_class_name="boat",
                confidence=0.85,
                temporal_iou=0.7,
                weather=WeatherCondition.HIGH_WAVE,
            )
            for index in range(3)
        ) + (
            PseudoLabel(
                sample_id="bad-disagreement",
                video_id="video-a",
                frame_index=4,
                track_id=4,
                class_name="boat",
                student_class_name="buoy",
                confidence=0.9,
                temporal_iou=0.8,
                weather=WeatherCondition.HIGH_WAVE,
            ),
        )
        decisions = filter_pseudo_labels(
            labels,
            PseudoLabelFilterConfig(
                weather_confidence=((WeatherCondition.HIGH_WAVE, 0.82),),
                minimum_track_length=3,
            ),
        )

        self.assertTrue(all(item.accepted for item in decisions[:3]))
        self.assertEqual(decisions[3].rejection_reasons, ("teacher_student_disagreement",))

    def test_short_tracklet_and_weather_low_confidence_are_audited(self) -> None:
        labels = (
            PseudoLabel("one", "video", 0, 1, "boat", "boat", 0.7, 0.8, WeatherCondition.FOG),
            PseudoLabel("two", "video", 1, 1, "boat", "boat", 0.9, 0.8, WeatherCondition.FOG),
        )
        decisions = filter_pseudo_labels(
            labels,
            PseudoLabelFilterConfig(weather_confidence=((WeatherCondition.FOG, 0.8),), minimum_track_length=3),
        )
        self.assertEqual(decisions[0].rejection_reasons, ("low_confidence",))
        self.assertEqual(decisions[1].rejection_reasons, ("short_tracklet",))


if __name__ == "__main__":
    unittest.main()
