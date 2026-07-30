from __future__ import annotations

import unittest

from seahunter.innovation import (
    DownsampleVariant,
    JointConfidenceEvidence,
    JointConfidenceUpdater,
    TemporalFeatureFrame,
    TemporalSignalClass,
    TemporalSmallTargetMonitor,
    joint_spatial_frequency_downsample,
    operator_manifest,
)


def frame(
    index: int, spatial: tuple[float, ...], *, scene: str = "scene", wave: float = 0.0, camera: float = 0.0
) -> TemporalFeatureFrame:
    return TemporalFeatureFrame(
        source_id="camera",
        scene_id=scene,
        frame_index=index,
        spatial_features=spatial,
        motion_residual=spatial,
        wave_periodicity=(wave,) * len(spatial),
        camera_motion=camera,
    )


class TemporalAndConfidenceInnovationTests(unittest.TestCase):
    def test_temporal_monitor_fuses_change_and_resets_on_scene_change(self) -> None:
        monitor = TemporalSmallTargetMonitor()
        first = monitor.update(frame(0, (0.0, 0.0)))
        changed = monitor.update(frame(1, (1.0, 0.0)))
        reset = monitor.update(frame(2, (1.0, 0.0), scene="new-scene"))

        self.assertEqual(first.reset_reason, "new_source")
        self.assertEqual(changed.signal_classes[0], TemporalSignalClass.TARGET_CANDIDATE)
        self.assertEqual(reset.reset_reason, "scene_change")
        self.assertEqual(reset.history_size, 1)

    def test_camera_and_periodic_wave_are_not_target_candidates(self) -> None:
        monitor = TemporalSmallTargetMonitor()
        camera = monitor.update(frame(0, (1.0,), camera=0.9))
        monitor.reset()
        wave = monitor.update(frame(1, (1.0,), wave=0.9))

        self.assertEqual(camera.signal_classes, (TemporalSignalClass.CAMERA_MOTION,))
        self.assertEqual(wave.signal_classes, (TemporalSignalClass.PERIODIC_WAVE,))
        self.assertLess(wave.fused_scores[0], 0.2)

    def test_joint_confidence_has_bounded_persistence_and_wave_veto(self) -> None:
        updater = JointConfidenceUpdater()
        saturated = updater.update(JointConfidenceEvidence(0.7, 0.8, 0.8, 0.7, 0.1, 0.0, 100))
        at_cap = updater.update(JointConfidenceEvidence(0.7, 0.8, 0.8, 0.7, 0.1, 0.0, 5))
        wave = updater.update(JointConfidenceEvidence(0.9, 0.9, 0.9, 0.9, 0.0, 0.95, 100))

        self.assertEqual(saturated.persistence_boost, at_cap.persistence_boost)
        self.assertTrue(wave.vetoed_by_wave)
        self.assertLessEqual(wave.confidence, 0.25)

    def test_joint_antialias_uses_exportable_real_primitives(self) -> None:
        image = tuple(tuple(float(row * 4 + column) for column in range(4)) for row in range(4))
        output = joint_spatial_frequency_downsample(image)
        manifest = operator_manifest(DownsampleVariant.JOINT_ANTIALIAS)

        self.assertEqual((len(output), len(output[0])), (2, 2))
        self.assertFalse(manifest.uses_fft)
        self.assertFalse(manifest.uses_complex)
        self.assertFalse(manifest.tensorrt_plugin_required)
        self.assertEqual(manifest.onnx_primitives, ("Conv", "Abs", "Mul", "Add"))


if __name__ == "__main__":
    unittest.main()
