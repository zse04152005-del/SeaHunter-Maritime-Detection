from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from seahunter.events import EvidenceFrame, EvidenceRecorder, EvidenceRecorderConfig
from seahunter.schemas import (
    EventStatus,
    ModelManifest,
    ObservationKind,
    RiskEvent,
    TelemetryPacket,
    TrackState,
)

START = datetime(2026, 7, 30, tzinfo=timezone.utc)


def make_event() -> RiskEvent:
    return RiskEvent(
        event_id="event-1",
        rule_id="enter",
        zone_id="restricted",
        track_id=8,
        opened_at=START + timedelta(seconds=1.0),
        updated_at=START + timedelta(seconds=1.0),
        status=EventStatus.OPEN,
        risk_score=0.9,
        evidence_uri="file:///evidence/event-1.evidence.zip",
    )


def make_frame(seconds: float) -> EvidenceFrame:
    captured_at = START + timedelta(seconds=seconds)
    return EvidenceFrame(
        captured_at=captured_at,
        jpeg=b"\xff\xd8" + str(seconds).encode() + b"\xff\xd9",
        tracks=(
            TrackState(
                track_id=8,
                frame_id=int(seconds * 10),
                captured_at=captured_at,
                bbox_xyxy=(1.0, 2.0, 5.0, 6.0),
                class_id=1,
                confidence=0.9,
                observation=ObservationKind.OBSERVED,
            ),
        ),
        telemetry=TelemetryPacket(
            source_id="drone-1",
            captured_at=captured_at,
            latitude_deg=30.0,
            longitude_deg=120.0,
            altitude_m=100.0,
            platform_roll_deg=0.0,
            platform_pitch_deg=0.0,
            platform_yaw_deg=0.0,
        ),
        model_manifests=(
            ModelManifest(
                model_id="detector",
                version="1.2.3",
                framework="onnxruntime",
                input_size=(640, 640),
                class_names=("boat",),
                artifact_sha256="a" * 64,
                git_commit="deadbeef",
                data_version="dataset-v1",
            ),
        ),
    )


class EvidenceRecorderTests(unittest.TestCase):
    def test_bundle_contains_pre_post_context_and_verified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvidenceRecorder(
                Path(directory),
                EvidenceRecorderConfig(pre_event_seconds=1.0, post_event_seconds=1.0, maximum_ring_frames=8),
            )
            recorder.ingest(make_frame(0.0))
            event = make_event()
            recorder.open_event(event)
            recorder.ingest(make_frame(1.0))
            recorder.update_event(replace(event, status=EventStatus.CLOSED, updated_at=START + timedelta(seconds=1.5)))

            completed = recorder.ingest(make_frame(2.0))

            self.assertEqual(len(completed), 1)
            with zipfile.ZipFile(completed[0]) as archive:
                names = set(archive.namelist())
                self.assertTrue({"manifest.json", "tracks.jsonl", "telemetry.jsonl", "models.json"} <= names)
                self.assertEqual(len([name for name in names if name.startswith("frames/")]), 3)
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["event"]["status"], "closed")
                self.assertEqual(manifest["frame_count"], 3)
                self.assertEqual(manifest["model_manifests"]["detector:1.2.3"]["git_commit"], "deadbeef")
                for name, expected in manifest["artifact_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)

    def test_close_flushes_pending_event_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvidenceRecorder(Path(directory))
            recorder.open_event(make_event())
            completed = recorder.close()

            self.assertEqual(len(completed), 1)
            self.assertTrue(completed[0].exists())
            self.assertFalse(completed[0].with_suffix(".zip.tmp").exists())
            self.assertEqual(recorder.close(), ())


if __name__ == "__main__":
    unittest.main()
