from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from seahunter.schemas import Detection, FramePacket
from seahunter.services import FrameResult, TrackingSink
from seahunter.tracking import ByteTrackConfig, ByteTracker, MOTChallengeWriter, TrackJsonlWriter


def result(frame_id: int, detections: tuple[Detection, ...]) -> FrameResult:
    return FrameResult(
        frame=FramePacket(
            source_id="flight-01",
            frame_id=frame_id,
            captured_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
            width=640,
            height=360,
        ),
        detections=detections,
        detector_id="detector:v1",
        dropped_before=0,
        inference_duration_ms=1.0,
    )


def swimmer(x: float) -> Detection:
    return Detection(
        bbox_xyxy=(x, 40.0, x + 10.0, 55.0),
        class_id=0,
        class_name="swimmer",
        confidence=0.9,
        detector_id="detector:v1",
    )


class TrackingOutputTests(unittest.TestCase):
    def test_tracking_sink_writes_mot_and_auditable_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mot = MOTChallengeWriter(root / "tracks.txt")
            audit = TrackJsonlWriter(root / "tracks.jsonl")
            tracking = TrackingSink(
                ByteTracker(ByteTrackConfig(emit_lost_predictions=True)),
                sinks=[mot, audit],
            )

            tracking.write(result(0, (swimmer(10.0),)))
            tracking.write(result(1, ()))
            tracking.write(result(2, (swimmer(11.0),)))
            tracking.close()

            mot_lines = (root / "tracks.txt").read_text(encoding="utf-8").splitlines()
            audit_rows = [json.loads(line) for line in (root / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(mot_lines), 2)
            self.assertTrue(mot_lines[0].startswith("1,1,"))
            self.assertTrue(all(row["schema_version"] == 5 for row in audit_rows))
            self.assertEqual([row["observation"] for row in audit_rows], ["observed", "inferred", "observed"])
            self.assertEqual([row["association_stage"] for row in audit_rows], ["new", None, "high"])
            self.assertEqual(audit_rows[1]["lost_reason"], "unmatched")
            self.assertEqual(audit_rows[1]["inference_method"], "extrapolated")
            self.assertEqual(tracking.frames_processed, 3)
            self.assertEqual(tracking.states_emitted, 3)

    def test_mot_writer_can_include_inferred_rows_explicitly(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.txt"
            writer = MOTChallengeWriter(path, include_inferred=True)
            tracking = TrackingSink(
                ByteTracker(ByteTrackConfig(emit_lost_predictions=True)),
                sinks=[writer],
            )
            tracking.write(result(0, (swimmer(10.0),)))
            tracking.write(result(1, ()))
            tracking.close()
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
