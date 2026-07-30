from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from seahunter.services import PreviewHub, PreviewServer, create_preview_app


class PreviewTests(unittest.TestCase):
    def test_hub_keeps_only_latest_frame(self) -> None:
        hub = PreviewHub()
        first = hub.publish(b"first", {"frame_id": 1})
        second = hub.publish(b"second", {"frame_id": 2})

        self.assertEqual(first.sequence, 0)
        self.assertEqual(second.sequence, 1)
        self.assertEqual(hub.snapshot(), second)
        self.assertEqual(hub.wait_for_next(0, timeout=0.0), second)
        self.assertIsNone(hub.wait_for_next(1, timeout=0.0))

    def test_fastapi_health_metrics_and_websocket_preview(self) -> None:
        hub = PreviewHub()
        hub.publish(b"\xff\xd8preview\xff\xd9", {"frame_id": 7, "source_id": "drone-01"})
        app = create_preview_app(hub, metrics_provider=lambda: {"peak_process_rss_mb": 123.0})

        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["latest_sequence"], 0)
            self.assertEqual(client.get("/metrics").json()["peak_process_rss_mb"], 123.0)
            self.assertIn("SeaHunter-VIS Edge Preview", client.get("/").text)
            with client.websocket_connect("/ws/preview") as websocket:
                metadata = websocket.receive_json()
                jpeg = websocket.receive_bytes()

            self.assertEqual(metadata["type"], "frame")
            self.assertEqual(metadata["frame_id"], 7)
            self.assertEqual(jpeg, b"\xff\xd8preview\xff\xd9")

    def test_preview_server_validates_bind_address(self) -> None:
        app = create_preview_app(PreviewHub())
        with self.assertRaises(ValueError):
            PreviewServer(app, port=0)


if __name__ == "__main__":
    unittest.main()
