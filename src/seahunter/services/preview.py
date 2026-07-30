"""Latest-frame JPEG hub and FastAPI WebSocket preview application."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic, sleep
from typing import Any


@dataclass(frozen=True, slots=True)
class PreviewFrame:
    """One encoded preview frame with canonical metadata."""

    sequence: int
    jpeg: bytes
    metadata: dict[str, object]


class PreviewHub:
    """Thread-safe single-slot preview store that never accumulates frames."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._latest: PreviewFrame | None = None
        self._next_sequence = 0
        self._closed = False

    def publish(self, jpeg: bytes, metadata: Mapping[str, object]) -> PreviewFrame:
        if not jpeg:
            raise ValueError("jpeg must not be empty")
        with self._condition:
            if self._closed:
                raise RuntimeError("preview hub is closed")
            frame = PreviewFrame(
                sequence=self._next_sequence,
                jpeg=bytes(jpeg),
                metadata=dict(metadata),
            )
            self._next_sequence += 1
            self._latest = frame
            self._condition.notify_all()
            return frame

    def wait_for_next(self, after_sequence: int, timeout: float | None = None) -> PreviewFrame | None:
        """Wait for a newer preview or return None on timeout/closed exhaustion."""

        if after_sequence < -1:
            raise ValueError("after_sequence must be at least -1")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while (self._latest is None or self._latest.sequence <= after_sequence) and not self._closed:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._latest is None or self._latest.sequence <= after_sequence:
                return None
            return self._latest

    def snapshot(self) -> PreviewFrame | None:
        with self._condition:
            return self._latest

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def create_preview_app(
    hub: PreviewHub,
    *,
    metrics_provider: Callable[[], Mapping[str, object]] | None = None,
) -> Any:
    """Create a FastAPI app with health, metrics, HTML, and WebSocket routes."""

    try:
        fastapi: Any = importlib.import_module("fastapi")
        responses: Any = importlib.import_module("fastapi.responses")
    except ModuleNotFoundError as exc:
        raise RuntimeError("WebSocket preview requires the 'edge' project extra") from exc
    app = fastapi.FastAPI(title="SeaHunter-VIS edge preview", version="0.1.0")

    async def index() -> str:
        return _PREVIEW_HTML

    async def health() -> dict[str, object]:
        latest = hub.snapshot()
        return {
            "status": "stopped" if hub.closed else "running",
            "latest_sequence": None if latest is None else latest.sequence,
        }

    async def metrics() -> dict[str, object]:
        return {} if metrics_provider is None else dict(metrics_provider())

    async def preview(websocket: Any) -> None:
        await websocket.accept()
        after_sequence = -1
        idle_polls = 0
        try:
            while True:
                frame = await asyncio.to_thread(hub.wait_for_next, after_sequence, 0.25)
                if frame is None:
                    if hub.closed:
                        await websocket.close(code=1000)
                        return
                    idle_polls += 1
                    if idle_polls >= 40:
                        await websocket.send_json({"type": "heartbeat"})
                        idle_polls = 0
                    continue
                idle_polls = 0
                after_sequence = frame.sequence
                metadata = dict(frame.metadata)
                metadata["type"] = "frame"
                metadata["preview_sequence"] = frame.sequence
                await websocket.send_json(metadata)
                await websocket.send_bytes(frame.jpeg)
        except fastapi.WebSocketDisconnect:
            return

    preview.__annotations__["websocket"] = fastapi.WebSocket
    app.add_api_route("/", index, methods=["GET"], response_class=responses.HTMLResponse)
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_websocket_route("/ws/preview", preview)
    return app


class PreviewServer:
    """Run a Uvicorn preview application in a managed background thread."""

    def __init__(
        self,
        app: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_level: str = "warning",
    ) -> None:
        if not host.strip():
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be within [1, 65535]")
        self._app = app
        self.host = host
        self.port = port
        self._log_level = log_level
        self._server: Any | None = None
        self._thread: Thread | None = None

    def start(self, timeout: float = 5.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if self._thread is not None:
            raise RuntimeError("preview server has already been started")
        try:
            uvicorn: Any = importlib.import_module("uvicorn")
        except ModuleNotFoundError as exc:
            raise RuntimeError("preview serving requires the 'edge' project extra") from exc
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level=self._log_level,
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, name="seahunter-preview", daemon=True)
        self._thread.start()

        deadline = monotonic() + timeout
        while not bool(self._server.started):
            if not self._thread.is_alive():
                raise RuntimeError("preview server stopped during startup")
            if monotonic() >= deadline:
                self.stop()
                raise TimeoutError("timed out starting preview server")
            sleep(0.01)

    def stop(self, timeout: float = 5.0) -> bool:
        server = self._server
        thread = self._thread
        if server is None or thread is None:
            return True
        server.should_exit = True
        thread.join(timeout)
        return not thread.is_alive()


_PREVIEW_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>SeaHunter-VIS Preview</title>
<style>body{margin:0;background:#07121c;color:#d9edf7;font-family:system-ui}main{max-width:1200px;margin:auto;padding:16px}
img{width:100%;background:#000;border:1px solid #29485d}pre{white-space:pre-wrap;color:#9fc5d8}</style></head>
<body><main><h1>SeaHunter-VIS Edge Preview</h1><img id="preview" alt="Waiting for video"><pre id="meta"></pre></main>
<script>
const image=document.getElementById('preview'); const meta=document.getElementById('meta');
const scheme=location.protocol==='https:'?'wss':'ws'; const socket=new WebSocket(`${scheme}://${location.host}/ws/preview`);
socket.binaryType='blob'; let previous=null;
socket.onmessage=(event)=>{if(typeof event.data==='string'){
meta.textContent=JSON.stringify(JSON.parse(event.data),null,2);return;}
if(previous)URL.revokeObjectURL(previous); previous=URL.createObjectURL(event.data); image.src=previous;};
</script></body></html>"""
