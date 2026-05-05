# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import quote, urlparse
import zlib

import aiohttp
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from PIL import Image
from pydantic import BaseModel
import socketio  # type: ignore[import-untyped]
from sse_starlette.sse import EventSourceResponse
import uvicorn

from dimos.models.vl.openai import OpenAIVlModel
from dimos.msgs.sensor_msgs.Image import ImageFormat
from dimos.msgs.sensor_msgs.Image import Image as DimosImage
from dimos.robot.unitree.unitree_skill_container import UNITREE_WEBRTC_CONTROLS
from dimos.slamass.chat_agent import ChatMessage, SlamassChatAgent
from dimos.slamass.map_memory import ActiveMapState, RawCostmap
from dimos.slamass.storage import (
    ActiveMapRecord,
    PoiObservationRecord,
    PoiRecord,
    SlamassStorage,
    YoloObjectRecord,
    YoloObservationRecord,
    utc_now_iso,
)
from dimos.utils.llm_utils import extract_json
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

UI_MIN_ZOOM = 0.25
UI_MAX_ZOOM = 4.0
# Overview after focus-map; keep in sync with DEFAULT_MAP_ZOOM in slamass-app mapViewport.ts
UI_DEFAULT_MAP_ZOOM = 0.78
# Active map fusion grid (m / cell). Finer than before for a sharper UI map; incoming
# costmap cells are merged by world position in ActiveMapState.update_from_costmap.
ACTIVE_MAP_GRID_RESOLUTION = 0.075
UI_FOCUS_POI_ZOOM = 1.0
UI_FOCUS_ROBOT_ZOOM = 1.0
POI_ARRIVAL_TOLERANCE_METERS = 0.6
POI_ARRIVAL_SETTLE_SECONDS = 1.0
POI_ARRIVAL_TIMEOUT_SECONDS = 120.0
POI_ROTATION_THRESHOLD_DEGREES = 8.0
DEFAULT_POV_POLL_INTERVAL_SECONDS = 0.25
MIN_POV_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_POV_MAX_WIDTH = 1024
DEFAULT_POV_JPEG_QUALITY = 76
MAP_SOCKET_POV_STALE_AFTER_SECONDS = 0.75
INSPECTION_MODE_AI_GATE = "ai_gate"
INSPECTION_MODE_ALWAYS_CREATE = "always_create"
SEMANTIC_KIND_POI = "vlm_poi"
SEMANTIC_KIND_YOLO = "yolo_object"
YOLO_MODE_LIVE = "live"
YOLO_MODE_PAUSED = "paused"
VALID_YOLO_MODES = (YOLO_MODE_LIVE, YOLO_MODE_PAUSED)
# SLAMASS consumes world-projected 3D detections, which are naturally sparser and
# noisier than the raw per-frame 2D detector output. For a live demo we want
# objects to appear after a quick confirmation, not after the robot stares at the
# same thing for several seconds.
YOLO_PROMOTION_WINDOW_SECONDS = 12.0
YOLO_PROMOTION_MIN_HITS = 2
YOLO_DEDUPE_DISTANCE_METERS = 0.75
YOLO_DEFAULT_LAYER_VISIBLE = True
VALID_MANUAL_INSPECTION_MODES = (
    INSPECTION_MODE_AI_GATE,
    INSPECTION_MODE_ALWAYS_CREATE,
)
YOLO_CLASS_WHITELIST = {
    "chair",
    "couch",
    "potted plant",
    "dining table",
    "tv",
    "laptop",
    "book",
    "clock",
    "bottle",
    "cup",
    "backpack",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "bed",
    "toilet",
    "bench",
    "vase",
}


def load_slamass_env() -> None:
    """Load dotenv files for the standalone SLAMASS service."""
    dotenv_candidates: list[Path] = []

    cwd_dotenv = find_dotenv(filename=".env", usecwd=True)
    if cwd_dotenv:
        dotenv_candidates.append(Path(cwd_dotenv))

    repo_dotenv = Path(__file__).resolve().parents[2] / ".env"
    if repo_dotenv.exists():
        dotenv_candidates.append(repo_dotenv)

    seen: set[Path] = set()
    for dotenv_path in dotenv_candidates:
        resolved = dotenv_path.resolve()
        if resolved in seen:
            continue
        load_dotenv(dotenv_path=resolved, override=False)
        seen.add(resolved)


load_slamass_env()


def slamass_openai_configured() -> bool:
    """True when a non-empty ``OPENAI_API_KEY`` is set (env or dotenv loaded at import)."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def default_state_dir() -> Path:
    base = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "dimos" / "slamass"


@dataclass(slots=True)
class RobotPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(slots=True)
class InspectionAnalysis:
    title: str
    summary: str
    category: str
    interest_score: float
    should_create_poi: bool
    gate_reason: str
    objects: list[str]

    def as_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "interest_score": self.interest_score,
            "should_create_poi": self.should_create_poi,
            "gate_reason": self.gate_reason,
            "objects": self.objects,
        }


@dataclass(slots=True)
class InspectionSettings:
    manual_mode: str = INSPECTION_MODE_AI_GATE


@dataclass(slots=True)
class YoloRuntimeState:
    mode: str = YOLO_MODE_LIVE
    inference_enabled: bool = True


@dataclass(slots=True)
class LayerVisibilityState:
    show_pois: bool = True
    show_yolo: bool = YOLO_DEFAULT_LAYER_VISIBLE


@dataclass(slots=True)
class UiCameraState:
    center_x: float | None = None
    center_y: float | None = None
    zoom: float = UI_DEFAULT_MAP_ZOOM


@dataclass(slots=True)
class SemanticItemRef:
    kind: str
    entity_id: str


@dataclass(slots=True)
class SlamassUiState:
    camera: UiCameraState = field(default_factory=UiCameraState)
    selected_item: SemanticItemRef | None = None
    highlighted_items: list[SemanticItemRef] = field(default_factory=list)
    revision: int = 0


@dataclass(slots=True)
class ChatState:
    messages: list[ChatMessage] = field(default_factory=list)
    running: bool = False


@dataclass(slots=True)
class YoloDetection:
    label: str
    class_id: int
    confidence: float
    world_x: float
    world_y: float
    world_z: float
    size_x: float
    size_y: float
    size_z: float
    view_x: float
    view_y: float
    view_yaw: float
    crop_jpeg: bytes


@dataclass(slots=True)
class PendingYoloHit:
    ts: float
    detection: YoloDetection


@dataclass(slots=True)
class PendingYoloCandidate:
    label: str
    class_id: int
    hits: deque[PendingYoloHit] = field(default_factory=deque)

    def add_hit(self, ts: float, detection: YoloDetection) -> None:
        self.hits.append(PendingYoloHit(ts=ts, detection=detection))
        while self.hits and ts - self.hits[0].ts > YOLO_PROMOTION_WINDOW_SECONDS:
            self.hits.popleft()

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    def representative(self) -> YoloDetection:
        best_hit = max(self.hits, key=lambda hit: hit.detection.confidence)
        return best_hit.detection

    def averaged_detection(self) -> YoloDetection:
        representative = self.representative()
        count = float(len(self.hits))
        world_x = sum(hit.detection.world_x for hit in self.hits) / count
        world_y = sum(hit.detection.world_y for hit in self.hits) / count
        world_z = sum(hit.detection.world_z for hit in self.hits) / count
        size_x = sum(hit.detection.size_x for hit in self.hits) / count
        size_y = sum(hit.detection.size_y for hit in self.hits) / count
        size_z = sum(hit.detection.size_z for hit in self.hits) / count
        return YoloDetection(
            label=representative.label,
            class_id=representative.class_id,
            confidence=representative.confidence,
            world_x=world_x,
            world_y=world_y,
            world_z=world_z,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            view_x=representative.view_x,
            view_y=representative.view_y,
            view_yaw=representative.view_yaw,
            crop_jpeg=representative.crop_jpeg,
        )


class NavigateRequest(BaseModel):
    x: float
    y: float
    yaw: float | None = None


class UiCameraRequest(BaseModel):
    center_x: float
    center_y: float
    zoom: float


class UiSelectionRequest(BaseModel):
    kind: str | None = None
    entity_id: str | None = None


class UiHighlightRequest(BaseModel):
    items: list[dict[str, str]]
    selected_item: dict[str, str] | None = None


class UiFocusRequest(BaseModel):
    zoom: float | None = None


class InspectionSettingsRequest(BaseModel):
    manual_mode: str


class TeleopCommandRequest(BaseModel):
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0


class YoloRuntimeRequest(BaseModel):
    mode: str | None = None
    inference_enabled: bool | None = None


class LayerVisibilityRequest(BaseModel):
    show_pois: bool | None = None
    show_yolo: bool | None = None


class ChatSubmitRequest(BaseModel):
    message: str


class SearchlightRequest(BaseModel):
    enabled: bool


class ObstacleAvoidanceRequest(BaseModel):
    enabled: bool


class SportCommandRequest(BaseModel):
    command: str


class McpToolClient:
    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url
        self._session: aiohttp.ClientSession | None = None
        self._mcp_initialized = False

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=20)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def stop(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def observe_jpeg(self) -> bytes | None:
        body = await self.call_tool("observe")
        content = body.get("result", {}).get("content", [])
        if not isinstance(content, list):
            return None

        flat: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        flat.append(sub)
            elif isinstance(item, dict):
                flat.append(item)

        for part in flat:
            if part.get("type") != "image_url":
                continue
            url_holder = part.get("image_url")
            image_url = url_holder.get("url", "") if isinstance(url_holder, dict) else ""
            if not isinstance(image_url, str) or not image_url.startswith("data:image/"):
                continue
            _, encoded = image_url.split(",", 1)
            try:
                return base64.b64decode(encoded)
            except Exception:
                return None
        return None

    async def relative_move(
        self, *, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0
    ) -> dict[str, Any]:
        return await self.call_tool(
            "relative_move",
            {"forward": forward, "left": left, "degrees": degrees},
        )

    async def set_yolo_inference(self, *, enabled: bool) -> str:
        text = await self.call_tool_text("set_yolo_inference", {"enabled": enabled})
        if text.startswith("Tool not found:") or text.startswith("Error running tool"):
            raise RuntimeError(text)
        return text

    async def call_tool_text(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> str:
        body = await self.call_tool(name, arguments, request_id=request_id)
        content = body.get("result", {}).get("content", [])
        if not isinstance(content, list):
            return ""
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    return text
        return ""

    async def _ensure_mcp_initialized(self) -> None:
        if self._mcp_initialized:
            return
        await self.start()
        assert self._session is not None
        async with self._session.post(
            self.mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": "slamass-mcp-init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "dimos-slamass", "version": "1.0.0"},
                },
            },
        ) as response:
            response.raise_for_status()
            body = await response.json()
        err = body.get("error")
        if err is not None:
            raise RuntimeError(f"MCP initialize failed: {err}")
        self._mcp_initialized = True

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        await self._ensure_mcp_initialized()
        assert self._session is not None
        async with self._session.post(
            self.mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": request_id or name,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            },
        ) as response:
            response.raise_for_status()
            body = await response.json()
        error = body.get("error")
        if error is not None:
            raise RuntimeError(f"MCP tool '{name}' failed: {error}")
        return body


def run_dimos_stop_command(force: bool = False) -> dict[str, Any]:
    """Stop the active DimOS run using the local CLI."""
    dimos_binary = shutil.which("dimos")
    if dimos_binary is None:
        raise RuntimeError("Could not find 'dimos' on PATH")

    command = [dimos_binary, "stop"]
    if force:
        command.append("--force")

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"dimos stop failed with exit code {completed.returncode}"
        raise RuntimeError(detail)

    return {
        "ok": True,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


class OpenAIInspectionAnalyzer:
    PROMPT = """Analyze this robot camera frame for a live robotics demo.

Return exactly one JSON object with these keys:
- title: short human-friendly POI title
- summary: 1-2 sentence description of what is notable in the frame
- category: short category like window, desk, lounge, poster, kitchen, entrance, hallway, display, plant
- interest_score: number from 0.0 to 1.0
- should_create_poi: boolean
- gate_reason: short reason for accepting or rejecting the frame
- objects: array of short object/location strings

Acceptance rules:
- Accept frames whenever they contain usable semantic information a human could later recognize on the map.
- Accept identifiable places or objects even if they are ordinary: desk areas, doors, windows, kitchen corners, seating, posters, shelves, appliances, hallway intersections, entryways, displays, plants, signage, workstations.
- Reject only when the frame provides almost no usable semantic information:
  - plain featureless wall, floor, or ceiling
  - severe motion blur
  - an extremely close-up crop with no scene context
  - a generic empty corridor or corner with nothing distinctive to point at
- Do not reject just because the scene is common or not visually dramatic.
- When the frame has weak but still usable context, accept it with a lower interest_score instead of rejecting it.

The title should name the place or salient thing, not a sentence.
The summary should describe visible evidence only.
"""

    def __init__(self, model_name: str = "gpt-5.4-mini") -> None:
        self._vlm = OpenAIVlModel(model_name=model_name)

    def analyze(self, image: DimosImage) -> InspectionAnalysis:
        response_text = self._vlm.query(
            image,
            self.PROMPT,
            response_format={"type": "json_object"},
        )
        parsed = extract_json(response_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from VLM, got: {type(parsed)}")

        objects = parsed.get("objects") or []
        if not isinstance(objects, list):
            objects = []

        interest_score = float(parsed.get("interest_score", 0.0))
        interest_score = max(0.0, min(1.0, interest_score))
        title = str(parsed.get("title", "")).strip() or "Untitled POI"
        summary = str(parsed.get("summary", "")).strip() or "No summary provided."
        category = str(parsed.get("category", "")).strip() or "unknown"
        should_create_poi = bool(parsed.get("should_create_poi", False))
        gate_reason = str(parsed.get("gate_reason", "")).strip() or "No gate reason provided."

        return InspectionAnalysis(
            title=title[:80],
            summary=summary[:500],
            category=category[:40],
            interest_score=interest_score,
            should_create_poi=should_create_poi,
            gate_reason=gate_reason[:200],
            objects=[str(item)[:80] for item in objects],
        )


def decode_raw_costmap_message(payload: dict[str, Any]) -> RawCostmap:
    encoded = payload.get("data")
    shape = payload.get("shape")
    if not isinstance(encoded, str) or not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Invalid raw costmap payload")

    compressed = base64.b64decode(encoded)
    data = zlib.decompress(compressed)
    grid = np_frombuffer_int8(data, int(shape[0]), int(shape[1]))

    origin_data = payload.get("origin", {}).get("c", [0.0, 0.0, 0.0])
    return RawCostmap(
        grid=grid,
        origin_x=float(origin_data[0]),
        origin_y=float(origin_data[1]),
        resolution=float(payload.get("resolution", 0.1)),
        ts=float(payload.get("ts", time.time())),
    )


def decode_pov_frame_message(payload: dict[str, Any]) -> bytes | None:
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        return None
    raw = encoded.strip()
    if raw.startswith("data:"):
        if "," not in raw:
            return None
        _, raw = raw.split(",", 1)
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


def np_frombuffer_int8(data: bytes, rows: int, cols: int) -> Any:
    import numpy as np

    return np.frombuffer(data, dtype=np.int8).reshape((rows, cols))


def jpeg_bytes_to_dimos_image(image_bytes: bytes) -> DimosImage:
    with Image.open(io_bytes(image_bytes)) as pil_image:
        rgb = pil_image.convert("RGB")
        return DimosImage.from_numpy(np_array(rgb), format=ImageFormat.RGB)


def np_frombuffer_u8(data: bytes) -> Any:
    import numpy as np

    return np.frombuffer(data, dtype=np.uint8)


def make_thumbnail(image_bytes: bytes, width: int = 320) -> bytes:
    with Image.open(io_bytes(image_bytes)) as pil_image:
        rgb = pil_image.convert("RGB")
        target_height = max(1, int(round(rgb.height * (width / rgb.width))))
        resized = rgb.resize((width, target_height), Image.Resampling.LANCZOS)
        output = io_bytes()
        resized.save(output, format="JPEG", quality=82)
        return output.getvalue()


def prepare_pov_jpeg(
    image_bytes: bytes,
    *,
    max_width: int | None,
    quality: int,
) -> bytes:
    with Image.open(io_bytes(image_bytes)) as pil_image:
        rgb = pil_image.convert("RGB")
        if max_width is not None and max_width > 0 and rgb.width > max_width:
            target_height = max(1, int(round(rgb.height * (max_width / rgb.width))))
            rgb = rgb.resize((max_width, target_height), Image.Resampling.LANCZOS)
        output = io_bytes()
        rgb.save(output, format="JPEG", quality=max(1, min(100, quality)))
        return output.getvalue()


def make_placeholder_jpeg(width: int = 96, height: int = 72) -> bytes:
    image = Image.new("RGB", (width, height), color=(22, 24, 30))
    output = io_bytes()
    image.save(output, format="JPEG", quality=82)
    return output.getvalue()


def io_bytes(data: bytes | None = None) -> Any:
    import io

    return io.BytesIO(data or b"")


def np_array(pil_image: Image.Image) -> Any:
    import numpy as np

    return np.asarray(pil_image)


def normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch.isspace()).strip()


def yaw_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def normalize_manual_inspection_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_MANUAL_INSPECTION_MODES:
        raise ValueError(f"Unsupported inspection mode: {value}")
    return normalized


def normalize_yolo_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in VALID_YOLO_MODES:
        raise ValueError(f"Unsupported YOLO mode: {value}")
    return normalized


# Rerun web UI + gRPC proxy (keep in sync with dimos/visualization/rerun/bridge.py).
_RERUN_GRPC_PORT = 9876
_RERUN_WEB_PORT = 9090


def build_dimos_rerun_web_viewer_url(map_socket_url: str) -> str:
    """Rerun 3D web viewer URL (same right-pane target as web/templates/rerun_dashboard.html)."""
    parsed = urlparse(map_socket_url)
    host = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"
    inner = f"rerun+http://{host}:{_RERUN_GRPC_PORT}/proxy"
    return f"{scheme}://{host}:{_RERUN_WEB_PORT}/?url={quote(inner, safe='')}"


class MapSocketClient:
    def __init__(
        self,
        url: str,
        *,
        on_connection: Any,
        on_pose: Any,
        on_path: Any,
        on_raw_costmap: Any,
        on_pov_frame: Any,
        on_yolo_detections: Any,
    ) -> None:
        self.url = url
        self._on_connection = on_connection
        self._on_pose = on_pose
        self._on_path = on_path
        self._on_raw_costmap = on_raw_costmap
        self._on_pov_frame = on_pov_frame
        self._on_yolo_detections = on_yolo_detections
        self._client = socketio.AsyncClient(reconnection=True)
        self._stopped = False
        self._task: asyncio.Task[None] | None = None
        self._register_handlers()

    @property
    def connected(self) -> bool:
        return self._client.connected

    async def start(self) -> None:
        self._task = asyncio.create_task(self._connect_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._client.connected:
            await self._client.disconnect()
        if self._task is not None:
            self._task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._task

    async def emit_click(self, x: float, y: float, yaw: float | None = None) -> None:
        if not self._client.connected:
            raise RuntimeError("Map socket is not connected")
        payload: list[float] | dict[str, float]
        if yaw is None:
            # Preserve legacy click payload shape so plain click-to-go works
            # even if the running websocket_vis module has not been restarted
            # since yaw-aware click handling was added.
            payload = [x, y]
        else:
            payload = {"x": x, "y": y, "yaw": yaw}
        await self._client.emit("click", payload)

    async def emit_move_command(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        linear_z: float = 0.0,
        angular_x: float = 0.0,
        angular_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> None:
        if not self._client.connected:
            raise RuntimeError("Map socket is not connected")
        await self._client.emit(
            "move_command",
            {
                "linear": {
                    "x": linear_x,
                    "y": linear_y,
                    "z": linear_z,
                },
                "angular": {
                    "x": angular_x,
                    "y": angular_y,
                    "z": angular_z,
                },
            },
        )

    def _register_handlers(self) -> None:
        @self._client.event
        async def connect() -> None:
            await self._on_connection(True)

        @self._client.event
        async def disconnect() -> None:
            await self._on_connection(False)

        @self._client.on("robot_pose")
        async def on_robot_pose(data: dict[str, Any]) -> None:
            coords = data.get("c", [])
            if len(coords) < 2:
                return
            pose = RobotPose(
                x=float(coords[0]),
                y=float(coords[1]),
                z=float(coords[2]) if len(coords) > 2 else 0.0,
                yaw=float(coords[3]) if len(coords) > 3 else 0.0,
            )
            await self._on_pose(pose)

        @self._client.on("path")
        async def on_path(data: dict[str, Any]) -> None:
            points = data.get("points", [])
            parsed = [[float(point[0]), float(point[1])] for point in points if len(point) >= 2]
            await self._on_path(parsed)

        @self._client.on("raw_costmap")
        async def on_raw_costmap(data: dict[str, Any]) -> None:
            await self._on_raw_costmap(decode_raw_costmap_message(data))

        @self._client.on("pov_frame")
        async def on_pov_frame(data: dict[str, Any]) -> None:
            frame_bytes = decode_pov_frame_message(data)
            if frame_bytes is None:
                return
            await self._on_pov_frame(frame_bytes)

        @self._client.on("yolo_detections")
        async def on_yolo_detections(data: dict[str, Any]) -> None:
            await self._on_yolo_detections(data)

        @self._client.on("full_state")
        async def on_full_state(data: dict[str, Any]) -> None:
            if "robot_pose" in data:
                await on_robot_pose(data["robot_pose"])
            if "path" in data:
                await on_path(data["path"])
            if "raw_costmap" in data:
                await on_raw_costmap(data["raw_costmap"])
            if "pov_frame" in data:
                await on_pov_frame(data["pov_frame"])
            if "yolo_detections" in data:
                await on_yolo_detections(data["yolo_detections"])

    async def _connect_loop(self) -> None:
        while not self._stopped:
            if self._client.connected:
                await asyncio.sleep(1.0)
                continue
            try:
                await self._client.connect(self.url, transports=["websocket", "polling"])
            except Exception as exc:
                logger.warning(
                    "SLAMASS service could not connect to websocket map source yet",
                    error=str(exc),
                    url=self.url,
                )
                await self._on_connection(False)
                await asyncio.sleep(2.0)
            else:
                while self._client.connected and not self._stopped:
                    await asyncio.sleep(1.0)


class SlamassService:
    def __init__(
        self,
        *,
        map_socket_url: str,
        mcp_url: str,
        state_dir: Path,
        model_name: str = "gpt-5.4-mini",
        chat_model_name: str = "gpt-5.4",
        storage: SlamassStorage | None = None,
        mcp_client: McpToolClient | None = None,
        analyzer: OpenAIInspectionAnalyzer | None = None,
        chat_agent: SlamassChatAgent | None = None,
        stop_command_runner: Any | None = None,
        poi_arrival_tolerance_m: float = POI_ARRIVAL_TOLERANCE_METERS,
        poi_arrival_settle_seconds: float = POI_ARRIVAL_SETTLE_SECONDS,
        poi_arrival_timeout_seconds: float = POI_ARRIVAL_TIMEOUT_SECONDS,
        poi_rotation_threshold_degrees: float = POI_ROTATION_THRESHOLD_DEGREES,
        pov_poll_interval_seconds: float = DEFAULT_POV_POLL_INTERVAL_SECONDS,
        pov_max_width: int | None = DEFAULT_POV_MAX_WIDTH,
        pov_jpeg_quality: int = DEFAULT_POV_JPEG_QUALITY,
    ) -> None:
        self.state_dir = state_dir
        self.storage = storage or SlamassStorage(state_dir)
        self.mcp_client = mcp_client or McpToolClient(mcp_url)
        self._openai_configured = slamass_openai_configured()
        if analyzer is not None:
            self.analyzer = analyzer
        elif self._openai_configured:
            self.analyzer = OpenAIInspectionAnalyzer(model_name=model_name)
        else:
            self.analyzer = None
        self.chat_agent = chat_agent or SlamassChatAgent(model_name=chat_model_name)
        self._chat_vlm = (
            OpenAIVlModel(model_name=chat_model_name) if self._openai_configured else None
        )
        self._stop_command_runner = stop_command_runner or run_dimos_stop_command
        self._dimos_viewer_url = map_socket_url.rstrip("/")
        self._dimos_rerun_web_viewer_url = build_dimos_rerun_web_viewer_url(map_socket_url)
        self.map_client = MapSocketClient(
            map_socket_url,
            on_connection=self._handle_connection_change,
            on_pose=self._handle_pose,
            on_path=self._handle_path,
            on_raw_costmap=self._handle_raw_costmap,
            on_pov_frame=self._handle_live_pov_frame,
            on_yolo_detections=self._handle_yolo_detections,
        )
        self._tasks: list[asyncio.Task[None]] = []
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._state_lock = asyncio.Lock()
        self._dirty_map = False
        self._stopped = False
        self._last_pose_event = 0.0
        self._goto_poi_task: asyncio.Task[None] | None = None
        self._chat_task: asyncio.Task[None] | None = None
        self._poi_arrival_tolerance_m = poi_arrival_tolerance_m
        self._poi_arrival_settle_seconds = poi_arrival_settle_seconds
        self._poi_arrival_timeout_seconds = poi_arrival_timeout_seconds
        self._poi_rotation_threshold_degrees = poi_rotation_threshold_degrees
        self._pov_poll_interval_seconds = max(
            MIN_POV_POLL_INTERVAL_SECONDS,
            pov_poll_interval_seconds,
        )
        self._pov_max_width = max(1, pov_max_width) if pov_max_width is not None and pov_max_width > 0 else None
        self._pov_jpeg_quality = max(1, min(100, pov_jpeg_quality))
        self._last_socket_pov_monotonic = 0.0
        self._pov_live: bool = False

        self.robot_pose: RobotPose | None = None
        self.path: list[list[float]] = []
        self.map_state: ActiveMapState | None = None
        self.map_record: ActiveMapRecord | None = None
        self.pois: dict[str, PoiRecord] = {}
        self.yolo_objects: dict[str, YoloObjectRecord] = {}
        self.pending_yolo_candidates: list[PendingYoloCandidate] = []
        self.latest_pov_jpeg: bytes | None = None
        self.pov_seq: int = 0
        self.pov_updated_at: str | None = None
        self.connected: bool = False
        # Battery state-of-charge (0–100) reported by the Go2; None until the
        # robot publishes a recognised battery field on a subscribed topic.
        self.battery_percent: int | None = None
        self.inspection_state: dict[str, Any] = {
            "status": "idle",
            "message": "",
            "poi_id": None,
        }
        self.inspection_settings = InspectionSettings()
        self.yolo_runtime = YoloRuntimeState()
        self.layer_visibility = LayerVisibilityState()
        self.ui_state = SlamassUiState()
        self.chat_state = ChatState()

    async def start(self) -> None:
        self._load_from_storage()
        await self._maybe_start_mcp_client()
        await self._set_yolo_inference_enabled(
            self.yolo_runtime.inference_enabled,
            best_effort=True,
        )
        await self.map_client.start()
        self._tasks = [
            asyncio.create_task(self._pov_loop()),
            asyncio.create_task(self._checkpoint_loop()),
            asyncio.create_task(self._battery_loop()),
        ]

    async def stop(self) -> None:
        self._stopped = True
        if self._goto_poi_task is not None:
            self._goto_poi_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._goto_poi_task
            self._goto_poi_task = None
        if self._chat_task is not None:
            self._chat_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._chat_task
            self._chat_task = None
        await self.map_client.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib_suppress(asyncio.CancelledError):
                await task
        with contextlib_suppress(asyncio.CancelledError):
            await self.flush_active_map(force=True)
        with contextlib_suppress(asyncio.CancelledError):
            await self._maybe_stop_mcp_client()
        if self._chat_vlm is not None:
            self._chat_vlm.stop()
        self.storage.close()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def publish_event(self, event: str, data: Any) -> None:
        dead_queues: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in list(self._subscribers):
            try:
                queue.put_nowait({"event": event, "data": data})
            except asyncio.QueueFull:
                dead_queues.append(queue)
        for queue in dead_queues:
            self._subscribers.discard(queue)

    async def snapshot(self) -> dict[str, Any]:
        async with self._state_lock:
            return {
                "connected": self.connected,
                "battery_percent": self.battery_percent,
                "robot_pose": self._serialize_pose(self.robot_pose),
                "path": self.path,
                "pov": {
                    "available": self._pov_live,
                    "seq": self.pov_seq,
                    "updated_at": self.pov_updated_at,
                    "image_url": f"/api/pov/latest.jpg?v={self.pov_seq}",
                },
                "map": self._serialize_map(),
                "pois": [self._serialize_poi(poi) for poi in self._active_pois()],
                "yolo_objects": [
                    self._serialize_yolo_object(object_record)
                    for object_record in self._active_yolo_objects()
                ],
                "inspection": dict(self.inspection_state),
                "inspection_settings": self._serialize_inspection_settings_locked(),
                "yolo_runtime": self._serialize_yolo_runtime_locked(),
                "layers": self._serialize_layers_locked(),
                "ui": self._serialize_ui_locked(),
                "chat": self._serialize_chat_locked(),
                "openai_configured": self._openai_configured,
                "dimos_viewer_url": self._dimos_viewer_url,
                "dimos_rerun_web_viewer_url": self._dimos_rerun_web_viewer_url,
            }

    async def navigate(self, x: float, y: float, yaw: float | None = None) -> None:
        await self.map_client.emit_click(x, y, yaw)

    async def send_move_command(
        self,
        *,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        linear_z: float = 0.0,
        angular_x: float = 0.0,
        angular_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> dict[str, Any]:
        try:
            await self.map_client.emit_move_command(
                linear_x=linear_x,
                linear_y=linear_y,
                linear_z=linear_z,
                angular_x=angular_x,
                angular_y=angular_y,
                angular_z=angular_z,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    async def stop_motion(self) -> dict[str, Any]:
        return await self.send_move_command()

    async def stop_dimos(self, *, force: bool = False) -> dict[str, Any]:
        with contextlib_suppress(HTTPException):
            await self.stop_motion()
        try:
            result = await asyncio.to_thread(self._stop_command_runner, force)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result

    async def go_to_poi(self, poi_id: str) -> None:
        async with self._state_lock:
            poi = self.pois.get(poi_id)
            if poi is None or poi.status == "deleted":
                raise HTTPException(status_code=404, detail="POI not found")
        await self._go_to_view_pose(
            entity_id=poi.poi_id,
            target_x=poi.anchor_x,
            target_y=poi.anchor_y,
            target_yaw=poi.anchor_yaw,
        )

    async def go_to_yolo_object(self, object_id: str) -> None:
        async with self._state_lock:
            object_record = self.yolo_objects.get(object_id)
            if object_record is None or object_record.status == "deleted":
                raise HTTPException(status_code=404, detail="YOLO object not found")
        await self._go_to_view_pose(
            entity_id=object_record.object_id,
            target_x=object_record.best_view_x,
            target_y=object_record.best_view_y,
            target_yaw=object_record.best_view_yaw,
        )

    async def _go_to_view_pose(
        self,
        *,
        entity_id: str,
        target_x: float,
        target_y: float,
        target_yaw: float,
    ) -> None:
        await self.navigate(target_x, target_y)
        if self._goto_poi_task is not None:
            self._goto_poi_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._goto_poi_task
        self._goto_poi_task = asyncio.create_task(
            self._finish_poi_navigation(
                poi_id=entity_id,
                target_x=target_x,
                target_y=target_y,
                target_yaw=target_yaw,
            )
        )

    async def set_manual_inspection_mode(self, manual_mode: str) -> dict[str, Any]:
        try:
            normalized_mode = normalize_manual_inspection_mode(manual_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with self._state_lock:
            self.inspection_settings.manual_mode = normalized_mode
            payload = self._serialize_inspection_settings_locked()
            self.storage.save_json_setting("inspection_settings", payload)
        await self.publish_event("state_updated", {"inspection_settings": payload})
        return payload

    async def set_yolo_mode(self, mode: str) -> dict[str, Any]:
        return await self.set_yolo_runtime(mode=mode)

    async def set_yolo_runtime(
        self,
        *,
        mode: str | None = None,
        inference_enabled: bool | None = None,
    ) -> dict[str, Any]:
        normalized_mode: str | None = None
        if mode is not None:
            try:
                normalized_mode = normalize_yolo_mode(mode)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if inference_enabled is not None:
            try:
                await self._set_yolo_inference_enabled(bool(inference_enabled))
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        async with self._state_lock:
            if normalized_mode is not None:
                self.yolo_runtime.mode = normalized_mode
            if inference_enabled is not None:
                self.yolo_runtime.inference_enabled = bool(inference_enabled)
            payload = self._serialize_yolo_runtime_locked()
            self.storage.save_json_setting("yolo_runtime", payload)
        await self.publish_event("state_updated", {"yolo_runtime": payload})
        return payload

    async def set_layer_visibility(
        self,
        *,
        show_pois: bool | None = None,
        show_yolo: bool | None = None,
    ) -> dict[str, Any]:
        async with self._state_lock:
            if show_pois is not None:
                self.layer_visibility.show_pois = bool(show_pois)
            if show_yolo is not None:
                self.layer_visibility.show_yolo = bool(show_yolo)
            payload = self._serialize_layers_locked()
            self.storage.save_json_setting("layer_visibility", payload)
        await self.publish_event("state_updated", {"layers": payload})
        return payload

    async def ui_snapshot(self) -> dict[str, Any]:
        async with self._state_lock:
            return self._serialize_ui_locked()

    async def set_ui_camera(self, center_x: float, center_y: float, zoom: float) -> dict[str, Any]:
        async with self._state_lock:
            self._apply_ui_camera_locked(center_x=center_x, center_y=center_y, zoom=zoom)
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def select_item(self, kind: str | None, entity_id: str | None) -> dict[str, Any]:
        async with self._state_lock:
            self.ui_state.selected_item = self._validate_optional_item_locked(kind, entity_id)
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def highlight_items(
        self,
        items: list[SemanticItemRef],
        *,
        selected_item: SemanticItemRef | None = None,
    ) -> dict[str, Any]:
        async with self._state_lock:
            self.ui_state.highlighted_items = self._normalize_item_refs_locked(items)
            self.ui_state.selected_item = self._validate_optional_item_locked(
                selected_item.kind if selected_item is not None else None,
                selected_item.entity_id if selected_item is not None else None,
            )
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def clear_ui_focus(self) -> dict[str, Any]:
        async with self._state_lock:
            self.ui_state.selected_item = None
            self.ui_state.highlighted_items = []
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def focus_item(self, kind: str, entity_id: str, zoom: float | None = None) -> dict[str, Any]:
        async with self._state_lock:
            item_ref = self._require_item_ref_locked(kind, entity_id)
            world_x, world_y = self._item_world_xy_locked(item_ref)
            self.ui_state.selected_item = item_ref
            self.ui_state.highlighted_items = [item_ref]
            self._apply_ui_camera_locked(
                center_x=world_x,
                center_y=world_y,
                zoom=zoom if zoom is not None else UI_FOCUS_POI_ZOOM,
            )
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def focus_poi(self, poi_id: str, zoom: float | None = None) -> dict[str, Any]:
        return await self.focus_item(SEMANTIC_KIND_POI, poi_id, zoom)

    async def focus_yolo_object(self, object_id: str, zoom: float | None = None) -> dict[str, Any]:
        return await self.focus_item(SEMANTIC_KIND_YOLO, object_id, zoom)

    async def select_poi(self, poi_id: str | None) -> dict[str, Any]:
        return await self.select_item(SEMANTIC_KIND_POI if poi_id is not None else None, poi_id)

    async def highlight_pois(
        self, poi_ids: list[str], *, selected_poi_id: str | None = None
    ) -> dict[str, Any]:
        return await self.highlight_items(
            [SemanticItemRef(kind=SEMANTIC_KIND_POI, entity_id=poi_id) for poi_id in poi_ids],
            selected_item=(
                SemanticItemRef(kind=SEMANTIC_KIND_POI, entity_id=selected_poi_id)
                if selected_poi_id is not None
                else None
            ),
        )

    async def focus_robot(self, zoom: float | None = None) -> dict[str, Any]:
        async with self._state_lock:
            if self.robot_pose is None:
                raise HTTPException(status_code=409, detail="No robot pose available")
            self._apply_ui_camera_locked(
                center_x=self.robot_pose.x,
                center_y=self.robot_pose.y,
                zoom=zoom if zoom is not None else UI_FOCUS_ROBOT_ZOOM,
            )
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def focus_map(self) -> dict[str, Any]:
        async with self._state_lock:
            if self.map_state is None:
                raise HTTPException(status_code=404, detail="No active map available")
            center_x, center_y = self._map_center_locked()
            self._apply_ui_camera_locked(center_x=center_x, center_y=center_y, zoom=UI_DEFAULT_MAP_ZOOM)
            payload = self._commit_ui_state_locked()
        await self.publish_event("ui_state_updated", payload)
        return payload

    async def clear_low_level_map_memory(self) -> dict[str, Any]:
        async with self._state_lock:
            self.storage.clear_active_map()
            self.map_state = None
            self.path = []
            self._dirty_map = False
            self.ui_state.camera = UiCameraState(center_x=None, center_y=None, zoom=UI_DEFAULT_MAP_ZOOM)
            ui_payload = self._commit_ui_state_locked()

        await self.publish_event("state_updated", {"map": None, "path": []})
        await self.publish_event("map_updated", None)
        await self.publish_event("ui_state_updated", ui_payload)
        return {"cleared": True, "scope": "low_level_map"}

    async def clear_semantic_memory(self) -> dict[str, Any]:
        if self._goto_poi_task is not None:
            self._goto_poi_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._goto_poi_task
            self._goto_poi_task = None

        if self._chat_task is not None:
            self._chat_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._chat_task

        async with self._state_lock:
            self.storage.clear_semantic_memory()
            self._chat_task = None
            self.chat_state = ChatState()
            self.pois = {}
            self.yolo_objects = {}
            self.ui_state.selected_item = None
            self.ui_state.highlighted_items = []
            self.inspection_state = {"status": "idle", "message": "", "poi_id": None}
            ui_payload = self._commit_ui_state_locked()
            chat_payload = self._serialize_chat_locked()
            inspection_payload = dict(self.inspection_state)

        await self.publish_event("state_updated", {"pois": [], "yolo_objects": []})
        await self.publish_event("inspection_updated", inspection_payload)
        await self.publish_event("ui_state_updated", ui_payload)
        await self.publish_event("chat_state_updated", chat_payload)
        return {"cleared": True, "scope": "semantic_memory"}

    async def chat_snapshot(self) -> dict[str, Any]:
        async with self._state_lock:
            return self._serialize_chat_locked()

    async def chat_tools_manifest(self) -> list[dict[str, Any]]:
        return self.chat_agent.tool_manifest()

    async def submit_chat_message(self, message: str) -> dict[str, Any]:
        content = message.strip()
        if not content:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        async with self._state_lock:
            if self.chat_state.running:
                raise HTTPException(status_code=409, detail="Chat agent is already running")
            history = list(self.chat_state.messages)
            user_message = self._new_chat_message(role="user", content=content)
            assistant_message = self._new_chat_message(role="assistant", content="", status="running")
            self.chat_state.messages.extend([user_message, assistant_message])
            self.chat_state.running = True
            self._persist_chat_state_locked()
            payload = self._serialize_chat_locked()
            self._chat_task = asyncio.create_task(
                self._run_chat_turn(
                    assistant_message_id=assistant_message.message_id,
                    history=history,
                    user_message=content,
                )
            )
        await self.publish_event("chat_state_updated", payload)
        return payload

    async def reset_chat(self) -> dict[str, Any]:
        if self._chat_task is not None:
            self._chat_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await self._chat_task
        async with self._state_lock:
            self._chat_task = None
            self.chat_state = ChatState()
            self._persist_chat_state_locked()
            payload = self._serialize_chat_locked()
        await self.publish_event("chat_state_updated", payload)
        return payload

    async def _run_chat_turn(
        self,
        *,
        assistant_message_id: str,
        history: list[ChatMessage],
        user_message: str,
    ) -> None:
        try:
            result = await self.chat_agent.run_turn(
                self,
                history=history,
                user_message=user_message,
            )
            async with self._state_lock:
                assistant_message = self._require_chat_message_locked(assistant_message_id)
                assistant_message.content = result.content
                assistant_message.status = "final"
                assistant_message.tools_used = result.tools_used
                self.chat_state.running = False
                self._chat_task = None
                self._persist_chat_state_locked()
                payload = self._serialize_chat_locked()
        except Exception as exc:
            async with self._state_lock:
                assistant_message = self._require_chat_message_locked(assistant_message_id)
                assistant_message.content = str(exc)
                assistant_message.status = "error"
                assistant_message.tools_used = []
                self.chat_state.running = False
                self._chat_task = None
                self._persist_chat_state_locked()
                payload = self._serialize_chat_locked()
        await self.publish_event("chat_state_updated", payload)

    async def chat_runtime_overview(self) -> dict[str, Any]:
        async with self._state_lock:
            return {
                "connected": self.connected,
                "robot_pose": self._serialize_pose(self.robot_pose),
                "path_points": len(self.path),
                "map_available": self.map_state is not None,
                "poi_count": len(self._active_pois()),
                "yolo_object_count": len(self._active_yolo_objects()),
                "selected_item": self._serialize_item_ref(self.ui_state.selected_item),
                "highlighted_items": [
                    self._serialize_item_ref(item_ref) for item_ref in self.ui_state.highlighted_items
                ],
                "layers": self._serialize_layers_locked(),
                "yolo_runtime": self._serialize_yolo_runtime_locked(),
                "openai_configured": self._openai_configured,
            }

    async def chat_search_semantic_memory(
        self,
        *,
        query: str,
        kind: str = "all",
        limit: int = 5,
    ) -> dict[str, Any]:
        async with self._state_lock:
            results = self._search_semantic_items_locked(query=query, kind=kind, limit=limit)
        return {"query": query, "kind": kind, "results": results}

    async def chat_get_semantic_item(self, *, kind: str, entity_id: str) -> dict[str, Any]:
        async with self._state_lock:
            item_ref = self._require_item_ref_locked(kind, entity_id)
            if item_ref.kind == SEMANTIC_KIND_POI:
                poi = self._require_active_poi_locked(item_ref.entity_id)
                return {
                    "kind": SEMANTIC_KIND_POI,
                    "entity_id": poi.poi_id,
                    "title": poi.title,
                    "summary": poi.summary,
                    "category": poi.category,
                    "objects": poi.objects,
                    "anchor_x": poi.anchor_x,
                    "anchor_y": poi.anchor_y,
                    "anchor_yaw": poi.anchor_yaw,
                    "target_x": poi.target_x,
                    "target_y": poi.target_y,
                    "updated_at": poi.updated_at,
                }
            object_record = self._require_active_yolo_object_locked(item_ref.entity_id)
            return {
                "kind": SEMANTIC_KIND_YOLO,
                "entity_id": object_record.object_id,
                "label": object_record.label,
                "detections_count": object_record.detections_count,
                "best_confidence": object_record.best_confidence,
                "world_x": object_record.world_x,
                "world_y": object_record.world_y,
                "world_z": object_record.world_z,
                "best_view_x": object_record.best_view_x,
                "best_view_y": object_record.best_view_y,
                "best_view_yaw": object_record.best_view_yaw,
                "updated_at": object_record.updated_at,
                "last_seen_at": object_record.last_seen_at,
            }

    async def chat_focus_semantic_item(
        self,
        *,
        kind: str,
        entity_id: str,
        zoom: float | None = None,
    ) -> dict[str, Any]:
        payload = await self.focus_item(kind, entity_id, zoom=zoom)
        return {"ok": True, "ui": payload}

    async def chat_highlight_semantic_items(
        self,
        *,
        items: list[dict[str, str]],
        selected_item: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = await self.highlight_items(
            [SemanticItemRef(kind=item["kind"], entity_id=item["entity_id"]) for item in items],
            selected_item=(
                SemanticItemRef(kind=selected_item["kind"], entity_id=selected_item["entity_id"])
                if selected_item is not None
                else None
            ),
        )
        return {"ok": True, "ui": payload}

    async def chat_focus_map(self) -> dict[str, Any]:
        payload = await self.focus_map()
        return {"ok": True, "ui": payload}

    async def chat_focus_robot(self, *, zoom: float | None = None) -> dict[str, Any]:
        payload = await self.focus_robot(zoom=zoom)
        return {"ok": True, "ui": payload}

    async def chat_clear_map_focus(self) -> dict[str, Any]:
        payload = await self.clear_ui_focus()
        return {"ok": True, "ui": payload}

    async def chat_set_layer_visibility(
        self,
        *,
        show_pois: bool | None = None,
        show_yolo: bool | None = None,
    ) -> dict[str, Any]:
        payload = await self.set_layer_visibility(show_pois=show_pois, show_yolo=show_yolo)
        return {"ok": True, "layers": payload}

    async def chat_set_yolo_runtime(self, *, mode: str) -> dict[str, Any]:
        payload = await self.set_yolo_mode(mode)
        return {"ok": True, "yolo_runtime": payload}

    async def chat_save_map(self) -> dict[str, Any]:
        payload = await self.save_map()
        return {"ok": True, **payload}

    async def chat_go_to_semantic_item(self, *, kind: str, entity_id: str) -> dict[str, Any]:
        if kind == SEMANTIC_KIND_POI:
            await self.go_to_poi(entity_id)
        elif kind == SEMANTIC_KIND_YOLO:
            await self.go_to_yolo_object(entity_id)
        else:
            raise HTTPException(status_code=404, detail=f"Unknown semantic item kind: {kind}")
        return {"ok": True, "kind": kind, "entity_id": entity_id}

    async def chat_inspect_now(self) -> dict[str, Any]:
        return await self.inspect_now()

    async def chat_look_current_view(self, *, query: str) -> dict[str, Any]:
        image_bytes = await self._observe_jpeg()
        if image_bytes is None:
            return {"ok": False, "error": "No image available from observe()"}
        dimos_image = jpeg_bytes_to_dimos_image(image_bytes)
        prompt = (
            "Answer the operator's question about the current live robot view. "
            "Base the answer only on visible evidence and keep it concise.\n\n"
            f"Question: {query.strip() or 'What is visible in the current view?'}"
        )
        if self._chat_vlm is None:
            return {
                "ok": False,
                "error": "OpenAI is not configured (set OPENAI_API_KEY to enable look_current_view).",
            }
        answer = await asyncio.to_thread(self._chat_vlm.query, dimos_image, prompt)
        return {"ok": True, "answer": answer}

    async def chat_relative_move(
        self,
        *,
        forward: float = 0.0,
        left: float = 0.0,
        degrees: float = 0.0,
    ) -> dict[str, Any]:
        text = await self.mcp_client.call_tool_text(
            "relative_move",
            {"forward": forward, "left": left, "degrees": degrees},
        )
        return {"ok": True, "result": text}

    async def chat_wait(self, *, seconds: float) -> dict[str, Any]:
        text = await self.mcp_client.call_tool_text("wait", {"seconds": seconds})
        return {"ok": True, "result": text}

    async def chat_execute_sport_command(self, *, command_name: str) -> dict[str, Any]:
        text = await self.mcp_client.call_tool_text(
            "execute_sport_command",
            {"command_name": command_name},
        )
        return {"ok": True, "result": text}

    async def chat_list_sport_commands(self) -> dict[str, Any]:
        commands = [
            {"name": name, "description": description}
            for name, _, description in UNITREE_WEBRTC_CONTROLS
            if name not in {"Reverse", "Spin"}
        ]
        return {"commands": commands}

    async def save_map(self) -> dict[str, Any]:
        record = await self.flush_active_map(force=True)
        if record is None:
            raise HTTPException(status_code=400, detail="No active map to save")
        await self.publish_event("map_updated", self._serialize_map())
        return {"saved": True, "updated_at": record.updated_at}

    async def inspect_now(self) -> dict[str, Any]:
        async with self._state_lock:
            if self.inspection_state["status"] == "running":
                raise HTTPException(status_code=409, detail="Inspection already running")
            pose = self.robot_pose
            manual_mode = self.inspection_settings.manual_mode
            self.inspection_state = {"status": "running", "message": "Inspecting ...", "poi_id": None}
        await self.publish_event("inspection_updated", dict(self.inspection_state))

        if pose is None:
            await self._set_inspection_state("failed", "No robot pose available", None)
            raise HTTPException(status_code=409, detail="No robot pose available")

        try:
            image_bytes = await self._observe_jpeg()
            if image_bytes is None:
                raise RuntimeError("No image available from observe()")

            thumbnail_bytes = make_thumbnail(image_bytes)
            hero_path = self.storage.create_image_asset(image_bytes, ".jpg")
            thumb_path = self.storage.create_image_asset(thumbnail_bytes, ".jpg")
            if self.analyzer is None:
                # No VLM: still save the frame as a POI so Inspect works offline (like user POV snapshot).
                payload_json = json.dumps(
                    {
                        "source": "inspect_now",
                        "vlm": False,
                        "title": "Inspection",
                        "summary": (
                            "Captured without OpenAI VLM. Set OPENAI_API_KEY for AI title, summary, "
                            "and quality gate."
                        ),
                    },
                    ensure_ascii=True,
                )
                await self._set_inspection_state(
                    "running",
                    "Marking Point of Interest ...",
                    None,
                )
                async with self._state_lock:
                    map_id = self.map_state.map_id if self.map_state is not None else "active"
                    poi = self.storage.new_poi(
                        map_id=map_id,
                        anchor_x=pose.x,
                        anchor_y=pose.y,
                        anchor_yaw=pose.yaw,
                        title="Inspection",
                        summary=(
                            "Saved without VLM. Add OPENAI_API_KEY to enable AI analysis on Inspect."
                        ),
                        category="Inspection",
                        interest_score=0.5,
                        thumbnail_path=thumb_path,
                        hero_image_path=hero_path,
                        objects=[],
                    )
                    self.storage.upsert_poi(poi)
                    self.pois[poi.poi_id] = poi

                await self.publish_event("poi_upserted", self._serialize_poi(poi))
                await self.focus_poi(poi.poi_id, zoom=UI_FOCUS_POI_ZOOM)
                gate_message = (
                    "Saved without OpenAI (VLM). Add OPENAI_API_KEY for AI inspect."
                )
                await self._set_inspection_state("accepted", gate_message, poi.poi_id)

                observation = self.storage.new_observation(
                    poi_id=poi.poi_id,
                    world_x=pose.x,
                    world_y=pose.y,
                    world_yaw=pose.yaw,
                    image_path=hero_path,
                    thumbnail_path=thumb_path,
                    model_payload_json=payload_json,
                    gate_result="no_vlm",
                )
                self.storage.insert_observation(observation)
                return {
                    "status": self.inspection_state["status"],
                    "poi_id": poi.poi_id,
                    "analysis": {
                        "source": "inspect_now",
                        "vlm": False,
                        "should_create_poi": True,
                        "title": "Inspection",
                        "summary": (
                            "Captured without OpenAI VLM. Set OPENAI_API_KEY for AI title, summary, "
                            "and quality gate."
                        ),
                    },
                    "manual_mode": manual_mode,
                }

            dimos_image = jpeg_bytes_to_dimos_image(image_bytes)
            analysis = self.analyzer.analyze(dimos_image)
            payload_json = json.dumps(analysis.as_payload())
            effective_create_poi = analysis.should_create_poi
            gate_result = "accepted"
            gate_message = analysis.gate_reason
            if manual_mode == INSPECTION_MODE_ALWAYS_CREATE and not analysis.should_create_poi:
                effective_create_poi = True
                gate_result = "forced_accept"
                gate_message = f"Saved by manual override. AI note: {analysis.gate_reason}"
            elif not analysis.should_create_poi:
                gate_result = "rejected"

            poi: PoiRecord | None = None
            if effective_create_poi:
                await self._set_inspection_state(
                    "running",
                    "Marking Point of Interest ...",
                    None,
                )
                async with self._state_lock:
                    duplicate = self._find_duplicate_poi_locked(analysis, pose)
                    if duplicate is not None:
                        poi = self._updated_poi_from_existing(
                            duplicate,
                            analysis,
                            pose,
                            hero_path,
                            thumb_path,
                        )
                    else:
                        map_id = self.map_state.map_id if self.map_state is not None else "active"
                        poi = self.storage.new_poi(
                            map_id=map_id,
                            anchor_x=pose.x,
                            anchor_y=pose.y,
                            anchor_yaw=pose.yaw,
                            title=analysis.title,
                            summary=analysis.summary,
                            category=analysis.category,
                            interest_score=analysis.interest_score,
                            thumbnail_path=thumb_path,
                            hero_image_path=hero_path,
                            objects=analysis.objects,
                        )

                    self.storage.upsert_poi(poi)
                    self.pois[poi.poi_id] = poi

                await self.publish_event("poi_upserted", self._serialize_poi(poi))
                await self.focus_poi(poi.poi_id, zoom=UI_FOCUS_POI_ZOOM)
                await self._set_inspection_state("accepted", gate_message, poi.poi_id)
            else:
                await self._set_inspection_state("rejected", gate_message, None)

            observation = self.storage.new_observation(
                poi_id=poi.poi_id if poi is not None else None,
                world_x=pose.x,
                world_y=pose.y,
                world_yaw=pose.yaw,
                image_path=hero_path,
                thumbnail_path=thumb_path,
                model_payload_json=payload_json,
                gate_result=gate_result,
            )
            self.storage.insert_observation(observation)
            return {
                "status": self.inspection_state["status"],
                "poi_id": poi.poi_id if poi is not None else None,
                "analysis": analysis.as_payload(),
                "manual_mode": manual_mode,
            }
        except (OpenAIError, aiohttp.ClientError, RuntimeError, ValueError) as exc:
            logger.exception("Inspect Now failed")
            await self._set_inspection_state("failed", str(exc), None)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def ingest_user_pov_snapshot(self, image_bytes: bytes) -> dict[str, Any]:
        """Persist a user-captured POV JPEG as a POI so it appears in Detections (semantic list)."""
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image body")
        if len(image_bytes) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large")

        try:
            dimos_image = jpeg_bytes_to_dimos_image(image_bytes)
        except Exception as exc:
            logger.info("POV snapshot JPEG decode failed: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid JPEG image") from exc

        async with self._state_lock:
            pose = self.robot_pose
            map_state = self.map_state
        if pose is None:
            raise HTTPException(status_code=409, detail="No robot pose available")

        map_id = map_state.map_id if map_state is not None else "active"

        thumbnail_bytes = make_thumbnail(image_bytes)
        hero_path = self.storage.create_image_asset(image_bytes, ".jpg")
        thumb_path = self.storage.create_image_asset(thumbnail_bytes, ".jpg")

        poi: PoiRecord
        payload_json: str

        try:
            if self.analyzer is not None:
                analysis = self.analyzer.analyze(dimos_image)
                payload_json = json.dumps(analysis.as_payload())
                async with self._state_lock:
                    duplicate = self._find_duplicate_poi_locked(analysis, pose)
                    if duplicate is not None:
                        poi = self._updated_poi_from_existing(
                            duplicate, analysis, pose, hero_path, thumb_path
                        )
                    else:
                        poi = self.storage.new_poi(
                            map_id=map_id,
                            anchor_x=pose.x,
                            anchor_y=pose.y,
                            anchor_yaw=pose.yaw,
                            title=analysis.title,
                            summary=analysis.summary,
                            category=analysis.category,
                            interest_score=analysis.interest_score,
                            thumbnail_path=thumb_path,
                            hero_image_path=hero_path,
                            objects=analysis.objects,
                        )
                    self.storage.upsert_poi(poi)
                    self.pois[poi.poi_id] = poi
            else:
                payload_json = json.dumps({"source": "user_pov_snapshot", "vlm": False})
                async with self._state_lock:
                    poi = self.storage.new_poi(
                        map_id=map_id,
                        anchor_x=pose.x,
                        anchor_y=pose.y,
                        anchor_yaw=pose.yaw,
                        title="POV snapshot",
                        summary="Saved from the live camera feed.",
                        category="User snapshot",
                        interest_score=0.5,
                        thumbnail_path=thumb_path,
                        hero_image_path=hero_path,
                        objects=[],
                    )
                    self.storage.upsert_poi(poi)
                    self.pois[poi.poi_id] = poi
        except OpenAIError as exc:
            logger.exception("POV snapshot VLM failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        observation = self.storage.new_observation(
            poi_id=poi.poi_id,
            world_x=pose.x,
            world_y=pose.y,
            world_yaw=pose.yaw,
            image_path=hero_path,
            thumbnail_path=thumb_path,
            model_payload_json=payload_json,
            gate_result="user_snapshot",
        )
        self.storage.insert_observation(observation)

        await self.publish_event("poi_upserted", self._serialize_poi(poi))
        await self.focus_poi(poi.poi_id, zoom=UI_FOCUS_POI_ZOOM)
        return {"poi_id": poi.poi_id, "poi": self._serialize_poi(poi)}

    async def delete_poi(self, poi_id: str) -> None:
        ui_payload: dict[str, Any] | None = None
        async with self._state_lock:
            poi = self.pois.get(poi_id)
            if poi is None:
                raise HTTPException(status_code=404, detail="POI not found")
            was_selected = self._is_selected_locked(SEMANTIC_KIND_POI, poi_id)
            was_highlighted = self._is_highlighted_locked(SEMANTIC_KIND_POI, poi_id)
            self.storage.soft_delete_poi(poi_id)
            self.pois[poi_id] = PoiRecord(
                poi_id=poi.poi_id,
                map_id=poi.map_id,
                anchor_x=poi.anchor_x,
                anchor_y=poi.anchor_y,
                anchor_yaw=poi.anchor_yaw,
                target_x=poi.target_x,
                target_y=poi.target_y,
                title=poi.title,
                summary=poi.summary,
                category=poi.category,
                interest_score=poi.interest_score,
                status="deleted",
                thumbnail_path=poi.thumbnail_path,
                hero_image_path=poi.hero_image_path,
                objects_json=poi.objects_json,
                created_at=poi.created_at,
                updated_at=utc_now_iso(),
            )
            if was_selected:
                self.ui_state.selected_item = None
            if was_highlighted:
                self.ui_state.highlighted_items = [
                    highlighted
                    for highlighted in self.ui_state.highlighted_items
                    if not (
                        highlighted.kind == SEMANTIC_KIND_POI
                        and highlighted.entity_id == poi_id
                    )
                ]
            if was_selected or was_highlighted:
                ui_payload = self._commit_ui_state_locked()
        await self.publish_event("poi_deleted", {"poi_id": poi_id})
        if ui_payload is not None:
            await self.publish_event("ui_state_updated", ui_payload)

    async def delete_yolo_object(self, object_id: str) -> None:
        ui_payload: dict[str, Any] | None = None
        async with self._state_lock:
            object_record = self.yolo_objects.get(object_id)
            if object_record is None:
                raise HTTPException(status_code=404, detail="YOLO object not found")
            was_selected = self._is_selected_locked(SEMANTIC_KIND_YOLO, object_id)
            was_highlighted = self._is_highlighted_locked(SEMANTIC_KIND_YOLO, object_id)
            self.storage.soft_delete_yolo_object(object_id)
            self.yolo_objects[object_id] = YoloObjectRecord(
                object_id=object_record.object_id,
                map_id=object_record.map_id,
                label=object_record.label,
                class_id=object_record.class_id,
                world_x=object_record.world_x,
                world_y=object_record.world_y,
                world_z=object_record.world_z,
                size_x=object_record.size_x,
                size_y=object_record.size_y,
                size_z=object_record.size_z,
                best_view_x=object_record.best_view_x,
                best_view_y=object_record.best_view_y,
                best_view_yaw=object_record.best_view_yaw,
                status="deleted",
                thumbnail_path=object_record.thumbnail_path,
                hero_image_path=object_record.hero_image_path,
                detections_count=object_record.detections_count,
                best_confidence=object_record.best_confidence,
                created_at=object_record.created_at,
                updated_at=utc_now_iso(),
                last_seen_at=object_record.last_seen_at,
            )
            if was_selected:
                self.ui_state.selected_item = None
            if was_highlighted:
                self.ui_state.highlighted_items = [
                    highlighted
                    for highlighted in self.ui_state.highlighted_items
                    if not (
                        highlighted.kind == SEMANTIC_KIND_YOLO
                        and highlighted.entity_id == object_id
                    )
                ]
            if was_selected or was_highlighted:
                ui_payload = self._commit_ui_state_locked()
        await self.publish_event("yolo_object_deleted", {"object_id": object_id})
        if ui_payload is not None:
            await self.publish_event("ui_state_updated", ui_payload)

    def resolve_asset(self, relative_path: str) -> Path:
        resolved = (self.state_dir / relative_path).resolve()
        state_root = self.state_dir.resolve()
        if state_root != resolved and state_root not in resolved.parents:
            raise HTTPException(status_code=404, detail="Asset not found")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Asset not found")
        return resolved

    async def flush_active_map(self, force: bool = False) -> ActiveMapRecord | None:
        async with self._state_lock:
            if self.map_state is None:
                return None
            if not self._dirty_map and not force:
                return self.map_record

            preview = self.map_state.preview_png_bytes()
            record = self.storage.save_active_map(
                map_id=self.map_state.map_id,
                resolution=self.map_state.resolution,
                origin_x=self.map_state.origin_x,
                origin_y=self.map_state.origin_y,
                log_odds=self.map_state.log_odds,
                observation_count=self.map_state.observation_count,
                preview_png=preview,
            )
            self.map_record = record
            self.map_state.updated_at = record.updated_at
            self._dirty_map = False
            return record

    def _load_from_storage(self) -> None:
        record, log_odds, observation_count = self.storage.load_active_map()
        self.map_record = record
        settings = self.storage.load_json_setting("inspection_settings")
        if settings is not None:
            try:
                raw_manual_mode = settings.get("manual_mode", INSPECTION_MODE_AI_GATE)
                self.inspection_settings = InspectionSettings(
                    manual_mode=normalize_manual_inspection_mode(str(raw_manual_mode))
                )
            except ValueError:
                logger.warning("Ignoring invalid persisted inspection settings", settings=settings)
        yolo_runtime = self.storage.load_json_setting("yolo_runtime")
        if yolo_runtime is not None:
            try:
                self.yolo_runtime = YoloRuntimeState(
                    mode=normalize_yolo_mode(str(yolo_runtime.get("mode", YOLO_MODE_LIVE))),
                    inference_enabled=bool(yolo_runtime.get("inference_enabled", True)),
                )
            except ValueError:
                logger.warning("Ignoring invalid persisted YOLO runtime", settings=yolo_runtime)
        layer_visibility = self.storage.load_json_setting("layer_visibility")
        if layer_visibility is not None:
            self.layer_visibility = LayerVisibilityState(
                show_pois=bool(layer_visibility.get("show_pois", True)),
                show_yolo=bool(layer_visibility.get("show_yolo", YOLO_DEFAULT_LAYER_VISIBLE)),
            )
        chat_state = self.storage.load_json_setting("chat_state")
        if chat_state is not None:
            messages = chat_state.get("messages", [])
            if isinstance(messages, list):
                restored_messages: list[ChatMessage] = []
                for item in messages:
                    if not isinstance(item, dict):
                        continue
                    restored_messages.append(
                        ChatMessage(
                            message_id=str(item.get("message_id", "")),
                            role=str(item.get("role", "assistant")),
                            content=str(item.get("content", "")),
                            created_at=str(item.get("created_at", utc_now_iso())),
                            status=str(item.get("status", "final")),
                            tools_used=[
                                str(tool_name)
                                for tool_name in item.get("tools_used", [])
                                if isinstance(tool_name, str)
                            ],
                        )
                    )
                self.chat_state = ChatState(messages=restored_messages, running=False)
        if record is not None and log_odds is not None and observation_count is not None:
            self.map_state = ActiveMapState.from_arrays(
                map_id=record.map_id,
                resolution=record.resolution,
                origin_x=record.origin_x,
                origin_y=record.origin_y,
                log_odds=log_odds,
                observation_count=observation_count,
                updated_at=record.updated_at,
            )
        self.pois = {poi.poi_id: poi for poi in self.storage.list_pois(include_deleted=True)}
        self.yolo_objects = {
            object_record.object_id: object_record
            for object_record in self.storage.list_yolo_objects(include_deleted=True)
        }
        if self.map_state is not None:
            center_x, center_y = self._map_center_locked()
            self.ui_state.camera = UiCameraState(center_x=center_x, center_y=center_y, zoom=UI_DEFAULT_MAP_ZOOM)

    async def _set_inspection_state(self, status: str, message: str, poi_id: str | None) -> None:
        async with self._state_lock:
            self.inspection_state = {"status": status, "message": message, "poi_id": poi_id}
        await self.publish_event("inspection_updated", dict(self.inspection_state))

    async def _handle_connection_change(self, connected: bool) -> None:
        async with self._state_lock:
            self.connected = connected
        await self.publish_event("state_updated", await self._state_delta())

    async def _handle_pose(self, pose: RobotPose) -> None:
        should_publish = False
        async with self._state_lock:
            self.robot_pose = pose
            now = time.monotonic()
            if now - self._last_pose_event > 0.2:
                self._last_pose_event = now
                should_publish = True
        if should_publish:
            await self.publish_event("state_updated", await self._state_delta())

    async def _handle_path(self, path: list[list[float]]) -> None:
        async with self._state_lock:
            self.path = path
        await self.publish_event("state_updated", await self._state_delta())

    async def _handle_raw_costmap(self, raw_costmap: RawCostmap) -> None:
        ui_payload: dict[str, Any] | None = None
        async with self._state_lock:
            if self.map_state is None:
                self.map_state = ActiveMapState.empty_from_extent(
                    map_id="active",
                    resolution=ACTIVE_MAP_GRID_RESOLUTION,
                    min_x=raw_costmap.origin_x,
                    min_y=raw_costmap.origin_y,
                    max_x=raw_costmap.origin_x + raw_costmap.width * raw_costmap.resolution,
                    max_y=raw_costmap.origin_y + raw_costmap.height * raw_costmap.resolution,
                )
                center_x, center_y = self._map_center_locked()
                self.ui_state.camera = UiCameraState(center_x=center_x, center_y=center_y, zoom=UI_DEFAULT_MAP_ZOOM)
                ui_payload = self._commit_ui_state_locked()

            changed = self.map_state.update_from_costmap(raw_costmap)
            if not changed:
                if ui_payload is None:
                    return

            payload: dict[str, Any] | None = None
            if changed:
                self.map_state.updated_at = utc_now_iso()
                preview = self.map_state.preview_png_bytes()
                self.storage.write_active_map_preview(preview)
                self._dirty_map = True
                payload = self._serialize_map()

        if payload is not None:
            await self.publish_event("map_updated", payload)
        if ui_payload is not None:
            await self.publish_event("ui_state_updated", ui_payload)

    async def _handle_yolo_detections(self, payload: dict[str, Any]) -> None:
        detections_payload = payload.get("detections")
        view_pose = payload.get("view_pose")
        if not isinstance(detections_payload, list) or not isinstance(view_pose, dict):
            return

        published_objects: list[dict[str, Any]] = []
        async with self._state_lock:
            if (
                self.yolo_runtime.mode != YOLO_MODE_LIVE
                or not self.yolo_runtime.inference_enabled
            ):
                return
            map_id = self.map_state.map_id if self.map_state is not None else "active"
            for item in detections_payload:
                detection = self._parse_yolo_detection_payload(item, view_pose)
                if detection is None:
                    continue
                object_record, observation = self._ingest_yolo_detection_locked(
                    map_id=map_id,
                    ts=float(payload.get("ts", time.time())),
                    detection=detection,
                )
                if observation is not None:
                    self.storage.insert_yolo_observation(observation)
                if object_record is not None:
                    published_objects.append(self._serialize_yolo_object(object_record))

        for object_payload in published_objects:
            await self.publish_event("yolo_object_upserted", object_payload)

    async def _handle_live_pov_frame(self, image_bytes: bytes) -> None:
        try:
            prepared = prepare_pov_jpeg(
                image_bytes,
                max_width=self._pov_max_width,
                quality=self._pov_jpeg_quality,
            )
        except Exception:
            logger.debug("POV socket frame could not be normalized", exc_info=True)
            return
        async with self._state_lock:
            self._last_socket_pov_monotonic = time.monotonic()
            self.latest_pov_jpeg = prepared
            self._pov_live = True
            self.pov_seq += 1
            self.pov_updated_at = utc_now_iso()
        await self.publish_event("state_updated", await self._state_delta())

    def _socket_pov_is_fresh(self) -> bool:
        if self._last_socket_pov_monotonic <= 0.0:
            return False
        return (
            time.monotonic() - self._last_socket_pov_monotonic
            <= MAP_SOCKET_POV_STALE_AFTER_SECONDS
        )

    async def _pov_loop(self) -> None:
        next_poll_at = time.monotonic()
        while not self._stopped:
            if self._socket_pov_is_fresh():
                next_poll_at += self._pov_poll_interval_seconds
                sleep_for = next_poll_at - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    next_poll_at = time.monotonic()
                    await asyncio.sleep(0)
                continue
            try:
                image_bytes = await self._observe_jpeg()
                if image_bytes is not None:
                    image_bytes = prepare_pov_jpeg(
                        image_bytes,
                        max_width=self._pov_max_width,
                        quality=self._pov_jpeg_quality,
                    )
                    async with self._state_lock:
                        self.latest_pov_jpeg = image_bytes
                        self._pov_live = True
                        self.pov_seq += 1
                        self.pov_updated_at = utc_now_iso()
                    await self.publish_event("state_updated", await self._state_delta())
            except Exception as exc:
                logger.debug("POV polling failed", error=str(exc))
            next_poll_at += self._pov_poll_interval_seconds
            sleep_for = next_poll_at - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_poll_at = time.monotonic()
                await asyncio.sleep(0)

    async def _checkpoint_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(10.0)
            try:
                await self.flush_active_map(force=False)
            except Exception:
                logger.exception("Periodic SLAMASS map checkpoint failed")

    async def _battery_loop(self) -> None:
        """Poll the GO2Connection's `get_battery_percent` skill every few seconds.

        The Go2 publishes battery state on its WebRTC datachannel — the dimos
        connection caches the latest value, and this loop reads it through MCP
        so the slamass UI state reflects it. If the firmware never publishes a
        recognised battery field, `battery_percent` stays None and the UI shows
        a dash.
        """
        while not self._stopped:
            await asyncio.sleep(5.0)
            try:
                text = await self.mcp_client.call_tool_text("get_battery_percent")
            except Exception:
                continue
            stripped = text.strip()
            if not stripped or stripped.lower() == "unknown":
                continue
            try:
                pct = int(stripped)
            except ValueError:
                continue
            if 0 <= pct <= 100 and pct != self.battery_percent:
                self.battery_percent = pct
                await self.publish_event("battery_updated", {"battery_percent": pct})

    async def _finish_poi_navigation(
        self,
        *,
        poi_id: str,
        target_x: float,
        target_y: float,
        target_yaw: float,
    ) -> None:
        try:
            arrived = await self._wait_for_poi_arrival(target_x=target_x, target_y=target_y)
            if not arrived:
                logger.warning(
                    "POI Go To timed out before reaching target pose",
                    poi_id=poi_id,
                    x=round(target_x, 3),
                    y=round(target_y, 3),
                )
                return

            async with self._state_lock:
                current_pose = self.robot_pose
            if current_pose is None:
                logger.warning("POI Go To lost robot pose before yaw restore", poi_id=poi_id)
                return

            rotation_degrees = math.degrees(angle_delta(target_yaw, current_pose.yaw))
            if abs(rotation_degrees) < self._poi_rotation_threshold_degrees:
                logger.info(
                    "POI Go To arrived already aligned with saved viewpoint",
                    poi_id=poi_id,
                    yaw_error_degrees=round(rotation_degrees, 1),
                )
                return

            logger.info(
                "POI Go To restoring saved viewpoint yaw",
                poi_id=poi_id,
                degrees=round(rotation_degrees, 1),
            )
            await self._relative_rotate(rotation_degrees)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("POI Go To yaw restoration failed", poi_id=poi_id)
        finally:
            current_task = asyncio.current_task()
            if self._goto_poi_task is current_task:
                self._goto_poi_task = None

    async def _wait_for_poi_arrival(self, *, target_x: float, target_y: float) -> bool:
        deadline = time.monotonic() + self._poi_arrival_timeout_seconds
        in_tolerance_since: float | None = None

        while not self._stopped and time.monotonic() < deadline:
            async with self._state_lock:
                pose = self.robot_pose

            if pose is not None:
                distance = math.hypot(pose.x - target_x, pose.y - target_y)
                if distance <= self._poi_arrival_tolerance_m:
                    if in_tolerance_since is None:
                        in_tolerance_since = time.monotonic()
                    elif time.monotonic() - in_tolerance_since >= self._poi_arrival_settle_seconds:
                        return True
                else:
                    in_tolerance_since = None

            await asyncio.sleep(0.25)

        return False

    async def _relative_rotate(self, degrees: float) -> None:
        result = self.mcp_client.relative_move(forward=0.0, left=0.0, degrees=degrees)
        if asyncio.iscoroutine(result):
            await result

    async def _observe_jpeg(self) -> bytes | None:
        result = self.mcp_client.observe_jpeg()
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _set_yolo_inference_enabled(
        self,
        inference_enabled: bool,
        *,
        best_effort: bool = False,
    ) -> None:
        setter = getattr(self.mcp_client, "set_yolo_inference", None)
        try:
            if setter is not None:
                result = setter(enabled=inference_enabled)
            else:
                result = self.mcp_client.call_tool_text(
                    "set_yolo_inference",
                    {"enabled": inference_enabled},
                )
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, str) and (
                result.startswith("Tool not found:") or result.startswith("Error running tool")
            ):
                raise RuntimeError(result)
        except Exception as exc:
            if best_effort:
                logger.warning(
                    "Could not sync YOLO inference state to detector",
                    error=str(exc),
                    inference_enabled=inference_enabled,
                )
                return
            raise RuntimeError(f"Failed to update YOLO inference: {exc}") from exc

    async def _maybe_start_mcp_client(self) -> None:
        start = getattr(self.mcp_client, "start", None)
        if start is None:
            return
        result = start()
        if asyncio.iscoroutine(result):
            await result

    async def _maybe_stop_mcp_client(self) -> None:
        stop = getattr(self.mcp_client, "stop", None)
        if stop is None:
            return
        result = stop()
        if asyncio.iscoroutine(result):
            await result

    async def _state_delta(self) -> dict[str, Any]:
        async with self._state_lock:
            return {
                "connected": self.connected,
                "battery_percent": self.battery_percent,
                "robot_pose": self._serialize_pose(self.robot_pose),
                "path": self.path,
                "pov": {
                    "available": self._pov_live,
                    "seq": self.pov_seq,
                    "updated_at": self.pov_updated_at,
                    "image_url": f"/api/pov/latest.jpg?v={self.pov_seq}",
                },
                "inspection_settings": self._serialize_inspection_settings_locked(),
                "yolo_runtime": self._serialize_yolo_runtime_locked(),
                "layers": self._serialize_layers_locked(),
            }

    def _require_active_poi_locked(self, poi_id: str) -> PoiRecord:
        poi = self.pois.get(poi_id)
        if poi is None or poi.status == "deleted":
            raise HTTPException(status_code=404, detail="POI not found")
        return poi

    def _require_active_yolo_object_locked(self, object_id: str) -> YoloObjectRecord:
        object_record = self.yolo_objects.get(object_id)
        if object_record is None or object_record.status == "deleted":
            raise HTTPException(status_code=404, detail="YOLO object not found")
        return object_record

    def _require_item_ref_locked(self, kind: str, entity_id: str) -> SemanticItemRef:
        if kind == SEMANTIC_KIND_POI:
            self._require_active_poi_locked(entity_id)
            return SemanticItemRef(kind=kind, entity_id=entity_id)
        if kind == SEMANTIC_KIND_YOLO:
            self._require_active_yolo_object_locked(entity_id)
            return SemanticItemRef(kind=kind, entity_id=entity_id)
        raise HTTPException(status_code=404, detail=f"Unknown semantic item kind: {kind}")

    def _validate_optional_item_locked(
        self,
        kind: str | None,
        entity_id: str | None,
    ) -> SemanticItemRef | None:
        if kind is None or entity_id is None:
            return None
        return self._require_item_ref_locked(kind, entity_id)

    def _normalize_item_refs_locked(self, item_refs: list[SemanticItemRef]) -> list[SemanticItemRef]:
        normalized: list[SemanticItemRef] = []
        seen: set[tuple[str, str]] = set()
        for item_ref in item_refs:
            validated = self._require_item_ref_locked(item_ref.kind, item_ref.entity_id)
            key = (validated.kind, validated.entity_id)
            if key in seen:
                continue
            normalized.append(validated)
            seen.add(key)
        return normalized

    def _is_selected_locked(self, kind: str, entity_id: str) -> bool:
        return (
            self.ui_state.selected_item is not None
            and self.ui_state.selected_item.kind == kind
            and self.ui_state.selected_item.entity_id == entity_id
        )

    def _is_highlighted_locked(self, kind: str, entity_id: str) -> bool:
        return any(
            highlighted.kind == kind and highlighted.entity_id == entity_id
            for highlighted in self.ui_state.highlighted_items
        )

    def _item_world_xy_locked(self, item_ref: SemanticItemRef) -> tuple[float, float]:
        if item_ref.kind == SEMANTIC_KIND_POI:
            poi = self._require_active_poi_locked(item_ref.entity_id)
            return poi.target_x, poi.target_y
        object_record = self._require_active_yolo_object_locked(item_ref.entity_id)
        return object_record.world_x, object_record.world_y

    def _map_center_locked(self) -> tuple[float, float]:
        assert self.map_state is not None
        return (
            self.map_state.origin_x + (self.map_state.width * self.map_state.resolution) / 2.0,
            self.map_state.origin_y + (self.map_state.height * self.map_state.resolution) / 2.0,
        )

    def _apply_ui_camera_locked(self, *, center_x: float, center_y: float, zoom: float) -> None:
        clamped_zoom = max(UI_MIN_ZOOM, min(UI_MAX_ZOOM, float(zoom)))
        if self.map_state is not None:
            min_x = self.map_state.origin_x
            max_x = self.map_state.origin_x + self.map_state.width * self.map_state.resolution
            min_y = self.map_state.origin_y
            max_y = self.map_state.origin_y + self.map_state.height * self.map_state.resolution
            center_x = min(max(center_x, min_x), max_x)
            center_y = min(max(center_y, min_y), max_y)
        self.ui_state.camera = UiCameraState(
            center_x=float(center_x),
            center_y=float(center_y),
            zoom=clamped_zoom,
        )

    def _commit_ui_state_locked(self) -> dict[str, Any]:
        self.ui_state.revision += 1
        return self._serialize_ui_locked()

    def _updated_poi_from_existing(
        self,
        existing: PoiRecord,
        analysis: InspectionAnalysis,
        pose: RobotPose,
        hero_path: str,
        thumb_path: str,
    ) -> PoiRecord:
        return self.storage.new_poi(
            map_id=existing.map_id,
            anchor_x=pose.x,
            anchor_y=pose.y,
            anchor_yaw=pose.yaw,
            target_x=existing.target_x,
            target_y=existing.target_y,
            title=analysis.title,
            summary=analysis.summary,
            category=analysis.category,
            interest_score=analysis.interest_score,
            thumbnail_path=thumb_path,
            hero_image_path=hero_path,
            objects=analysis.objects,
            poi_id=existing.poi_id,
            created_at=existing.created_at,
        )

    def _find_duplicate_poi_locked(
        self, analysis: InspectionAnalysis, pose: RobotPose
    ) -> PoiRecord | None:
        title = normalize_text(analysis.title)
        category = normalize_text(analysis.category)
        for poi in self._active_pois():
            if normalize_text(poi.category) != category:
                continue
            if normalize_text(poi.title) != title:
                continue
            distance = math.hypot(poi.anchor_x - pose.x, poi.anchor_y - pose.y)
            if distance > 1.5:
                continue
            if yaw_distance(poi.anchor_yaw, pose.yaw) > math.radians(45):
                continue
            return poi
        return None

    def _active_pois(self) -> list[PoiRecord]:
        return [poi for poi in self.pois.values() if poi.status != "deleted"]

    def _active_yolo_objects(self) -> list[YoloObjectRecord]:
        return [object_record for object_record in self.yolo_objects.values() if object_record.status != "deleted"]

    def _search_semantic_items_locked(
        self,
        *,
        query: str,
        kind: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        normalized_kind = kind.strip().lower() if kind else "all"
        if normalized_kind not in {"all", SEMANTIC_KIND_POI, SEMANTIC_KIND_YOLO}:
            raise HTTPException(status_code=400, detail=f"Unsupported semantic kind: {kind}")

        normalized_query = normalize_text(query)
        tokens = [token for token in normalized_query.split() if token]
        capped_limit = max(1, min(int(limit), 8))
        scored: list[tuple[float, str, dict[str, Any]]] = []

        if normalized_kind in {"all", SEMANTIC_KIND_POI}:
            for poi in self._active_pois():
                score = self._field_score(normalized_query, tokens, poi.title, 4.0)
                score += self._field_score(normalized_query, tokens, poi.category, 3.0)
                score += sum(self._field_score(normalized_query, tokens, item, 2.0) for item in poi.objects)
                score += self._field_score(normalized_query, tokens, poi.summary, 1.0)
                if normalized_query and score <= 0:
                    continue
                scored.append(
                    (
                        score if normalized_query else poi.interest_score,
                        poi.updated_at,
                        {
                            "kind": SEMANTIC_KIND_POI,
                            "entity_id": poi.poi_id,
                            "title": poi.title,
                            "subtitle": poi.category,
                            "summary": poi.summary,
                            "anchor_x": poi.anchor_x,
                            "anchor_y": poi.anchor_y,
                            "anchor_yaw": poi.anchor_yaw,
                            "target_x": poi.target_x,
                            "target_y": poi.target_y,
                            "score": round(score, 3),
                        },
                    )
                )

        if normalized_kind in {"all", SEMANTIC_KIND_YOLO}:
            for object_record in self._active_yolo_objects():
                score = self._field_score(normalized_query, tokens, object_record.label, 4.0)
                if normalized_query and score <= 0:
                    continue
                scored.append(
                    (
                        score if normalized_query else object_record.best_confidence,
                        object_record.updated_at,
                        {
                            "kind": SEMANTIC_KIND_YOLO,
                            "entity_id": object_record.object_id,
                            "title": object_record.label.title(),
                            "subtitle": f"{object_record.best_confidence:.2f} confidence",
                            "summary": (
                                f"{object_record.detections_count} detections. "
                                f"Best view stored at {object_record.best_view_x:.2f}, {object_record.best_view_y:.2f}."
                            ),
                            "world_x": object_record.world_x,
                            "world_y": object_record.world_y,
                            "world_yaw": object_record.best_view_yaw,
                            "score": round(score, 3),
                        },
                    )
                )

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [payload for _, _, payload in scored[:capped_limit]]

    def _field_score(
        self,
        normalized_query: str,
        tokens: list[str],
        value: str,
        weight: float,
    ) -> float:
        normalized_value = normalize_text(value)
        if not normalized_value:
            return 0.0
        score = 0.0
        if normalized_query:
            if normalized_value == normalized_query:
                score += 6.0 * weight
            if normalized_query in normalized_value:
                score += 4.0 * weight
        for token in tokens:
            if token in normalized_value:
                score += weight
        return score

    def _parse_yolo_detection_payload(
        self,
        payload: dict[str, Any],
        view_pose: dict[str, Any],
    ) -> YoloDetection | None:
        label = str(payload.get("label", "")).strip().lower()
        if not label or label not in YOLO_CLASS_WHITELIST:
            return None
        crop_base64 = payload.get("crop_base64", "")
        crop_jpeg = base64.b64decode(crop_base64) if isinstance(crop_base64, str) and crop_base64 else b""
        try:
            return YoloDetection(
                label=label,
                class_id=int(payload.get("class_id", -1)),
                confidence=float(payload.get("confidence", 0.0)),
                world_x=float(payload.get("world_x", 0.0)),
                world_y=float(payload.get("world_y", 0.0)),
                world_z=float(payload.get("world_z", 0.0)),
                size_x=float(payload.get("size_x", 0.0)),
                size_y=float(payload.get("size_y", 0.0)),
                size_z=float(payload.get("size_z", 0.0)),
                view_x=float(view_pose.get("x", 0.0)),
                view_y=float(view_pose.get("y", 0.0)),
                view_yaw=float(view_pose.get("yaw", 0.0)),
                crop_jpeg=crop_jpeg,
            )
        except (TypeError, ValueError):
            return None

    def _ingest_yolo_detection_locked(
        self,
        *,
        map_id: str,
        ts: float,
        detection: YoloDetection,
    ) -> tuple[YoloObjectRecord | None, YoloObservationRecord | None]:
        existing = self._find_yolo_object_locked(detection.label, detection.world_x, detection.world_y)
        if existing is not None:
            updated_object, observation = self._updated_yolo_object_from_existing(existing, detection)
            self.storage.upsert_yolo_object(updated_object)
            self.yolo_objects[updated_object.object_id] = updated_object
            return updated_object, observation

        candidate = self._find_pending_yolo_candidate_locked(
            detection.label,
            detection.world_x,
            detection.world_y,
        )
        if candidate is None:
            candidate = PendingYoloCandidate(label=detection.label, class_id=detection.class_id)
            self.pending_yolo_candidates.append(candidate)
        candidate.add_hit(ts, detection)
        if candidate.hit_count < YOLO_PROMOTION_MIN_HITS:
            return None, None

        promoted_detection = candidate.averaged_detection()
        hero_path, thumb_path = self._create_yolo_assets(promoted_detection.crop_jpeg)
        object_record = self.storage.new_yolo_object(
            map_id=map_id,
            label=promoted_detection.label,
            class_id=promoted_detection.class_id,
            world_x=promoted_detection.world_x,
            world_y=promoted_detection.world_y,
            world_z=promoted_detection.world_z,
            size_x=promoted_detection.size_x,
            size_y=promoted_detection.size_y,
            size_z=promoted_detection.size_z,
            best_view_x=promoted_detection.view_x,
            best_view_y=promoted_detection.view_y,
            best_view_yaw=promoted_detection.view_yaw,
            thumbnail_path=thumb_path,
            hero_image_path=hero_path,
            detections_count=candidate.hit_count,
            best_confidence=promoted_detection.confidence,
        )
        self.storage.upsert_yolo_object(object_record)
        self.yolo_objects[object_record.object_id] = object_record
        self.pending_yolo_candidates = [entry for entry in self.pending_yolo_candidates if entry is not candidate]
        observation = self.storage.new_yolo_observation(
            object_id=object_record.object_id,
            label=promoted_detection.label,
            class_id=promoted_detection.class_id,
            confidence=promoted_detection.confidence,
            world_x=promoted_detection.world_x,
            world_y=promoted_detection.world_y,
            world_z=promoted_detection.world_z,
            size_x=promoted_detection.size_x,
            size_y=promoted_detection.size_y,
            size_z=promoted_detection.size_z,
            view_x=promoted_detection.view_x,
            view_y=promoted_detection.view_y,
            view_yaw=promoted_detection.view_yaw,
            image_path=hero_path,
            thumbnail_path=thumb_path,
        )
        return object_record, observation

    def _find_yolo_object_locked(
        self,
        label: str,
        world_x: float,
        world_y: float,
    ) -> YoloObjectRecord | None:
        for object_record in self._active_yolo_objects():
            if normalize_text(object_record.label) != normalize_text(label):
                continue
            if math.hypot(object_record.world_x - world_x, object_record.world_y - world_y) <= YOLO_DEDUPE_DISTANCE_METERS:
                return object_record
        return None

    def _find_pending_yolo_candidate_locked(
        self,
        label: str,
        world_x: float,
        world_y: float,
    ) -> PendingYoloCandidate | None:
        for candidate in self.pending_yolo_candidates:
            if normalize_text(candidate.label) != normalize_text(label):
                continue
            if not candidate.hits:
                continue
            representative = candidate.representative()
            if math.hypot(representative.world_x - world_x, representative.world_y - world_y) <= YOLO_DEDUPE_DISTANCE_METERS:
                return candidate
        return None

    def _create_yolo_assets(self, crop_jpeg: bytes) -> tuple[str, str]:
        hero_bytes = crop_jpeg or (make_thumbnail(self.latest_pov_jpeg, width=256) if self.latest_pov_jpeg else b"")
        if not hero_bytes:
            hero_bytes = make_placeholder_jpeg()
        thumb_bytes = make_thumbnail(hero_bytes, width=144)
        hero_path = self.storage.create_image_asset(hero_bytes, ".jpg")
        thumb_path = self.storage.create_image_asset(thumb_bytes, ".jpg")
        return hero_path, thumb_path

    def _updated_yolo_object_from_existing(
        self,
        existing: YoloObjectRecord,
        detection: YoloDetection,
    ) -> tuple[YoloObjectRecord, YoloObservationRecord]:
        detections_count = max(1, existing.detections_count)
        next_count = detections_count + 1
        hero_path = existing.hero_image_path
        thumb_path = existing.thumbnail_path
        if detection.confidence >= existing.best_confidence and detection.crop_jpeg:
            hero_path, thumb_path = self._create_yolo_assets(detection.crop_jpeg)
        updated_object = self.storage.new_yolo_object(
            map_id=existing.map_id,
            label=existing.label,
            class_id=existing.class_id,
            world_x=((existing.world_x * detections_count) + detection.world_x) / next_count,
            world_y=((existing.world_y * detections_count) + detection.world_y) / next_count,
            world_z=((existing.world_z * detections_count) + detection.world_z) / next_count,
            size_x=((existing.size_x * detections_count) + detection.size_x) / next_count,
            size_y=((existing.size_y * detections_count) + detection.size_y) / next_count,
            size_z=((existing.size_z * detections_count) + detection.size_z) / next_count,
            best_view_x=detection.view_x if detection.confidence >= existing.best_confidence else existing.best_view_x,
            best_view_y=detection.view_y if detection.confidence >= existing.best_confidence else existing.best_view_y,
            best_view_yaw=detection.view_yaw if detection.confidence >= existing.best_confidence else existing.best_view_yaw,
            thumbnail_path=thumb_path,
            hero_image_path=hero_path,
            detections_count=next_count,
            best_confidence=max(existing.best_confidence, detection.confidence),
            object_id=existing.object_id,
            created_at=existing.created_at,
            last_seen_at=utc_now_iso(),
        )
        observation = self.storage.new_yolo_observation(
            object_id=existing.object_id,
            label=detection.label,
            class_id=detection.class_id,
            confidence=detection.confidence,
            world_x=detection.world_x,
            world_y=detection.world_y,
            world_z=detection.world_z,
            size_x=detection.size_x,
            size_y=detection.size_y,
            size_z=detection.size_z,
            view_x=detection.view_x,
            view_y=detection.view_y,
            view_yaw=detection.view_yaw,
            image_path=hero_path if hero_path != existing.hero_image_path else None,
            thumbnail_path=thumb_path if thumb_path != existing.thumbnail_path else None,
        )
        return updated_object, observation

    def _serialize_pose(self, pose: RobotPose | None) -> dict[str, Any] | None:
        if pose is None:
            return None
        return {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw}

    def _serialize_map(self) -> dict[str, Any] | None:
        if self.map_state is None:
            return None
        return {
            "map_id": self.map_state.map_id,
            "resolution": self.map_state.resolution,
            "origin_x": self.map_state.origin_x,
            "origin_y": self.map_state.origin_y,
            "width": self.map_state.width,
            "height": self.map_state.height,
            "updated_at": self.map_state.updated_at,
            "image_version": self.map_state.image_version,
            "image_url": f"/api/map/active/preview.png?v={self.map_state.image_version}",
        }

    def _serialize_ui_locked(self) -> dict[str, Any]:
        return {
            "revision": self.ui_state.revision,
            "camera": {
                "center_x": self.ui_state.camera.center_x,
                "center_y": self.ui_state.camera.center_y,
                "zoom": self.ui_state.camera.zoom,
            },
            "selected_item": self._serialize_item_ref(self.ui_state.selected_item),
            "highlighted_items": [
                self._serialize_item_ref(item_ref) for item_ref in self.ui_state.highlighted_items
            ],
        }

    def _serialize_inspection_settings_locked(self) -> dict[str, Any]:
        return {
            "manual_mode": self.inspection_settings.manual_mode,
        }

    def _serialize_yolo_runtime_locked(self) -> dict[str, Any]:
        return {
            "mode": self.yolo_runtime.mode,
            "inference_enabled": self.yolo_runtime.inference_enabled,
        }

    def _serialize_layers_locked(self) -> dict[str, Any]:
        return {
            "show_pois": self.layer_visibility.show_pois,
            "show_yolo": self.layer_visibility.show_yolo,
        }

    def _serialize_chat_locked(self) -> dict[str, Any]:
        return {
            "running": self.chat_state.running,
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "status": message.status,
                    "tools_used": list(message.tools_used or []),
                }
                for message in self.chat_state.messages
            ],
        }

    def _persist_chat_state_locked(self) -> None:
        self.storage.save_json_setting("chat_state", self._serialize_chat_locked())

    def _new_chat_message(self, *, role: str, content: str, status: str = "final") -> ChatMessage:
        return ChatMessage(
            message_id=f"chat-{time.time_ns()}",
            role=role,
            content=content,
            created_at=utc_now_iso(),
            status=status,
            tools_used=[],
        )

    def _require_chat_message_locked(self, message_id: str) -> ChatMessage:
        for message in self.chat_state.messages:
            if message.message_id == message_id:
                return message
        raise RuntimeError(f"Unknown chat message: {message_id}")

    def _serialize_item_ref(self, item_ref: SemanticItemRef | None) -> dict[str, str] | None:
        if item_ref is None:
            return None
        return {"kind": item_ref.kind, "entity_id": item_ref.entity_id}

    def _serialize_poi(self, poi: PoiRecord) -> dict[str, Any]:
        return {
            "poi_id": poi.poi_id,
            "map_id": poi.map_id,
            "anchor_x": poi.anchor_x,
            "anchor_y": poi.anchor_y,
            "anchor_yaw": poi.anchor_yaw,
            "target_x": poi.target_x,
            "target_y": poi.target_y,
            "title": poi.title,
            "summary": poi.summary,
            "category": poi.category,
            "interest_score": poi.interest_score,
            "status": poi.status,
            "objects": poi.objects,
            "created_at": poi.created_at,
            "updated_at": poi.updated_at,
            "thumbnail_url": f"/api/assets/{poi.thumbnail_path}",
            "hero_image_url": f"/api/assets/{poi.hero_image_path}",
        }

    def _serialize_yolo_object(self, object_record: YoloObjectRecord) -> dict[str, Any]:
        return {
            "object_id": object_record.object_id,
            "map_id": object_record.map_id,
            "label": object_record.label,
            "class_id": object_record.class_id,
            "world_x": object_record.world_x,
            "world_y": object_record.world_y,
            "world_z": object_record.world_z,
            "size_x": object_record.size_x,
            "size_y": object_record.size_y,
            "size_z": object_record.size_z,
            "best_view_x": object_record.best_view_x,
            "best_view_y": object_record.best_view_y,
            "best_view_yaw": object_record.best_view_yaw,
            "status": object_record.status,
            "detections_count": object_record.detections_count,
            "best_confidence": object_record.best_confidence,
            "created_at": object_record.created_at,
            "updated_at": object_record.updated_at,
            "last_seen_at": object_record.last_seen_at,
            "thumbnail_url": f"/api/assets/{object_record.thumbnail_path}",
            "hero_image_url": f"/api/assets/{object_record.hero_image_path}",
        }

    def _serialize_semantic_item(self, item_ref: SemanticItemRef) -> dict[str, Any]:
        if item_ref.kind == SEMANTIC_KIND_POI:
            poi = self._require_active_poi_locked(item_ref.entity_id)
            return {
                "kind": SEMANTIC_KIND_POI,
                "entity_id": poi.poi_id,
                "title": poi.title,
                "subtitle": poi.category,
                "world_x": poi.target_x,
                "world_y": poi.target_y,
                "world_yaw": poi.anchor_yaw,
                "anchor_x": poi.anchor_x,
                "anchor_y": poi.anchor_y,
                "anchor_yaw": poi.anchor_yaw,
                "target_x": poi.target_x,
                "target_y": poi.target_y,
                "thumbnail_url": f"/api/assets/{poi.thumbnail_path}",
                "updated_at": poi.updated_at,
            }
        object_record = self._require_active_yolo_object_locked(item_ref.entity_id)
        return {
            "kind": SEMANTIC_KIND_YOLO,
            "entity_id": object_record.object_id,
            "title": object_record.label.title(),
            "subtitle": f"{object_record.best_confidence:.2f}",
            "world_x": object_record.world_x,
            "world_y": object_record.world_y,
            "world_yaw": object_record.best_view_yaw,
            "thumbnail_url": f"/api/assets/{object_record.thumbnail_path}",
            "updated_at": object_record.updated_at,
        }


def contextlib_suppress(*exceptions: type[BaseException]) -> Any:
    from contextlib import suppress

    return suppress(*exceptions)


def create_app(
    *,
    map_socket_url: str = "http://localhost:7779",
    mcp_url: str = "http://localhost:9990/mcp",
    state_dir: Path | None = None,
    model_name: str = "gpt-5.4-mini",
    chat_model_name: str = "gpt-5.4",
    pov_poll_interval_seconds: float = DEFAULT_POV_POLL_INTERVAL_SECONDS,
    pov_max_width: int | None = DEFAULT_POV_MAX_WIDTH,
    pov_jpeg_quality: int = DEFAULT_POV_JPEG_QUALITY,
    service: SlamassService | None = None,
) -> FastAPI:
    _state_dir = state_dir or default_state_dir()
    slamass = service or SlamassService(
        map_socket_url=map_socket_url,
        mcp_url=mcp_url,
        state_dir=_state_dir,
        model_name=model_name,
        chat_model_name=chat_model_name,
        pov_poll_interval_seconds=pov_poll_interval_seconds,
        pov_max_width=pov_max_width,
        pov_jpeg_quality=pov_jpeg_quality,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        await slamass.start()
        try:
            yield
        finally:
            with contextlib_suppress(asyncio.CancelledError):
                await slamass.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.slamass = slamass

    _cors_origins = os.getenv("DIMOS_SLAMASS_CORS_ORIGINS", "").strip()
    if _cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return await slamass.snapshot()

    @app.get("/api/inspection-settings")
    async def get_inspection_settings() -> dict[str, Any]:
        async with slamass._state_lock:
            return slamass._serialize_inspection_settings_locked()

    @app.get("/api/ui")
    async def get_ui_state() -> dict[str, Any]:
        return await slamass.ui_snapshot()

    @app.get("/api/chat")
    async def get_chat_state() -> dict[str, Any]:
        return await slamass.chat_snapshot()

    @app.get("/api/chat/tools")
    async def get_chat_tools() -> list[dict[str, Any]]:
        return await slamass.chat_tools_manifest()

    @app.get("/api/events")
    async def get_events() -> EventSourceResponse:
        queue = await slamass.subscribe()

        async def event_generator() -> Any:
            try:
                while True:
                    event = await queue.get()
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"]),
                    }
            except asyncio.CancelledError:
                return
            finally:
                slamass.unsubscribe(queue)

        return EventSourceResponse(event_generator())

    @app.get("/api/pov/latest.jpg")
    async def get_latest_pov() -> Response:
        body = slamass.latest_pov_jpeg or make_placeholder_jpeg()
        return Response(
            content=body,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/pov/snapshot/ingest")
    async def post_pov_snapshot_ingest(file: UploadFile = File(...)) -> dict[str, Any]:
        body = await file.read()
        return await slamass.ingest_user_pov_snapshot(body)

    @app.get("/api/map/active")
    async def get_active_map() -> dict[str, Any]:
        payload = slamass._serialize_map()
        if payload is None:
            raise HTTPException(status_code=404, detail="No active map available")
        return payload

    @app.get("/api/map/active/preview.png")
    async def get_active_map_preview() -> Response:
        if slamass.map_state is None:
            raise HTTPException(status_code=404, detail="No active map available")
        preview_path = slamass.storage.asset_path("maps/active_map.png")
        if preview_path.exists():
            return FileResponse(preview_path, media_type="image/png")
        return Response(content=slamass.map_state.preview_png_bytes(), media_type="image/png")

    @app.post("/api/map/save")
    async def post_save_map() -> dict[str, Any]:
        return await slamass.save_map()

    @app.post("/api/memory/clear-map")
    async def post_clear_low_level_map_memory() -> dict[str, Any]:
        return await slamass.clear_low_level_map_memory()

    @app.post("/api/memory/clear-semantic")
    async def post_clear_semantic_memory() -> dict[str, Any]:
        return await slamass.clear_semantic_memory()

    @app.put("/api/ui/camera")
    async def put_ui_camera(request: UiCameraRequest) -> dict[str, Any]:
        return await slamass.set_ui_camera(request.center_x, request.center_y, request.zoom)

    @app.post("/api/ui/select-item")
    async def post_ui_select_item(request: UiSelectionRequest) -> dict[str, Any]:
        return await slamass.select_item(request.kind, request.entity_id)

    @app.post("/api/ui/highlight-items")
    async def post_ui_highlight_items(request: UiHighlightRequest) -> dict[str, Any]:
        return await slamass.highlight_items(
            [
                SemanticItemRef(kind=item["kind"], entity_id=item["entity_id"])
                for item in request.items
            ],
            selected_item=(
                SemanticItemRef(
                    kind=request.selected_item["kind"],
                    entity_id=request.selected_item["entity_id"],
                )
                if request.selected_item is not None
                else None
            ),
        )

    @app.post("/api/ui/clear-focus")
    async def post_ui_clear_focus() -> dict[str, Any]:
        return await slamass.clear_ui_focus()

    @app.post("/api/ui/focus-item/{kind}/{entity_id}")
    async def post_ui_focus_item(kind: str, entity_id: str, request: UiFocusRequest) -> dict[str, Any]:
        return await slamass.focus_item(kind, entity_id, zoom=request.zoom)

    @app.post("/api/ui/focus-robot")
    async def post_ui_focus_robot(request: UiFocusRequest) -> dict[str, Any]:
        return await slamass.focus_robot(zoom=request.zoom)

    @app.post("/api/ui/focus-map")
    async def post_ui_focus_map() -> dict[str, Any]:
        return await slamass.focus_map()

    @app.post("/api/navigate")
    async def post_navigate(request: NavigateRequest) -> dict[str, Any]:
        await slamass.navigate(request.x, request.y, request.yaw)
        return {"ok": True}

    @app.post("/api/teleop/command")
    async def post_teleop_command(request: TeleopCommandRequest) -> dict[str, Any]:
        return await slamass.send_move_command(
            linear_x=request.linear_x,
            linear_y=request.linear_y,
            linear_z=request.linear_z,
            angular_x=request.angular_x,
            angular_y=request.angular_y,
            angular_z=request.angular_z,
        )

    @app.post("/api/teleop/stop")
    async def post_teleop_stop() -> dict[str, Any]:
        return await slamass.stop_motion()

    @app.post("/api/explore/start")
    async def post_explore_start() -> dict[str, Any]:
        try:
            text = await slamass.mcp_client.call_tool_text("begin_exploration")
        except Exception as exc:  # noqa: BLE001 - surface to client
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "result": text}

    @app.post("/api/explore/stop")
    async def post_explore_stop() -> dict[str, Any]:
        try:
            text = await slamass.mcp_client.call_tool_text("end_exploration")
        except Exception as exc:  # noqa: BLE001 - surface to client
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "result": text}

    @app.post("/api/obstacle-avoidance")
    async def post_obstacle_avoidance(
        request: ObstacleAvoidanceRequest,
    ) -> dict[str, Any]:
        try:
            text = await slamass.mcp_client.call_tool_text(
                "set_obstacle_avoidance", {"enabled": request.enabled}
            )
        except Exception as exc:  # noqa: BLE001 - surface to client
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "enabled": request.enabled, "result": text}

    @app.post("/api/searchlight")
    async def post_searchlight(request: SearchlightRequest) -> dict[str, Any]:
        try:
            text = await slamass.mcp_client.call_tool_text(
                "set_searchlight", {"enabled": request.enabled}
            )
        except Exception as exc:  # noqa: BLE001 - surface to client
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "enabled": request.enabled, "result": text}

    @app.post("/api/system/stop")
    async def post_system_stop() -> dict[str, Any]:
        return await slamass.stop_dimos(force=False)

    @app.post("/api/chat")
    async def post_chat_message(request: ChatSubmitRequest) -> dict[str, Any]:
        return await slamass.submit_chat_message(request.message)

    @app.post("/api/chat/reset")
    async def post_chat_reset() -> dict[str, Any]:
        return await slamass.reset_chat()

    @app.post("/api/inspect/now")
    async def post_inspect_now() -> dict[str, Any]:
        return await slamass.inspect_now()

    @app.put("/api/inspection-settings")
    async def put_inspection_settings(request: InspectionSettingsRequest) -> dict[str, Any]:
        return await slamass.set_manual_inspection_mode(request.manual_mode)

    @app.put("/api/yolo/runtime")
    async def put_yolo_runtime(request: YoloRuntimeRequest) -> dict[str, Any]:
        return await slamass.set_yolo_runtime(
            mode=request.mode,
            inference_enabled=request.inference_enabled,
        )

    @app.put("/api/layers")
    async def put_layers(request: LayerVisibilityRequest) -> dict[str, Any]:
        return await slamass.set_layer_visibility(
            show_pois=request.show_pois,
            show_yolo=request.show_yolo,
        )

    @app.get("/api/pois")
    async def get_pois() -> list[dict[str, Any]]:
        return [slamass._serialize_poi(poi) for poi in slamass._active_pois()]

    @app.get("/api/pois/{poi_id}")
    async def get_poi(poi_id: str) -> dict[str, Any]:
        poi = slamass.pois.get(poi_id)
        if poi is None or poi.status == "deleted":
            raise HTTPException(status_code=404, detail="POI not found")
        return slamass._serialize_poi(poi)

    @app.post("/api/pois/{poi_id}/go")
    async def post_go_to_poi(poi_id: str) -> dict[str, Any]:
        await slamass.go_to_poi(poi_id)
        return {"ok": True}

    @app.post("/api/pois/{poi_id}/delete")
    async def post_delete_poi(poi_id: str) -> dict[str, Any]:
        await slamass.delete_poi(poi_id)
        return {"ok": True}

    @app.get("/api/yolo-objects")
    async def get_yolo_objects() -> list[dict[str, Any]]:
        return [slamass._serialize_yolo_object(obj) for obj in slamass._active_yolo_objects()]

    @app.get("/api/yolo-objects/{object_id}")
    async def get_yolo_object(object_id: str) -> dict[str, Any]:
        object_record = slamass.yolo_objects.get(object_id)
        if object_record is None or object_record.status == "deleted":
            raise HTTPException(status_code=404, detail="YOLO object not found")
        return slamass._serialize_yolo_object(object_record)

    @app.post("/api/yolo-objects/{object_id}/go")
    async def post_go_to_yolo_object(object_id: str) -> dict[str, Any]:
        await slamass.go_to_yolo_object(object_id)
        return {"ok": True}

    @app.post("/api/yolo-objects/{object_id}/delete")
    async def post_delete_yolo_object(object_id: str) -> dict[str, Any]:
        await slamass.delete_yolo_object(object_id)
        return {"ok": True}

    @app.get("/api/semantic-items")
    async def get_semantic_items() -> list[dict[str, Any]]:
        items = [
            slamass._serialize_semantic_item(
                SemanticItemRef(kind=SEMANTIC_KIND_POI, entity_id=poi.poi_id)
            )
            for poi in slamass._active_pois()
        ]
        items.extend(
            slamass._serialize_semantic_item(
                SemanticItemRef(kind=SEMANTIC_KIND_YOLO, entity_id=obj.object_id)
            )
            for obj in slamass._active_yolo_objects()
        )
        return items

    @app.get("/api/assets/{relative_path:path}")
    async def get_asset(relative_path: str) -> Response:
        return FileResponse(slamass.resolve_asset(relative_path))

    dist_dir = Path(__file__).resolve().parents[1] / "web" / "slamass-app" / "dist"
    dist_assets = dist_dir / "assets"
    if dist_assets.exists():
        app.mount("/assets", StaticFiles(directory=str(dist_assets)), name="slamass_assets")

    @app.get("/{full_path:path}")
    async def serve_app(full_path: str) -> Response:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if dist_dir.exists():
            requested = (dist_dir / full_path).resolve()
            if requested.exists() and requested.is_file() and dist_dir.resolve() in requested.parents:
                return FileResponse(requested)
            index_path = dist_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
        return Response(
            content=(
                "SLAMASS UI is not built yet. Run: "
                "cd dimos/web/slamass-app && npm install && npm run build"
            ),
            status_code=503,
            media_type="text/plain",
        )

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SLAMASS sidecar service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7780)
    parser.add_argument("--map-socket-url", default="http://localhost:7779")
    parser.add_argument("--mcp-url", default="http://localhost:9990/mcp")
    parser.add_argument("--pov-poll-interval", type=float, default=DEFAULT_POV_POLL_INTERVAL_SECONDS)
    parser.add_argument("--pov-max-width", type=int, default=DEFAULT_POV_MAX_WIDTH)
    parser.add_argument("--pov-jpeg-quality", type=int, default=DEFAULT_POV_JPEG_QUALITY)
    parser.add_argument("--state-dir", type=Path, default=default_state_dir())
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--chat-model", default="gpt-5.4")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    uvicorn.run(
        create_app(
            map_socket_url=args.map_socket_url,
            mcp_url=args.mcp_url,
            pov_poll_interval_seconds=args.pov_poll_interval,
            pov_max_width=args.pov_max_width,
            pov_jpeg_quality=args.pov_jpeg_quality,
            state_dir=args.state_dir,
            model_name=args.model,
            chat_model_name=args.chat_model,
        ),
        host=args.host,
        port=args.port,
    )


__all__ = ["SlamassService", "create_app", "main"]
