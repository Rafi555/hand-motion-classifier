from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

PROMPT = """You are classifying hand movement in a short video window.

You will receive a few sampled frames from one 2-second window of a video.
Focus only on visible hand motion. Ignore face, background, camera motion, and objects unless they directly affect the hand movement.

Return movement labels as short snake_case strings. Examples:
- swipe_left
- swipe_right
- move_up
- move_down
- reach_forward
- pull_back
- open_hand
- close_hand
- point
- tap
- hold_still
- rotate_wrist
- unknown

If the sampled frames are insufficient, use unknown.
Return only the requested JSON."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "movement_types": {
            "type": "array",
            "description": "Short snake_case labels describing the hand movement in this 2-second window.",
            "items": {"type": "string"},
        },
        "summary": {
            "type": "string",
            "description": "One short sentence explaining the classification.",
        },
    },
    "required": ["movement_types", "summary"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SampledFrame:
    timestamp_sec: float
    jpeg_bytes: bytes


@dataclass(frozen=True)
class WindowSamples:
    start_sec: float
    end_sec: float
    frames: list[SampledFrame]


@dataclass(frozen=True)
class WindowResult:
    start_sec: float
    end_sec: float
    movement_types: list[str]
    summary: str


def sample_video_frames(
    video_path: Path,
    window_seconds: float,
    frames_per_window: int,
    max_frames: int,
) -> list[WindowSamples]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata is invalid or unavailable.")

    duration_seconds = frame_count / fps
    window_count = max(1, math.ceil(duration_seconds / window_seconds))
    window_samples: list[WindowSamples] = []
    sampled_frame_count = 0

    for window_index in range(window_count):
        if sampled_frame_count >= max_frames:
            break

        start_sec = window_index * window_seconds
        end_sec = min(duration_seconds, start_sec + window_seconds)
        if end_sec <= start_sec:
            continue

        remaining_budget = max_frames - sampled_frame_count
        frames_this_window = min(frames_per_window, remaining_budget)
        offsets = [
            start_sec + (end_sec - start_sec) * (sample_index + 1) / (frames_this_window + 1)
            for sample_index in range(frames_this_window)
        ]

        frames: list[SampledFrame] = []
        for timestamp_sec in offsets:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames.append(SampledFrame(timestamp_sec=timestamp_sec, jpeg_bytes=encoded.tobytes()))
            sampled_frame_count += 1

        if frames:
            window_samples.append(WindowSamples(start_sec=start_sec, end_sec=end_sec, frames=frames))

    capture.release()
    if not window_samples:
        raise RuntimeError("No frames could be sampled from the video.")
    return window_samples


def classify_window(client: Any, model: str, window: WindowSamples) -> WindowResult:
    from google.genai import types

    contents: list = [
        PROMPT,
        f"Analyze only the {window.end_sec - window.start_sec:.2f}-second window from {window.start_sec:.2f}s to {window.end_sec:.2f}s.",
    ]
    for frame in window.frames:
        contents.append(f"Frame captured at {frame.timestamp_sec:.2f}s.")
        contents.append(types.Part.from_bytes(data=frame.jpeg_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
        ),
    )

    payload = json.loads(response.text)
    movement_types = sorted({label.strip() for label in payload["movement_types"] if label.strip()})
    if not movement_types:
        movement_types = ["unknown"]

    return WindowResult(
        start_sec=window.start_sec,
        end_sec=window.end_sec,
        movement_types=movement_types,
        summary=payload["summary"].strip(),
    )


app = FastAPI(title="Hand Motion Classifier")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/classify")
async def classify(
    video: UploadFile = File(...),
    window_seconds: float = Form(2.0),
    frames_per_window: int = Form(2),
    max_frames: int = Form(12),
    model: str = Form("gemini-2.5-flash"),
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on the server.")

    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = Path(tmp.name)

    try:
        windowed_frames = sample_video_frames(
            video_path=tmp_path,
            window_seconds=window_seconds,
            frames_per_window=frames_per_window,
            max_frames=max_frames,
        )

        from google import genai
        client = genai.Client(api_key=api_key)

        results = [
            classify_window(client=client, model=model, window=window)
            for window in windowed_frames
        ]

        all_labels = sorted({label for r in results for label in r.movement_types})
        windows_out = [
            {
                "start_sec": r.start_sec,
                "end_sec": r.end_sec,
                "movement_types": r.movement_types,
                "summary": r.summary,
            }
            for r in results
        ]

        return JSONResponse({"all_labels": all_labels, "windows": windows_out})

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
