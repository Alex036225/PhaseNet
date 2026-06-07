"""Sliding-window streaming inference for PhaseNet.

This entrypoint reads frames from a video file or camera, keeps a fixed-size
window, and emits rPPG predictions every N frames. It is near-real-time
streaming over a rolling context window; the PhaseNet architecture itself is not
strictly causal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.infer_phasenet import build_model, load_config, unwrap_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window PhaseNet streaming inference.")
    parser.add_argument("--config", default="configs/PhaseNet-UUU.yaml", help="PhaseNet yaml config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained PhaseNet checkpoint.")
    parser.add_argument("--source", required=True, help="Video path or camera index such as 0.")
    parser.add_argument("--output", required=True, help="CSV output path for streaming predictions.")
    parser.add_argument("--json-output", default="", help="Optional JSON summary output path.")
    parser.add_argument("--device", default="", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--window-size", type=int, default=128, help="Number of frames per inference window.")
    parser.add_argument("--stride", type=int, default=1, help="Run inference every N frames after warmup.")
    parser.add_argument("--height", type=int, default=128, help="Input resize height.")
    parser.add_argument("--width", type=int, default=128, help="Input resize width.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means no limit.")
    parser.add_argument("--normalize-255", action="store_true", help="Divide RGB values by 255 before inference.")
    parser.add_argument(
        "--emit",
        choices=("latest", "center", "mean"),
        default="latest",
        help="Which value to emit from each predicted window.",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    capture_source: Any = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def preprocess_frame(frame_bgr: np.ndarray, width: int, height: int, normalize_255: bool) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    frame = frame_rgb.astype(np.float32)
    if normalize_255:
        frame /= 255.0
    return frame


def window_to_tensor(window: deque[np.ndarray]) -> torch.Tensor:
    clip = np.stack(tuple(window), axis=0)
    clip = np.transpose(clip, (3, 0, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(clip)).float().unsqueeze(0)


def select_prediction(pred: np.ndarray, emit: str) -> float:
    values = pred.reshape(-1)
    if emit == "latest":
        return float(values[-1])
    if emit == "center":
        return float(values[len(values) // 2])
    return float(np.mean(values))


def load_model(args: argparse.Namespace, device: torch.device):
    config = load_config(args.config)
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(unwrap_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        print(f"Checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("Missing keys:", missing[:20])
        if unexpected:
            print("Unexpected keys:", unexpected[:20])
    model.to(device)
    model.eval()
    return model, config


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    config = load_config(args.config)
    device_name = args.device or (config.DEVICE if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, _ = load_model(args, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 0.0

    window: deque[np.ndarray] = deque(maxlen=args.window_size)
    rows: list[dict[str, Any]] = []
    frame_idx = -1
    inference_count = 0
    start_time = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "timestamp_sec", "prediction", "window_size", "emit"],
        )
        writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            window.append(preprocess_frame(frame, args.width, args.height, args.normalize_255))
            if len(window) < args.window_size:
                continue
            if (frame_idx - args.window_size + 1) % args.stride != 0:
                continue

            clip = window_to_tensor(window).to(device)
            with torch.no_grad():
                pred, _ = model(clip)
            value = select_prediction(pred.detach().cpu().numpy(), args.emit)
            timestamp = frame_idx / fps if fps > 0 else time.time() - start_time
            row = {
                "frame_idx": frame_idx,
                "timestamp_sec": f"{timestamp:.6f}",
                "prediction": f"{value:.8f}",
                "window_size": args.window_size,
                "emit": args.emit,
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            inference_count += 1
            print(json.dumps(row), flush=True)

    cap.release()

    summary = {
        "source": args.source,
        "checkpoint": args.checkpoint,
        "output": str(output_path),
        "frames_read": frame_idx + 1,
        "inference_count": inference_count,
        "window_size": args.window_size,
        "stride": args.stride,
        "emit": args.emit,
    }
    print(json.dumps(summary, indent=2))

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
