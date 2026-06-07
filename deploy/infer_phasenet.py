"""Standalone PhaseNet inference entrypoint.

The script loads a PhaseNet checkpoint and runs inference on a preprocessed
video clip saved as .npy or .npz. It is intentionally independent from the
training/test data-loader path so it can be used in deployment jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config
from neural_methods.model.PhaseNet.PhaseNet import PhaseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhaseNet on one preprocessed clip.")
    parser.add_argument("--config", default="configs/PhaseNet-UUU.yaml", help="PhaseNet yaml config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained PhaseNet checkpoint.")
    parser.add_argument("--input", required=True, help="Input .npy or .npz clip.")
    parser.add_argument("--output", required=True, help="Output .npy path for the predicted rPPG signal.")
    parser.add_argument("--json-output", default="", help="Optional JSON summary output path.")
    parser.add_argument("--npz-key", default="", help="Array key to read when --input is .npz.")
    parser.add_argument("--device", default="", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--normalize-255", action="store_true", help="Divide input values by 255 before inference.")
    return parser.parse_args()


def load_config(config_path: str):
    return get_config(SimpleNamespace(config_file=config_path))


def build_model(config) -> PhaseNet:
    params = config.MODEL.PHASENET.PARAMS
    return PhaseNet(
        feature_dim=params.FEATURE_DIM,
        latent_dim=params.LATENT_DIM,
        hidden_dim=params.HIDDEN_DIM,
    )


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a state_dict-like key.")

    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value
    return state_dict


def load_clip(path: Path, npz_key: str = "") -> np.ndarray:
    if path.suffix == ".npy":
        clip = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = npz_key or ("data" if "data" in data.files else data.files[0])
        clip = data[key]
    else:
        raise ValueError("Input must be .npy or .npz")
    return np.asarray(clip)


def to_ncthw(clip: np.ndarray) -> torch.Tensor:
    """Accept common clip layouts and return float tensor shaped [N, C, T, H, W]."""
    if clip.ndim == 4:
        # THWC, TCHW, or CTHW
        if clip.shape[-1] == 3:
            clip = np.transpose(clip, (3, 0, 1, 2))
        elif clip.shape[1] == 3:
            clip = np.transpose(clip, (1, 0, 2, 3))
        elif clip.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Cannot infer channel dimension for 4D input shape {clip.shape}")
        clip = clip[None, ...]
    elif clip.ndim == 5:
        # NTHWC, NTCHW, or NCTHW
        if clip.shape[-1] == 3:
            clip = np.transpose(clip, (0, 4, 1, 2, 3))
        elif clip.shape[2] == 3:
            clip = np.transpose(clip, (0, 2, 1, 3, 4))
        elif clip.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Cannot infer channel dimension for 5D input shape {clip.shape}")
    else:
        raise ValueError(f"Expected 4D or 5D clip, got shape {clip.shape}")

    return torch.from_numpy(np.ascontiguousarray(clip)).float()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = args.device or (config.DEVICE if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("Missing keys:", missing[:20])
        if unexpected:
            print("Unexpected keys:", unexpected[:20])

    clip = to_ncthw(load_clip(Path(args.input), args.npz_key))
    if args.normalize_255:
        clip = clip / 255.0

    model.to(device)
    model.eval()
    with torch.no_grad():
        pred, _ = model(clip.to(device))

    pred_np = pred.detach().cpu().numpy()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pred_np)

    summary = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "output": str(output_path),
        "prediction_shape": list(pred_np.shape),
        "prediction_mean": float(np.mean(pred_np)),
        "prediction_std": float(np.std(pred_np)),
    }
    print(json.dumps(summary, indent=2))

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
