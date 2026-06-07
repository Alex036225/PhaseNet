# PhaseNet Deployment

This directory contains a standalone inference entrypoint for a trained
PhaseNet checkpoint. It runs outside the training/test data-loader workflow and
expects one preprocessed clip saved as `.npy` or `.npz`.

It also contains a sliding-window streaming entrypoint for video files or camera
sources. The streaming mode is near-real-time over a rolling context window; the
current PhaseNet model is not strictly causal.

## Input

`infer_phasenet.py` accepts these layouts:

- `T,H,W,3`
- `T,3,H,W`
- `3,T,H,W`
- `N,T,H,W,3`
- `N,T,3,H,W`
- `N,3,T,H,W`

Use `--normalize-255` if the clip is stored as raw `0..255` RGB values and the
checkpoint was trained on `0..1` inputs.

## Run

```bash
cd /public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_current

python deploy/infer_phasenet.py \
  --config configs/PhaseNet-UUU.yaml \
  --checkpoint /path/to/phasenet_checkpoint.pth \
  --input /path/to/clip.npy \
  --output outputs/pred_rppg.npy \
  --json-output outputs/pred_rppg.json
```

For `.npz` files, pass `--npz-key` when the desired array is not the first array
or the key is not `data`.

## Streaming

```bash
cd /public_hw/home/cit_yingxinlai/project/PhysLLM/PhaseNet_current

python deploy/stream_infer_phasenet.py \
  --config configs/PhaseNet-UUU.yaml \
  --checkpoint /path/to/phasenet_checkpoint.pth \
  --source /path/to/video.mp4 \
  --output outputs/stream_predictions.csv \
  --json-output outputs/stream_summary.json \
  --window-size 128 \
  --stride 1 \
  --normalize-255
```

Use `--source 0` for camera index 0. The CSV is appended line-by-line while the
stream runs, with one prediction emitted per inference window.

## Checkpoint

The loader accepts a plain PyTorch `state_dict` or checkpoints containing one of
these keys:

- `state_dict`
- `model_state_dict`
- `model`
- `net`

It also strips common `module.` and `model.` prefixes from DDP-trained
checkpoints.
