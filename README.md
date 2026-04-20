# PhaseNet

PhaseNet is a deep learning framework for video-based physiological signal estimation. The model combines a spatiotemporal visual encoder, temporal interlacing for lightweight motion-aware feature mixing, spatial attention, gated temporal convolution, and latent reconstruction regularization to recover pulse-related temporal dynamics from face video clips.

This release provides the complete training and testing code path for PhaseNet, including configuration, data loading, optimization, and evaluation. The runtime organization follows the rPPG-Toolbox ecosystem for dataset handling, experiment control, and metric reporting.

## Features

- Spatiotemporal visual encoding with temporal interlacing
- Spatial attention for frame-wise region weighting
- Gated temporal convolution for sequence modeling
- Reconstruction-based latent regularization during training
- Single-source, multi-source, and test-only execution modes
- Built-in evaluation pipeline for rPPG prediction metrics

## Repository Layout

```text
PhaseNet/
├── main.py
├── config.py
├── configs/
│   └── PhaseNet-UUU.yaml
├── dataset/
│   └── data_loader/
├── evaluation/
├── neural_methods/
│   ├── loss/
│   ├── model/PhaseNet/
│   └── trainer/
└── unsupervised_methods/
```

## Core Components

- `neural_methods/model/PhaseNet/PhaseNet.py`: PhaseNet model definition
- `neural_methods/trainer/PhaseNetTrainer.py`: training and testing loop
- `main.py`: project entry for training and evaluation
- `config.py`: configuration parser and runtime path expansion
- `configs/PhaseNet-UUU.yaml`: provided training / testing configuration template
- `dataset/data_loader/`: dataset interfaces for `UBFC-rPPG`, `MMPD`, and `Zhuhai`
- `evaluation/`: test-time metric computation and post-processing

## Supported Modes

- `train_and_test`
- `multi_train_and_test`
- `only_test`

## Training

Training:

```bash
python main.py --config_file configs/PhaseNet-UUU.yaml
```

Multi-GPU training:

```bash
torchrun --nproc_per_node=2 main.py --config_file configs/PhaseNet-UUU.yaml
```

## Testing

Set `TOOLBOX_MODE: "only_test"` in `configs/PhaseNet-UUU.yaml`, fill `INFERENCE.MODEL_PATH`, and run:

```bash
python main.py --config_file configs/PhaseNet-UUU.yaml
```

## Configuration

- Update `TRAIN.DATA.DATA_PATH`, `VALID.DATA.DATA_PATH`, and `TEST.DATA.DATA_PATH` to your dataset paths.
- Update `TRAIN.DATA.CACHED_PATH`, `VALID.DATA.CACHED_PATH`, and `TEST.DATA.CACHED_PATH` if you store preprocessed data elsewhere.
- Set `MODEL.NAME: PhaseNet`.
- Set `INFERENCE.MODEL_PATH` before using `only_test`.
- Output logs are written under `LOG.PATH`.
- The provided `PhaseNet-UUU.yaml` file is the current example config and should be updated to match local dataset and checkpoint paths.

## Acknowledgement

This codebase builds on the design of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), which provides the underlying framework conventions for dataset processing, training, inference, and evaluation in remote physiological sensing. We thank the rPPG-Toolbox authors and open-source contributors for making that infrastructure publicly available.

## Citation

If you use this repository in research, please cite both the PhaseNet paper and the original rPPG-Toolbox work.

```latex
@article{zhao2025phase,
  title={PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement},
  author={Zhao, Bo and Guo, Dan and Cao, Junzhe and Xu, Yong and Zou, Bochao and Tan, Tao and Sun, Yue and Yu, Zitong},
  journal={arXiv preprint arXiv:2509.24850},
  year={2025}
}

@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```
