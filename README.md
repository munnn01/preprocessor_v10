# Adaptive Video Preprocessing Techniques for Optimizing Video Coding for Machines (VCM) on NVIDIA Jetson Orin NX

Research implementation for a standards-compatible video codec sandwich:

```text
RGB video + QP
    -> QP-FiLM Video Swin Lite preprocessor
    -> frozen H.264/H.265 codec
    -> QP-FiLM lightweight 3-D postprocessor
    -> frozen machine-vision analyzer
```

During training, the **forward pass is always the real FFmpeg codec and its measured bitstream BPP**. A frozen predictive-entropy proxy supplies only the backward Jacobian through the non-differentiable codec:

```python
decoded = proxy_decoded + (real_decoded - proxy_decoded).detach()
bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
```

This is the video/edge-device extension of the forward-real-codec strategy in Lu et al. The repository also combines joint neural pre/post wrappers from Sandwiched Compression, an RPP-inspired adaptive-DCT perceptual prior, frozen DINOv2 feature preservation, FiLM QP conditioning, the local virtual-video-codec design, and the earlier `proxy_v3`, `proxy_v4`, `film_deeper3d`, and `video_swin` implementations.

> **Scientific status (10 September 2026):** the method, training/evaluation code, tests, manuscript, and Jetson runbook are complete. No Kinetics checkpoint, held-out real-codec result, or Jetson Orin NX measurement was available in the workspace. The paper is therefore an explicitly marked **pre-results manuscript draft**, not a submission-ready empirical paper. Numbers reported by prior work are cited as prior-work results, never as results of this repository.

## What is implemented

- Trainable QP-conditioned Video Swin Lite preprocessor with an identity initialization.
- Trainable compact FiLM-3D residual postprocessor, also identity-initialized.
- Frozen H.264/H.265 forward path and frozen predictive-entropy codec proxy backward path.
- Composite VCM objective: real rate, supervised task loss, frozen DINOv2 semantic loss, Charbonnier, compact three-scale MS-SSIM, optional LPIPS, temporal-gradient consistency, and adaptive block-DCT sparsification.
- Real-codec evaluation of `anchor`, `pre_only`, and full `sandwich` at every codec/QP point, including task/PSNR/MS-SSIM BD-rate.
- Separate ONNX exports for pre/post deployment around Jetson hardware codecs.
- Reproducible PyTorch latency benchmark plus a parser for `tegrastats` power logs.
- Unit and integration tests for the real-forward/proxy-backward identity and gradient invariants.

The frozen analyzer and DINOv2 are teachers/evaluators; they are not included in the deployed pre/post TensorRT engines unless an application specifically requires on-device task inference.

## Install

Python 3.10+ and an FFmpeg build with `libx264` and/or `libx265` are required.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-research.txt
```

On Windows, use `.venv\Scripts\pip.exe`. LPIPS and ONNX are optional research/export dependencies. DINOv2 can be loaded from an already cloned official repository by passing `--dino-repo /path/to/dinov2`; otherwise PyTorch Hub needs network access on the first run.

## Reproducible workflow

### 1. Precompute real-codec supervision

```bash
python precompute_codec.py \
  --data-root /data/kinetics400/train \
  --codec h264 --qps 30 35 40 45 \
  --frames 16 --frame-stride 2 --frame-size 128 \
  --codec-io pipe --codec-workers 2 \
  --output-dir precomputed_codec/h264
```

### 2. Distill the predictive-entropy proxy

```bash
python train_proxy.py \
  --precomputed-root precomputed_codec/h264 \
  --codec h264 --qps 30 35 40 45 \
  --frames 16 --frame-stride 2 --frame-size 128 \
  --epochs 20 --batch-size 8 \
  --hidden-channels 48 --latent-channels 64 \
  --bottleneck-channels 96 --blocks-per-stage 2 \
  --film-channels 64 --rate-delta-weight 0.5 \
  --rate-direction-weight 0.1 --gradient-probe-batches 25 \
  --output-dir checkpoints/h264_proxy
```

Do not train the wrappers unless the proxy passes both reconstruction/rate audits and the real-codec gradient-direction probe described in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

### 3. Train the adaptive sandwich

```bash
python train_sandwich.py \
  --data-root /data/kinetics400/train \
  --proxy-checkpoint checkpoints/h264_proxy/best_feasible.pt \
  --preprocessor swin --postprocessor film3d \
  --codec h264 --codec-qps 30 35 40 45 \
  --frames 16 --frame-stride 2 --frame-size 128 \
  --epochs 30 --batch-size 1 --accumulation-steps 4 \
  --task-input postprocessed \
  --dino-model dinov2_vits14 \
  --sandwich-rate-weight 0.05 --sandwich-task-weight 1.0 \
  --dino-weight 0.25 --human-weight 1.0 \
  --output-dir checkpoints/adaptive_sandwich
```

Use `--postprocessor identity` for the pre-only ablation and `--dino-weight 0`, `--adaptive-dct-weight 0`, or `--lpips-weight 0` for controlled objective ablations. DINOv2 and the task analyzer remain frozen.

### 4. Evaluate only through real codecs

```bash
python evaluate_sandwich.py \
  --checkpoint checkpoints/adaptive_sandwich/best.pt \
  --data-root /data/kinetics400/train \
  --codecs h264 h265 \
  --qps 30 32 35 37 40 42 45 \
  --device cuda \
  --output-dir outputs/sandwich_real_codec
```

The command writes per-video CSV and aggregate JSON. Negative BD-rate means bitrate saving at equal quality/accuracy. Omit `--limit` for reportable experiments.

### 5. Export and benchmark on Jetson Orin NX

```bash
python export_jetson.py \
  --checkpoint checkpoints/adaptive_sandwich/best.pt \
  --output-dir artifacts/onnx

python benchmark_jetson.py \
  --checkpoint checkpoints/adaptive_sandwich/best.pt \
  --warmup 30 --iterations 200 \
  --output outputs/jetson_benchmark.json

python tools/parse_tegrastats.py \
  --input outputs/tegrastats.log \
  --output outputs/tegrastats_summary.json
```

See [`docs/JETSON_RUNBOOK.md`](docs/JETSON_RUNBOOK.md) for TensorRT conversion, hardware-codec checks, power-mode logging, and reporting requirements.

## Verification

```bash
python -m pytest -q
python -m compileall preprocessing train_sandwich.py evaluate_sandwich.py export_jetson.py benchmark_jetson.py
```

Current local verification: **89 tests passed**. Two PyTorch nested-tensor performance warnings are non-failures.

## Paper and research record

- [`paper/manuscript.pdf`](paper/manuscript.pdf): rendered pre-results paper.
- [`paper/manuscript.md`](paper/manuscript.md): editable manuscript source.
- [`paper/main.tex`](paper/main.tex): IEEE-style LaTeX source for eventual submission.
- [`paper/evidence/source_manifest.json`](paper/evidence/source_manifest.json): source provenance.
- [`paper/evidence/claims.csv`](paper/evidence/claims.csv): claim-to-source and result-status ledger.
- [`docs/DESIGN_V9_VI.md`](docs/DESIGN_V9_VI.md): detailed Vietnamese design rationale.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): locked experimental protocol.
- [`docs/PROXY_V4_BASELINE.md`](docs/PROXY_V4_BASELINE.md): inherited V4 documentation.

## Primary references and code provenance

- [Sandwiched Compression paper](https://arxiv.org/abs/2402.05887) and [official code](https://github.com/google/sandwiched_compression).
- [Lu et al., Preprocessing Enhanced Image Compression for Machine Vision](https://arxiv.org/abs/2206.05650) and [official code](https://github.com/XingtongGe/PreprocessingICM). The repository confirms the forward-real-codec/proxy-backward mechanism requested here.
- [Rate-Perception Optimized Preprocessing](https://arxiv.org/abs/2301.10455). No author-linked implementation was identified in the bounded GitHub/web search performed for this project; this repository implements only the paper-inspired adaptive-DCT idea, not a claimed reproduction.
- [DINOv2 paper](https://arxiv.org/abs/2304.07193) and [official code](https://github.com/facebookresearch/dinov2).
- [FiLM paper](https://arxiv.org/abs/1709.07871).
- Zhao et al., *A Preprocessing Framework for Video Machine Vision under Compression*, local supplied manuscript, arXiv:2512.15331.

## License and attribution

No upstream repository was copied wholesale. The implementation in this repository is derived from the user's local V3/V4/FiLM/Video-Swin code and reimplements the cited ideas in PyTorch. Review dependency and dataset licenses before redistribution or commercial deployment.
