# Video preprocessing for machine vision under standard compression

This repository keeps the requested end-to-end path unchanged:

```text
RGB video [B,T,3,H,W]
        |
        v
ViT preprocessor (trainable)
        |
        +---------------------> frozen differentiable codec proxy
        |                                      |
        v                                      | backward only
FFmpeg H.264/H.265 (standard codec)             |
        |                                      |
        v                                      |
real reconstruction + measured BPP <-----------+
        |
        v
frozen Kinetics-400 analyzer -> task logits
```

The analyzer and codec proxy are frozen while the preprocessor is trained. The
standard codec always determines the forward reconstruction and BPP. Because an
FFmpeg round trip has no gradient, the parallel proxy supplies only its Jacobian:

```python
reconstruction = proxy_reconstruction + (
    real_reconstruction - proxy_reconstruction
).detach()
```

Thus the loss value and analyzer input are exactly those from H.264/H.265, but
backpropagation reaches the preprocessor through the distilled proxy. The same bridge is
used for measured BPP and proxy-estimated BPP.

## Preprocessor

`VideoTransformerPreprocessor` follows a video Vision Transformer design:

- non-overlapping RGB patch embedding;
- spatial self-attention among patches inside each frame;
- temporal self-attention for the same patch location across frames;
- resolution-independent convolutional positional encoding;
- a bounded RGB residual head with exact identity initialization.

Factorized attention avoids the quadratic memory cost of attending over every
space-time token at once. The earlier two-branch CNN remains available with
`--preprocessor cnn`; ViT is the default.

## Objective

```text
L = alpha * (L_D + lambda * L_R) + L_Acc
```

- `L_D`: MSE between the original video and the real decoded video.
- `L_R`: measured elementary-stream bits per pixel from FFmpeg.
- `L_Acc`: cross-entropy from a frozen pretrained Kinetics-400 analyzer.
- Defaults: `alpha=10`, `lambda=0.001`, Adam `lr=1e-4`.

Only preprocessor parameters are passed to the optimizer.

## Requirements

Python 3.10+, PyTorch, torchvision, OpenCV, NumPy, tqdm, matplotlib, and an FFmpeg build
with `libx264` and/or `libx265` are required. `torchinfo` is used by the model-summary
command.

```bash
pip install -r requirements.txt
ffmpeg -hide_banner -encoders
```

## Kaggle và model summary

Hướng dẫn Kaggle từng bước, gồm smoke test, train proxy, train ViT, resume và đánh giá,
nằm tại [KAGGLE_GUIDE_VI.md](KAGGLE_GUIDE_VI.md).

Lệnh in summary cho cả ViT preprocessor và codec proxy:

```bash
python model_summary.py \
  --model all \
  --frames 16 \
  --frame-size 128 \
  --device cuda
```

Chỉ in preprocessor hoặc proxy bằng `--model preprocessor` và `--model proxy`. Có thể
nạp proxy đã distill bằng `--proxy-checkpoint /path/to/best.pt`.

The training data must contain real Kinetics-400 videos organized by class:

```text
kinetics400/
  train/
    abseiling/video_001.mp4
    air_drumming/video_002.mp4
  val/
    abseiling/video_101.mp4
```

When `val/` is absent, the scripts create a deterministic stratified split in memory.

## 1. Distill the codec proxy

The proxy must be trained for the same codec family, QPs, FPS, preset, frame count, and
resolution used by preprocessor training. A random proxy is intentionally not accepted.

```bash
python -u train_proxy.py \
  --data-root /path/to/kinetics400 \
  --codec h264 \
  --qps 30 35 40 45 50 \
  --frames 16 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 2 \
  --output-dir checkpoints/h264_proxy
```

This stage fits reconstruction with L1 loss and measured BPP with Smooth L1 loss. It
writes `best.pt` and `last.pt`; each checkpoint records proxy architecture and codec
settings.

## 2. Train the ViT preprocessor

```bash
python -u train.py \
  --data-root /path/to/kinetics400 \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --preprocessor vit \
  --vit-patch-size 8 \
  --vit-embed-dim 96 \
  --vit-depth 4 \
  --vit-heads 4 \
  --codec h264 \
  --codec-qps 30 35 40 45 50 \
  --epochs 30 \
  --frames 16 \
  --frame-size 128 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --output-dir checkpoints/preprocessor
```

Every epoch samples one QP. Validation disables the backward bridge and uses only the
real codec result. FFmpeg is CPU/process-bound, so a small batch and multiple data-loader
workers are usually preferable.

Run a short end-to-end check before a long job:

```bash
python -u train.py \
  --data-root /path/to/kinetics400 \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --codec h264 \
  --codec-qps 35 \
  --batch-size 1 \
  --workers 0 \
  --smoke-test \
  --output-dir checkpoints/smoke
```

## Evaluation and visualization

Evaluate anchor and learned preprocessing with the same real codec path:

```bash
python -u evaluate_real_codec.py \
  --checkpoint checkpoints/preprocessor/best.pt \
  --data-root /path/to/kinetics400 \
  --codecs h264 h265 \
  --qps 30 35 40 45 50 \
  --output-dir outputs/real_codec
```

Visualize one validation clip:

```bash
python -u visualize_pipeline.py \
  --checkpoint checkpoints/preprocessor/best.pt \
  --data-root /path/to/kinetics400 \
  --codec h264 \
  --codec-qp 35 \
  --output-dir outputs/visualization
```

## Main files

- `preprocessing/model.py`: ViT and legacy CNN preprocessors.
- `preprocessing/standard_codec.py`: FFmpeg codec, proxy, and forward/backward bridge.
- `model_summary.py`: model summary cho ViT preprocessor và codec proxy.
- `KAGGLE_GUIDE_VI.md`: hướng dẫn chạy đầy đủ trên Kaggle.
- `train_proxy.py`: distillation of the proxy from real H.264/H.265 outputs.
- `train.py`: rate-distortion-accuracy training of only the preprocessor.
- `evaluate_real_codec.py`: real-codec rate/accuracy comparison.
- `visualize_pipeline.py`: qualitative reconstruction and analyzer outputs.

The project implements the action-recognition branch. A tracking analyzer would require
its own frozen model and task-specific loss.
