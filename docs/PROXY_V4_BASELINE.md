# Proxy V4: Video Swin Lite with a predictive entropy codec surrogate

> Archived inheritance note: this is the original V4 design documentation copied into V9 for provenance. Paths and commands refer to the repository root; the active V9 workflow is documented in `../README.md`.

Task-aware video preprocessing with the requested pipeline kept intact:

```text
video -> Video Swin Lite -> H.264/H.265 -> reconstruction
      -> frozen analyzer -> task
```

During training, a frozen differentiable proxy runs beside the real codec:

```text
                            +-> frozen codec proxy -- backward gradients --+
                            |                                               |
video -> trainable preprocessor -> real H.264/H.265 -> reconstruction ------+
                                                    -> frozen analyzer -> task
```

The real FFmpeg codec always determines reconstruction and measured BPP in the
forward pass. The proxy supplies only the backward Jacobian:

```python
reconstruction = proxy_reconstruction + (
    real_reconstruction - proxy_reconstruction
).detach()

bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
```

Consequently, task and rate-distortion loss values correspond to the standard
codec while gradients still reach only the preprocessor. Codec proxy and analyzer
parameters remain frozen.

## Predictive entropy proxy

V4 replaces the unconstrained scalar rate head with a coding-structured surrogate:

```text
RGB clip + per-sample QP
  -> I-frame signal + previous-frame prediction residual
  -> QP-FiLM 3-D analysis transform
  -> spatial /8 latent
  -> QP-scaled STE quantization
  -> factorized Laplace likelihood -> entropy bits -> calibrated BPP
  -> 3-D synthesis transform without unquantized skips
  -> integrate decoded temporal residuals -> proxy reconstruction
```

The positive QP-conditioned affine calibration can match H.264/H.265 scale and
overhead but cannot reverse the entropy gradient. Paired examples supervise both
absolute BPP and within-clip BPP changes. A real-codec gradient probe checks whether
a small proxy-rate descent step also lowers measured BPP. The proxy remains frozen
during preprocessor training, while autograd differentiates reconstruction and rate
with respect to the VideoSwinLite output.

## Video Swin Lite preprocessor

`VideoSwinLitePreprocessor` is a compact dense video transformer:

```text
BTCHW RGB video + codec QP
  -> normalized QP embedding (MLP)
  -> Conv3D spatial patch embedding, patch=(1,4,4), 3 -> 48 channels
  -> depthwise Conv3D positional encoding
  -> four alternating regular/shifted 3-D Swin blocks
       QP FiLM: (1 + gamma(QP)) * feature + beta(QP)
       window=(4,8,8), heads=4, MLP ratio=4
  -> LayerNorm
  -> ConvTranspose3D spatial reconstruction, 48 -> RGB
  -> tanh * 0.25 * sigmoid(QP residual gate)
  -> input + RGB residual
```

Temporal resolution is never downsampled. Shifted windows exchange information
between neighboring clips and spatial regions while avoiding global space-time
attention. The RGB head is zero-initialized, so a new model is exactly the identity
mapping. The earlier factorized ViT and CNN remain available as `--preprocessor vit`
and `--preprocessor cnn` for ablation; `swin` is the default.
QP conditioning is enabled by default for new Swin checkpoints. It lets low QPs
suppress expensive texture without forcing high QPs to use the same residual.
Evaluation reconstructs the QP-specific preprocessed clip before every real-codec
operating point. Legacy checkpoints without QP parameters remain loadable.

## Objective

```text
L = alpha * (L_D + lambda * L_R) + L_Acc
```

- `L_D`: MSE between source video and real codec reconstruction.
- `L_R`: measured elementary-stream BPP from H.264/H.265.
- `L_Acc`: cross-entropy from the frozen Kinetics-400 analyzer.
- Defaults: `alpha=10`, `lambda=0.001`, Adam `lr=1e-4`.

Optional analyzer feature fidelity is available with `--feature-weight`, but its
default is zero. The main V4 recipe does not use feature loss because the preceding
V8 experiment improved accuracy while increasing real H.264 BPP at every QP.
See [V4_DESIGN_VI.md](V4_DESIGN_VI.md) for the gradient path, proxy losses, and
acceptance criteria before preprocessor training.

## Requirements

```bash
pip install -r requirements.txt
```

FFmpeg must include `libx264` and/or `libx265`. There is one unified requirements
file; no Kaggle-specific requirements file is needed.

## Training order

First build a deterministic codec cache. The raw pipe is checked against the
legacy PNG path before caching, two FFmpeg workers run concurrently, and each
source clip is stored only once as `uint8`. `train/` and `val/` have separate
cache trees; when the dataset has no `val/`, the split is stratified and fixed by
`--seed`.

```bash
python -u precompute_codec.py \
  --data-root /path/to/kinetics/train \
  --codec h264 \
  --qps 30 35 40 45 \
  --codec-io pipe \
  --codec-workers 2 \
  --output-dir precomputed_codec/h264
```

Then distill a codec-specific proxy without invoking FFmpeg in every epoch:

```bash
python -u train_proxy.py \
  --precomputed-root precomputed_codec/h264 \
  --codec h264 \
  --qps 30 35 40 45 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --max-delta 1.0 \
  --entropy-floor 1e-9 \
  --rate-delta-weight 0.5 \
  --rate-direction-weight 0.1 \
  --pair-strengths 0 0.05 0.1 0.2 \
  --preprocessor-checkpoint /path/to/v6_best_task_bd_rate.pt \
  --gradient-probe-batches 25 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --output-dir checkpoints/h264_proxy
```

Every cached training batch is balanced across the four QPs. Batch sizes 8 or
16 are recommended. Validation always evaluates the fixed cached split. The
legacy online path remains available by replacing `--precomputed-root` with
`--data-root`; it now uses raw pipes and two codec workers by default.

The proxy architecture changed, so a FiLM deeper-3D `last.pt` cannot be resumed.
Start predictive-entropy V4 training at epoch 1 with a new output directory. Existing
precomputed codec caches remain fully reusable because their real reconstruction
and BPP targets are architecture-independent. Paired variants and gradient probes
invoke real FFmpeg even when the base samples come from a cache.

Then train Video Swin Lite through the real codec and frozen proxy:

```bash
python -u train.py \
  --data-root /path/to/kinetics/train \
  --proxy-checkpoint checkpoints/h264_proxy/best_feasible.pt \
  --init-checkpoint /path/to/v6_best_task_bd_rate.pt \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --feature-weight 0 \
  --output-dir checkpoints/preprocessor
```

For the continued V6/V7 experiment, use
`kaggle_cells/v4_swin_rate_recovery.ipynb`. It accepts only a
`large_reaudit/best_feasible_reaudit.pt` predictive-entropy proxy, starts a new
optimizer and per-QP direct-rate controller from the retained V6 VideoSwin
checkpoint, targets a real BPP ratio of 0.95, and keeps feature/mask objectives
at zero. The forward values come from real H.264 while the frozen proxy supplies
the backward Jacobian. A feasible controller checkpoint is then evaluated on
the held-out full split at seven QPs with a paired bootstrap confidence interval.

If the five-epoch run remains infeasible without triggering the proxy guard,
`kaggle_cells/v4_swin_continue.ipynb` resumes epochs 6–10 with the saved
optimizer, scheduler, and per-QP controller state. Kaggle may relocate an Input
checkpoint between notebook versions; an unchanged frozen proxy is therefore
accepted by its recorded SHA-256 even when its path changes.

If `val/` is absent, training creates a deterministic stratified validation subset
using indices in memory. Classes containing only one video remain in training.

## Real-codec evaluation

Final evaluation never uses the proxy. It compares the anchor and preprocessed
clips through the real FFmpeg codec and frozen analyzer. If `val/` is absent,
`evaluate_real_codec.py` automatically recreates the checkpoint's stratified
validation split in memory from its saved `val_ratio` and `seed`; no validation
folder, symlinks or precomputed codec cache are required.

```bash
python -u evaluate_real_codec.py \
  --checkpoint checkpoints/preprocessor/best.pt \
  --data-root /path/to/kinetics/train \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir outputs/real_codec
```

The output includes `metrics.csv`, `metrics.json`, `bd_rate.json`, and one
`<codec>_top1_bpp_bd_rate.png` plot per codec. Task BD-rate uses Top-1 as the
quality axis; PSNR BD-rate is also reported. Negative BD-rate means bitrate
saving at equal quality. Task BD-rate is reported as undefined when discrete
Top-1 curves have too few distinct points or no overlapping accuracy range.

Omit `--limit` for the final result. `--limit 200` is useful for a faster pilot,
but produces noisier Top-1 and task BD-rate estimates. The limit applies only to
evaluation videos after the deterministic split and is independent of the proxy
precompute limits.

## Kaggle and model summary

Ready-to-run Kaggle cells are in [KAGGLE_GUIDE_VI.md](KAGGLE_GUIDE_VI.md).

```bash
python model_summary.py \
  --model all \
  --preprocessor swin \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Main files

- `preprocessing/swin.py`: Video Swin Lite and 3-D shifted-window attention.
- `preprocessing/model.py`: preprocessor factory plus factorized ViT/CNN ablations.
- `preprocessing/standard_codec.py`: FFmpeg codecs, proxy and gradient bridge.
- `precompute_codec.py`: deterministic train/val uint8 codec cache and pipe verification.
- `train_proxy.py`: distill the proxy from real codec outputs and measured BPP.
- `train.py`: train only the preprocessor with rate-distortion-task loss.
- `preprocessing/evaluation.py`: reproducible held-out split and BD-rate helpers.
- `model_summary.py`: torchinfo summaries for preprocessor and proxy.
- `evaluate_real_codec.py`: real-codec metrics, Top-1/BPP plots and BD-rate.
- `visualize_pipeline.py`: qualitative output from the same held-out split.
