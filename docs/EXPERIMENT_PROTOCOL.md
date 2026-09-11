# Locked Experimental Protocol

This protocol is frozen before generating reportable results. Deviations must be recorded in the run manifest and may not be silently merged with the primary result.

## Research questions

- **RQ1:** Does the full sandwich reduce real H.264/H.265 bitrate at equal action-recognition accuracy relative to the unprocessed anchor?
- **RQ2:** Does the postprocessor improve perceptual quality without erasing the machine-rate gain of the preprocessor?
- **RQ3:** Which gains come from DINOv2, adaptive DCT, QP-FiLM, Video Swin, and real-forward/proxy-backward training?

## Data and splits

Primary task: Kinetics-400 action recognition with the repository's frozen analyzer. Use the official train/validation identities where available. If only one root is supplied, the code creates a deterministic stratified split controlled by `--seed`; persist the indices and do not resample after seeing results.

Decode 16 RGB frames per example with temporal stride 2 and resize/crop to 128×128 for the primary experiment. Preserve original sample identifiers in the per-video results. Report dataset version, download source, exclusions, decode failures, and final sample counts.

GOT-10k tracking is a future extension and must not appear in the main empirical claims until a tracking adapter, dataset loader, and metric evaluator are implemented and tested.

## Codec operating points

- Codecs: H.264/AVC (`libx264`) and H.265/HEVC (`libx265`) for scientific evaluation.
- QPs: 30, 32, 35, 37, 40, 42, 45 for the final curves.
- Constant frame rate: record exact FPS; default 30.
- Record FFmpeg build, codec library versions, pixel format, preset, GOP/key-frame settings, threads, and all emitted encoder arguments.
- Compute BPP from the elementary-stream byte count divided by `T×H×W`; never use proxy BPP in a final result.

## Training gates

1. Build deterministic real-codec caches independently for training and validation.
2. Train one proxy per codec configuration family.
3. Require finite losses, frozen proxy parameters, reconstruction correlation, per-QP BPP rank/error checks, a positive success rate for a proxy-rate descent step reducing real BPP, and mean proxy hard-clamp fraction no greater than the predeclared 5% gate.
4. Only after the proxy audit passes, train the pre/post wrappers.
5. Select checkpoints by validation task BD-rate when defined. Before the first valid task BD-rate only, use the best validation loss as an explicitly recorded fallback. Once any valid BD-rate exists, an undefined value may not replace it. Persist the selection metric, value, epoch, and fallback status in every checkpoint.
6. Never tune hyperparameters on the held-out test set.

All trainable wrappers start as identity maps. Use three seeds for reportable means and dispersion. Persist `args`, seed, git commit, checkpoint hashes, environment versions, and validation metrics in each run directory.

## Methods and ablations

Required methods:

| ID | Pre | Post | Real forward | Proxy backward | DINO | adaptive DCT | QP-FiLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 anchor | no | no | yes | no | no | no | no |
| A1 pre-only | yes | identity | yes | yes | yes | yes | yes |
| A2 full sandwich | yes | yes | yes | yes | yes | yes | yes |
| B1 proxy-forward control | yes | yes | no | yes | yes | yes | yes |
| B2 no DINO | yes | yes | yes | yes | no | yes | yes |
| B3 no DCT | yes | yes | yes | yes | yes | no | yes |
| B4 no QP-FiLM | yes | yes | yes | yes | yes | yes | no |
| B5 CNN preprocessor | CNN | yes | yes | yes | yes | yes | yes |
| B6 no post task path | yes | eval-only | yes | yes | yes | yes | yes |

Train B1 only as a controlled test of the forward mismatch identified by Lu et al.; final deployment always uses the real codec.

## Metrics

Machine: Top-1 and Top-5 accuracy at every QP; task BD-rate with Top-1 as quality. Human/reconstruction: PSNR, a validated external MS-SSIM implementation, LPIPS, and VMAF where the toolchain is available. The primary PSNR operating point is `-10 log10(mean per-video MSE)`; per-video PSNR remains in the CSV for audit but may not be averaged into the primary curve. Rate: elementary-stream BPP. Complexity: parameters, model size, and MACs where supported.

BD-rate is valid only over an overlapping quality interval with enough distinct points. Report `undefined` instead of extrapolating. The primary noisy-task curve uses the documented monotone Pareto envelope on both methods. Preserve every raw point and report a raw-curve sensitivity value; if the raw curve is not strictly monotone, report that sensitivity as `undefined`. Negative BD-rate denotes bitrate saving. Use paired bootstrap resampling by video (10,000 samples, fixed seed) for 95% intervals on aggregate metric differences and task BD-rate; if a resample has an invalid BD-rate curve, report valid count, invalid count, and valid fraction.

## Statistical reporting and stopping rule

Primary endpoint: H.264 task BD-rate of A2 vs A0 on the held-out validation/test set. Secondary endpoints: H.265 task BD-rate and perceptual BD-rates.

Report all seeds and both codecs. The method is considered supported only if the primary point estimate is negative, the paired 95% interval is reported, and task accuracy does not collapse at the lowest-rate point. A null or unfavorable result remains part of the manuscript; do not replace it with prior-work values.

## Result artifact contract

Each final run directory must contain:

```text
args.json / checkpoint metadata
checkpoint_metadata.json
git_commit.txt
git_state.json
environment.txt
per_video_metrics.csv
summary.json
bd_rate.json
bootstrap.json
codec_commands.json
checkpoint_sha256.txt
```

Only artifacts satisfying this contract may populate the manuscript's Results section.
