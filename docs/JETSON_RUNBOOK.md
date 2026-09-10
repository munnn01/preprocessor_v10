# NVIDIA Jetson Orin NX Deployment Runbook

This runbook deliberately avoids fixed performance claims. NVIDIA software, power modes, and TensorRT operator support vary by JetPack release and module configuration; record what is actually installed on the target.

## 1. Capture the environment

```bash
uname -a
cat /etc/nv_tegra_release
cat /etc/nvpower/nvpmodel.conf | head
sudo nvpmodel -q
python -c "import torch; print(torch.__version__, torch.version.cuda)"
/usr/src/tensorrt/bin/trtexec --version
ffmpeg -version
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvv4l2h265enc
gst-inspect-1.0 nvv4l2decoder
```

Save the complete outputs with the experimental artifacts. Query the installed GStreamer element properties rather than assuming a QP/preset property name from another JetPack release.

## 2. Select and record the power mode

```bash
sudo nvpmodel -q
sudo nvpmodel -m <MODE_ID>
sudo jetson_clocks --show
```

Run `jetson_clocks` only if the selected protocol requires fixed maximum clocks, then disclose it. Allow the module to reach a stable thermal state before benchmarking. Repeat measurements if thermal throttling occurs.

## 3. Export the neural wrappers

On the training host or Jetson:

```bash
python export_jetson.py \
  --checkpoint checkpoints/adaptive_sandwich/best.pt \
  --frames 16 --height 128 --width 128 \
  --output-dir artifacts/onnx
```

The codec is intentionally absent from both ONNX graphs. The deployment topology is:

```text
preprocessor.engine -> nvv4l2h264enc/nvv4l2h265enc
                    -> nvv4l2decoder -> postprocessor.engine
```

## 4. Build TensorRT engines

Use the `trtexec` shipped with the recorded TensorRT release. Start with fixed shapes for the lowest deployment risk:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/preprocessor.onnx \
  --saveEngine=artifacts/preprocessor_fp16.engine \
  --fp16 --shapes=video:1x16x3x128x128,qp:1

/usr/src/tensorrt/bin/trtexec \
  --onnx=artifacts/onnx/postprocessor.onnx \
  --saveEngine=artifacts/postprocessor_fp16.engine \
  --fp16 --shapes=video:1x16x3x128x128,qp:1
```

If the installed parser rejects an operator, preserve the full log and TensorRT version. Do not silently replace the architecture for the reported experiment.

## 5. Validate numerical agreement

For a fixed set of clips/QPs, compare PyTorch and TensorRT output elementwise before measuring speed. Record max absolute error, mean absolute error, and downstream task delta. Set acceptance tolerances before looking at final rate/accuracy results.

## 6. Benchmark and power logging

First benchmark the PyTorch reference wrappers:

```bash
tegrastats --interval 100 --logfile outputs/tegrastats.log &
TEGRASTATS_PID=$!
python benchmark_jetson.py \
  --checkpoint checkpoints/adaptive_sandwich/best.pt \
  --warmup 30 --iterations 200 \
  --output outputs/jetson_benchmark.json
kill "$TEGRASTATS_PID"
python tools/parse_tegrastats.py \
  --input outputs/tegrastats.log \
  --output outputs/tegrastats_summary.json
```

Then repeat for TensorRT and the full hardware-codec pipeline. The shell PID snippet is Linux-only and intentionally not executed by this repository.

## 7. Reporting checklist

- Exact module memory size and carrier board.
- JetPack/L4T, CUDA, cuDNN, TensorRT, PyTorch, FFmpeg/GStreamer versions.
- Power-mode ID/name, `jetson_clocks` state, temperature range.
- Codec element, profile, level, rate-control, QP/bitrate, GOP, FPS, resolution, pixel format.
- FP32/FP16/INT8; calibration set/hash for INT8.
- Batch/clip shape, warm-up, iteration count.
- Pre, encode, decode, post, analyzer and end-to-end p50/p95 latency.
- FPS, peak memory, VDD_IN power and energy/frame.
- Checkpoint, ONNX, engine and source-commit hashes.

## 8. Primary vendor references

- NVIDIA Jetson accelerated GStreamer guide: <https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html>
- TensorRT ONNX deployment quick start: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-onnx-deployment.html>
- NVIDIA `tegrastats` utility documentation: <https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html>
