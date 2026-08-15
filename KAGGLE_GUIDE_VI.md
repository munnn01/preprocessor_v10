# Hướng dẫn chạy trên Kaggle

## 1. Tạo Notebook

Trong phần thiết lập Kaggle Notebook:

- Accelerator: `GPU T4 x2`, `GPU P100` hoặc GPU đang có sẵn.
- Internet: bật trong lần đầu để clone mã nguồn và tải trọng số analyzer.
- Gắn dataset Kinetics-400 vào notebook.

Clone project và cài thư viện:

```python
!git clone https://github.com/munnn01/preprocessor_proxy.git /kaggle/working/preprocessor_proxy
%cd /kaggle/working/preprocessor_proxy
%pip install -q --no-cache-dir -r requirements-kaggle.txt
```

Khởi động lại kernel nếu Kaggle yêu cầu sau khi cài package. Kiểm tra môi trường:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
!ffmpeg -version | head -n 1
!ffmpeg -hide_banner -encoders 2>&1 | grep -E "libx264|libx265"
```

## 2. Khai báo đường dẫn dữ liệu

`DATA` phải là thư mục chứa trực tiếp `train/` và, nếu có, `val/`:

```python
DATA = "/kaggle/input/kinetics-train-5per/kinetics400_5per/kinetics400_5per"
```

Cấu trúc mong đợi:

```text
DATA/
  train/
    abseiling/*.mp4
    air_drumming/*.mp4
  val/                    # không bắt buộc
    abseiling/*.mp4
```

Nếu không có `val/`, chương trình tự tạo tập validation phân tầng trong bộ nhớ.

## 3. In Model Summary

In đồng thời ViT preprocessor và codec proxy mặc định:

```python
!python model_summary.py \
  --model all \
  --frames 16 \
  --frame-size 128 \
  --device cuda
```

Chỉ in ViT preprocessor với cấu hình dùng khi train:

```python
!python model_summary.py \
  --model preprocessor \
  --preprocessor vit \
  --vit-patch-size 8 \
  --vit-embed-dim 96 \
  --vit-depth 4 \
  --vit-heads 4 \
  --frames 16 \
  --frame-size 128 \
  --device cuda
```

In proxy từ checkpoint đã distill:

```python
!python model_summary.py \
  --model proxy \
  --proxy-checkpoint /kaggle/working/checkpoints/h264_proxy/best.pt \
  --qp 35 \
  --frames 16 \
  --frame-size 128 \
  --device cuda
```

Lệnh summary không chạy FFmpeg và không tải analyzer Kinetics-400.

## 4. Distill codec proxy

Chạy smoke test trước:

```python
!python -u train_proxy.py \
  --data-root "$DATA" \
  --codec h264 \
  --qps 35 \
  --frames 16 \
  --frame-size 128 \
  --batch-size 1 \
  --workers 2 \
  --smoke-test \
  --output-dir /kaggle/working/checkpoints/proxy_smoke
```

Sau khi smoke test thành công, train proxy đầy đủ:

```python
!python -u train_proxy.py \
  --data-root "$DATA" \
  --codec h264 \
  --qps 30 35 40 45 50 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 2 \
  --workers 4 \
  --amp \
  --output-dir /kaggle/working/checkpoints/h264_proxy
```

Giai đoạn này gọi FFmpeg cho từng video nên thường bị giới hạn bởi CPU. Nếu RAM hoặc
VRAM không đủ, giảm `--batch-size` xuống `1`; không đổi `frames`, `frame-stride` hoặc
`frame-size` giữa lúc train proxy và train preprocessor.

## 5. Train ViT preprocessor

Smoke test toàn pipeline:

```python
!python -u train.py \
  --data-root "$DATA" \
  --proxy-checkpoint /kaggle/working/checkpoints/h264_proxy/best.pt \
  --preprocessor vit \
  --codec h264 \
  --codec-qps 35 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --batch-size 1 \
  --workers 2 \
  --smoke-test \
  --output-dir /kaggle/working/checkpoints/preprocessor_smoke
```

Train đầy đủ:

```python
!python -u train.py \
  --data-root "$DATA" \
  --proxy-checkpoint /kaggle/working/checkpoints/h264_proxy/best.pt \
  --preprocessor vit \
  --vit-patch-size 8 \
  --vit-embed-dim 96 \
  --vit-depth 4 \
  --vit-heads 4 \
  --codec h264 \
  --codec-qps 30 35 40 45 50 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --workers 4 \
  --amp \
  --output-dir /kaggle/working/checkpoints/preprocessor
```

Resume khi Kaggle ngắt session:

```python
!python -u train.py \
  --data-root "$DATA" \
  --proxy-checkpoint /kaggle/working/checkpoints/h264_proxy/best.pt \
  --resume /kaggle/working/checkpoints/preprocessor/last.pt \
  --codec h264 \
  --codec-qps 30 35 40 45 50 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --output-dir /kaggle/working/checkpoints/preprocessor
```

Các tham số codec, QP, FPS, preset, số frame, stride và kích thước frame phải khớp với
checkpoint proxy. Chương trình sẽ dừng sớm nếu phát hiện cấu hình không khớp.

## 6. Đánh giá và tải kết quả

```python
!python -u evaluate_real_codec.py \
  --checkpoint /kaggle/working/checkpoints/preprocessor/best.pt \
  --data-root "$DATA" \
  --codecs h264 h265 \
  --qps 30 35 40 45 50 \
  --limit 10 \
  --output-dir /kaggle/working/real_codec_eval
```

```python
!python -u visualize_pipeline.py \
  --checkpoint /kaggle/working/checkpoints/preprocessor/best.pt \
  --data-root "$DATA" \
  --codec h264 \
  --codec-qp 35 \
  --output-dir /kaggle/working/visualization
```

Nén checkpoint và kết quả để tải về:

```python
!cd /kaggle/working && zip -qr preprocessor_proxy_results.zip \
  checkpoints real_codec_eval visualization
```
