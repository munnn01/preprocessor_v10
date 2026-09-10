# Các cell Kaggle chạy trực tiếp

## Tiếp tục từ V6/V7: train Proxy V4 có audit codec thật

Notebook sẵn dùng nằm tại
`kaggle_cells/v4_proxy_stage.ipynb`. Bật GPU và Internet, rồi gắn đúng ba Input:

1. `kineticscleaned`, có thư mục
   `/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/kinetics400_5per/kinetics400_5per/train`.
2. Output V6 có `swin_ratio_090/best_task_bd_rate.pt`.
3. Output V7 có `recovery_split.json` và `selection.json`.

Notebook tự so SHA-256 trong V7 để chọn đúng checkpoint V6, lấy cân bằng 1.000
video train và 400 video validation, rồi train Proxy V4 từ epoch 1. Proxy cũ và
checkpoint V8 không phải Input. VideoSwinLite V6 chỉ chạy frozen để tạo paired
samples on-policy; feature loss bằng 0.

Sau 5 epoch, notebook ghi `best_audit.pt` để chẩn đoán và chỉ ghi
`best_feasible.pt` khi mọi QP đạt đồng thời: rate MAPE không quá 20%, paired-rate
direction accuracy ít nhất 60%, bước đi theo gradient làm BPP H.264 thật giảm và
tỉ lệ mẫu giảm BPP thật ít nhất 55%. Chỉ `best_feasible.pt` được dùng ở pha
fine-tune VideoSwin tiếp theo.

Nếu audit đầu tiên chỉ trượt MAPE tại QP30/35 nhưng paired direction và real-codec
descent đều đạt, chạy `recalibrate_low_qp(stage)`. Pha này khởi tạo từ
`best_audit.pt` với optimizer mới, LR `1e-4`, rate weight `0.30`, lấy mẫu QP theo
tỉ lệ `3:2:1:1`, và audit lại cả bốn QP sau ba epoch.

## Sau khi large re-audit đạt: fine-tune VideoSwin và đánh giá 7 QP

Notebook `kaggle_cells/v4_swin_rate_recovery.ipynb` chạy toàn bộ pha tiếp theo.
Gắn năm Input:

1. `kineticscleaned` tại đường dẫn nêu trên.
2. Output V6 chứa `swin_ratio_090/best_task_bd_rate.pt`.
3. Output V7 chứa `recovery_split.json` và `selection.json`.
4. Output V4 vừa re-audit, chứa `large_reaudit/best_feasible_reaudit.pt`.
5. Output `checking` chứa `v5_fixed_split/split_manifest.json` để khôi phục đúng
   `validation_full` cho đánh giá cuối.

Helper kiểm tra SHA-256 của split/checkpoint, khôi phục đủ 1.999 video train và
800 video controller, rồi tạo run mới từ checkpoint V6 bằng `--init-checkpoint`.
Optimizer và rate controller được làm mới. Mỗi QP có dual weight riêng; target BPP
ratio là `0.95`, LR `1e-5`, năm epoch. H.264 thật quyết định pixels và BPP ở
forward, Proxy V4 đã audit cung cấp gradient về VideoSwin. Feature loss và mask
loss đều bằng 0 trong phép thử này.

Chỉ `best_feasible.pt` được chuyển sang đánh giá toàn bộ `validation_full` tại
QP `30 32 35 37 40 42 45`. Kết quả cuối dùng PCHIP Task BD-rate và paired-video
bootstrap 2.000 lần; `goal_check.json` ghi rõ point estimate và cận trên 95% CI
có dưới −10% hay không.

Nếu run năm epoch kết thúc `feasible=False` nhưng
`rate_dual_proxy_guard_abort=False`, dùng
`kaggle_cells/v4_swin_continue.ipynb`. Notebook này tìm đúng
`v4_swin_rate_recovery_*/swin_ratio_095/last.pt`, sao chép checkpoint sang một
output ghi được, khôi phục optimizer, scheduler và per-QP rate-dual state, rồi
chạy epoch 6–10 với cùng objective và cùng proxy SHA-256. Một đường dẫn proxy mới
trên Kaggle được chấp nhận khi SHA-256 vẫn giống checkpoint đã ghi nhận.

Pha tiếp tục không cần Input V6 vì `last.pt` đã chứa trọng số VideoSwin. Nó cần
Kinetics cleaned, V7 recovery split, proxy `best_feasible_reaudit.pt`, Output của
run năm epoch, và `checking` cho full validation. Proxy cũng được sao chép vào
Output mới để các lần tiếp tục sau tự chứa đủ trọng số.

Bật GPU và Internet cho Kaggle Notebook, sửa `DATA` nếu dataset được gắn ở đường
dẫn khác, sau đó chọn **Run All**.

## Cell 1 — Clone và cài đặt

```python
%cd /kaggle/working
!git clone -q https://github.com/munnn01/proxy_v4.git
%cd /kaggle/working/proxy_v4
%pip install -q --no-cache-dir -r requirements.txt
```

## Cell 2 — Đường dẫn

```python
DATA = "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
PROJECT = "/kaggle/working/proxy_v4"
PROXY_DIR = "/kaggle/working/checkpoints/h264_proxy_v4"
CACHE_DIR = "/kaggle/working/precomputed_codec/h264"
MODEL_DIR = "/kaggle/working/checkpoints/video_swin_qp_lambda_k6_2000_400"
EVAL_DIR = "/kaggle/working/real_codec_eval"
VIS_DIR = "/kaggle/working/visualization"
```

`DATA` ở đây trỏ trực tiếp tới thư mục chứa các thư mục lớp. Không cần có `val/`;
hai script train tự tạo validation split phân tầng trong bộ nhớ.

## Cell 3 — Model summary

```python
!python "$PROJECT/model_summary.py" \
  --model all \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --qp 35 \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Cell 4 — Pre-compute codec xác định, tách train/val

Cell này chỉ chạy FFmpeg một lần cho mỗi clip/QP. Clip gốc được lưu đúng một bản
`uint8`; reconstruction của bốn QP nằm riêng trong `train/` và `val/`. Raw pipe
được so pixel/BPP với đường PNG cũ trước khi cache và dùng đúng 2 FFmpeg workers.

```python
!python -u "$PROJECT/precompute_codec.py" \
  --data-root "$DATA" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --codec-io pipe \
  --codec-workers 2 \
  --ffmpeg-threads 1 \
  --verify-pipe \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --val-ratio 0.1 \
  --seed 42 \
  --output-dir "$CACHE_DIR"
```

Cache `uint8` gồm một clip gốc và bốn reconstruction nên có thể lớn hơn dữ liệu
video nén. Khi quota `/kaggle/working` không đủ, thêm `--limit-train N` và
`--limit-val M`, hoặc dùng một Kinetics subset nhỏ hơn.

## Cell 5 — Distill H.264 proxy từ cache

```python
!python -u "$PROJECT/train_proxy.py" \
  --precomputed-root "$CACHE_DIR" \
  --codec h264 \
  --qps 30 35 40 45 \
  --fps 30 \
  --preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --workers 4 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --amp \
  --output-dir "$PROXY_DIR"
```

`--batch-size 8` tạo mỗi batch gồm cân bằng cả bốn QP; có thể tăng lên 16 nếu GPU
đủ VRAM. Checkpoint lưu cả optimizer, AMP scaler và scheduler để resume đúng.
Proxy shallow cũ không tương thích với kiến trúc này, vì vậy phải train từ epoch 1
với `PROXY_DIR` mới. Cache codec đã tạo trước đây vẫn dùng lại được.

## Cell 6 — Train Video Swin Lite preprocessor

```python
!python -u "$PROJECT/train.py" \
  --data-root "$DATA" \
  --proxy-checkpoint "$PROXY_DIR/best_feasible.pt" \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --max-residual 0.10 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --codec-fps 30 \
  --codec-preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --limit-train 2000 \
  --limit-val 400 \
  --epochs 10 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --workers 4 \
  --optimizer adamw \
  --lr 0.0001 \
  --alpha 10 \
  --rate-lambda 0.048 0.151 0.386 0.576 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --amp \
  --output-dir "$MODEL_DIR"
```

Video Swin nhận QP đang dùng và FiLM-modulate từng block, đồng thời điều khiển
cường độ residual theo QP. Vì kiến trúc preprocessor thay đổi, hãy dùng một
`MODEL_DIR` mới và train từ epoch 1. FiLM deeper-3D proxy cùng cache codec cũ vẫn
dùng lại được. Một giá trị `--rate-lambda` sẽ được dùng chung cho mọi QP; bốn
giá trị sẽ ánh xạ lần lượt theo thứ tự của `--codec-qps`. Validation chạy toàn bộ
400 clip ở cả bốn QP để chọn checkpoint ổn định hơn, tương đương 1.600 lượt codec
thật mỗi epoch.

## Cell 7 — Đánh giá codec thật

```python
!python -u "$PROJECT/evaluate_real_codec.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --data-root "$DATA" \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir "$EVAL_DIR"
```

Khi không có `val/`, script tự đọc `val_ratio` và `seed` trong checkpoint để tái
tạo đúng validation phân tầng trong bộ nhớ; không cần tạo `EVAL_DATA`, symlink hay
dùng cache precompute. Mặc định script đánh giá toàn bộ validation và ghi:

- `metrics.csv`, `metrics.json`: BPP, MSE, PSNR, Top-1 và Top-5 theo từng QP.
- `bd_rate.json`: Task BD-rate theo Top-1 và BD-rate chuẩn theo PSNR.
- `h264_top1_bpp_bd_rate.png`: biểu đồ QP-BPP, BPP-Top-1 và BPP-PSNR.

Chỉ thêm `--limit 200` khi cần chạy thử nhanh. Kết quả báo cáo chính thức nên bỏ
`--limit`; giới hạn eval độc lập với `--limit-train/--limit-val` của precompute.

## Cell 8 — Trực quan hóa Top-1

```python
!python -u "$PROJECT/visualize_pipeline.py" \
  --checkpoint "$MODEL_DIR/best.pt" \
  --data-root "$DATA" \
  --sample-index 0 \
  --codec h264 \
  --codec-qp 35 \
  --device cuda \
  --output-dir "$VIS_DIR"
```

File JSON và tiêu đề ảnh/video chỉ hiển thị dự đoán và accuracy Top-1, không tạo
danh sách Top-5.

## Cell 9 — Nén kết quả

```python
%cd /kaggle/working
!zip -qr proxy_v4_results.zip checkpoints real_codec_eval visualization
```

## Cell 10 — Link tải xuống

```python
from IPython.display import FileLink

FileLink("/kaggle/working/proxy_v4_results.zip")
```
