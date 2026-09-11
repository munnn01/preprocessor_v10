# Kế hoạch sửa và kiểm định V10

Tài liệu này là checklist vận hành. Không được chạy chiến dịch nhiều seed trước khi các gate theo thứ tự dưới đây đều đạt.

## 1. Tính đúng của huấn luyện

- `best.pt` phải tuân theo `--checkpoint-metric`.
- Với `task_bd_rate`, validation loss chỉ là fallback trước BD-rate hợp lệ đầu tiên.
- Sau khi có BD-rate hợp lệ, epoch `undefined` không được ghi đè `best.pt`.
- Luôn giữ riêng `best_loss.pt`, `best_ce.pt`, `best_top1.pt`, và `best_task_bd_rate.pt`.
- Checkpoint V10 phải chứa optimizer, AMP scaler, toàn bộ RNG, QP RNG, global step, proxy SHA-256, cấu hình codec và `selection_state`.
- Resume phải khóa dữ liệu, kiến trúc, codec, proxy, objective, optimizer, checkpoint metric và seed. Muốn thay đổi phải dùng `--init-checkpoint` trong run mới.
- Parser sandwich chỉ được công bố option thực sự được triển khai.

Gate: test checkpoint/resume/optimizer pass; run liên tục và run tách resume cho cùng chuỗi QP và kết quả tương đương.

## 2. Tính đúng của đánh giá

- CSV phải giữ `mse`, `psnr_db`, byte bitstream, pixel count và sample ID cho từng video/method/QP.
- Summary, BD-rate và bootstrap phải dùng cùng `--psnr-aggregation`.
- API BD-rate nhận tên anchor/proposed rõ ràng; không đổi tên row ngầm.
- Paired bootstrap dùng cùng một draw video cho mọi QP và cả hai method.
- Mặc định 10.000 draw, seed cố định; ghi valid/invalid count và valid fraction.
- Lưu cả raw RD points và điểm sau monotone Pareto envelope.
- Raw sensitivity chỉ được tính khi quality tăng nghiêm ngặt theo rate; nếu không phải `undefined`.

Gate: point estimate trong `bootstrap.json` bằng giá trị trong `summary.json`; chạy lại cùng seed cho JSON giống nhau; matrix thiếu video/method/QP phải fail-fast.

## 3. Artifact và provenance

Mỗi evaluation phải có:

```text
args.json
checkpoint_metadata.json
git_commit.txt
git_state.json
environment.txt
checkpoint_sha256.txt
per_video_metrics.csv
summary.json
bd_rate.json
bootstrap.json
codec_commands.json
```

Run chỉ được đánh dấu reportable khi không dùng `--limit`, checkpoint format V10, ít nhất 10.000 bootstrap draw và Git sạch trước khi tạo output. Artifact Jetson được bổ sung sau trên đúng checkpoint hash.

## 4. Proxy gate

Theo từng QP, kiểm tra:

- reconstruction fidelity;
- BPP MAPE và rank/correlation;
- pair-direction accuracy;
- real BPP delta sau bước theo hướng giảm proxy rate;
- real-down fraction;
- proxy hard-clamp fraction.

Gate mặc định của clamp là 5%. `train_sandwich.py` lưu `last.pt` nhưng không lựa chọn best checkpoint nếu mean clamp của epoch vượt gate. `--allow-unaudited-proxy` chỉ dành cho diagnostic và tự động làm run không reportable.

Nếu clamp fail, thử theo thứ tự: giảm residual range, thêm pre-clamp range penalty, đánh giá lại temporal cumulative decoder, rồi mới cân nhắc surrogate/straight-through clamp. Không đổi gradient estimator sau khi đã xem primary result.

## 5. Precompute

Được cache:

- source clip và deterministic split;
- real-codec supervision cho proxy;
- anchor bitstream/reconstruction/metrics;
- source-teacher features khi transform hoàn toàn cố định.

Không được xem là cache hợp lệ cho training sandwich:

- output của preprocessor đang học;
- FFmpeg bitstream/decoded output sinh từ output đó;
- output postprocessor;
- proxy output sau preprocessor.

Anchor cache key phải chứa codec, QP, FPS, preset, FFmpeg path/thread setting, frame sampling, resolution, analyzer, validation size, split ratio và seed. Cache V9 thiếu contract này phải được tạo lại.

## 6. Compute pilot

Trước full run, chạy smoke test và pilot nhỏ. Ghi train/validation seconds per batch, epoch time, ETA, codec workers và FFmpeg threads. Dùng `--gradient-audit-interval 50` trong pilot để đo gradient norm đã nhân weight của rate/task/DINO/human.

Theo dõi đồng thời:

- real BPP và proxy BPP;
- proxy clamp;
- boundary fraction của pre/post output;
- gradient norm từng nhóm objective;
- Top-1/Top-5/CE theo QP;
- validation BD-rate có hợp lệ hay không.

Dừng nếu real BPP tăng có hệ thống trong khi proxy BPP giảm, clamp vượt gate, gradient rate bị lấn nhiều bậc độ lớn, hoặc residual liên tục chạm biên.

## 7. Thứ tự thí nghiệm

1. Unit/integration tests.
2. Proxy audit trên validation độc lập.
3. Identity baseline.
4. Rate-only pilot.
5. Rate + task.
6. Thêm DINO.
7. Thêm human objective.
8. A0/A1/A2 một seed trên subset.
9. A0/A1/A2 một seed trên validation đầy đủ.
10. A0/A1/A2 ba seed.
11. B1-B6 một seed; chỉ mở rộng seed khi cần.
12. Khóa code/config/checkpoint policy.
13. Held-out test đúng một lần.
14. Bootstrap 10.000 draw.
15. Jetson latency, memory và power trên cùng checkpoint hash.

## 8. Lệnh kiểm định tối thiểu

```bash
python -m compileall preprocessing train.py train_proxy.py train_sandwich.py evaluate_sandwich.py
python -m pytest -q
python train_sandwich.py --help
python evaluate_sandwich.py --help
```

Một run bị giới hạn dữ liệu, proxy-only, smoke-test, proxy chưa audit, Git dirty hoặc bootstrap dưới 10.000 mẫu phải được ghi là diagnostic/non-reportable.
