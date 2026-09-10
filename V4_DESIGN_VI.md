# Thiết kế Proxy V4

V4 lấy nguyên lý từ *A Preprocessing Framework for Video Machine Vision under
Compression*: virtual codec phải biểu diễn prediction residual, lượng tử hóa và
entropy rate. Bài báo không công bố đủ chi tiết để sao chép codec; triển khai này
là một thiết kế có kiểm soát dựa trên nguyên lý đó, không phải code chính thức.

## Quyết định về feature loss

Feature loss không nằm trong cấu hình chính. Run V8 trước đó dùng feature loss
theo QP làm BPP H.264 tăng 7.17–9.07% và Task BD-rate chỉ còn -1.99%. V4 giữ tùy
chọn `--feature-weight` để làm ablation, mặc định `0`. Khi bật, clean feature được
detach, feature tái tạo vẫn giữ gradient qua analyzer và proxy về VideoSwinLite.

## Đường forward và backward

Khi train proxy, H.264/H.265 cung cấp reconstruction và BPP đích. Proxy học lại
hai giá trị và thứ tự thay đổi BPP của các biến thể cùng video.

Khi train VideoSwinLite, forward dùng H.264/H.265 thật. Bridge dùng:

```python
reconstruction = proxy_reconstruction + (real_reconstruction - proxy_reconstruction).detach()
bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
```

Vì vậy giá trị loss đến từ codec thật, còn Jacobian reconstruction/rate đến từ
proxy V4. Proxy và analyzer bị đóng băng; optimizer chỉ cập nhật VideoSwinLite.

## Virtual codec

```text
video
  -> frame đầu trừ 0.5; các frame sau trừ frame nguồn trước
  -> 3-D analysis transform có QP FiLM
  -> latent / q_step(QP)
  -> STE round
  -> factorized Laplace probability
  -> -log2(probability) / số pixel
  -> positive affine calibration theo QP
  -> proxy BPP

quantized latent
  -> 3-D synthesis transform, không có unquantized encoder skip
  -> decoded temporal residual
  -> cumulative integration
  -> proxy reconstruction
```

Affine calibration chỉ có gain dương và overhead dương. Nó có thể khớp thang BPP
và header của codec thật nhưng không thể đảo dấu entropy gradient.

## Loss huấn luyện proxy

```text
L_proxy = L1(reconstruction_proxy, reconstruction_real)
        + rate_weight * SmoothL1(log BPP_proxy, log BPP_real)
        + rate_delta_weight * L_delta
        + rate_direction_weight * L_direction
```

`L_delta` khớp thay đổi log-BPP giữa video cơ sở và biến thể. `L_direction` phạt
khi proxy và H.264 dự đoán hai hướng ngược nhau. Biến thể được tạo từ output của
checkpoint Swin rồi pha với spatial low-pass ở nhiều strength.

`--gradient-probe-batches` không cập nhật model. Nó thực hiện một bước pixel nhỏ
theo hướng giảm proxy rate rồi encode lại bằng codec thật. Không dùng proxy để
train Swin nếu `probe_real_down_fraction` thấp hoặc `probe_real_delta_percent`
dương ổn định. Không có một ngưỡng duy nhất được bài báo chứng minh; cần báo toàn
bộ kết quả theo QP và chọn ngưỡng trước khi chạy chính.

## Loss huấn luyện VideoSwinLite

Run V4 đầu tiên dùng:

```text
L = CE + alpha * (MSE + lambda[QP] * real_BPP)
feature_weight = 0
```

Forward `real_BPP` là H.264/H.265; backward của hạng rate là entropy gradient từ
proxy. Bắt đầu bằng `--init-checkpoint` từ checkpoint V6 để giữ trọng số Swin và
reset optimizer. Không resume checkpoint V8 feature-loss.

Checkpoint proxy cũ có architecture `film_deeper3d_v1` không tương thích với
`predictive_entropy_v1`; Proxy V4 phải train từ đầu. Codec cache cơ sở vẫn dùng
được, còn paired variants và gradient probes cần FFmpeg online.
