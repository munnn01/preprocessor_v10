# Thiết kế V9: Adaptive Video Preprocessing cho VCM trên Jetson Orin NX

## 1. Phạm vi và trạng thái khoa học

V9 là một thiết kế **pre/codec/post** tương thích codec chuẩn. Codec H.264/H.265 không bị sửa và không được cập nhật trọng số. Mục tiêu là giảm bitrate nhưng vẫn duy trì đồng thời độ chính xác cho máy và chất lượng cảm nhận cho người. Mã nguồn đã hoàn chỉnh ở mức framework; kết quả cuối cùng chỉ được công bố sau khi huấn luyện trên dữ liệu thật, đánh giá bằng bitstream codec thật, và đo trực tiếp trên Jetson Orin NX.

Không được dùng các con số −20.3%, −14.6%, >15%, 16.27%, hoặc khoảng 30% của các bài nguồn như kết quả của V9. Chúng chỉ là bằng chứng thiết kế/đối chứng từ prior work.

## 2. Ma trận thừa hưởng ý tưởng

| Nguồn | Ý tưởng được kế thừa | Hiện thực trong V9 | Điều không tuyên bố |
|---|---|---|---|
| Sandwiched Compression, arXiv:2402.05887 | Mạng pre/post bao quanh codec chuẩn; huấn luyện qua proxy; tối ưu metric ngoài thiết kế gốc của codec | `AdaptiveVideoSandwich`, `FiLM3DPostprocessor`, objective cảm nhận | Không sao chép mô hình TensorFlow/UNet; không nhận kết quả 30% là của V9 |
| Lu et al., arXiv:2206.05650 | Forward bằng codec thật, backward bằng proxy | `ParallelStandardVideoCodec`; hai biểu thức STE giá trị thật/Jacobian proxy | Không nhận −20.3% hoặc −14.6% là kết quả V9 |
| Zhao et al., *A Preprocessing Framework for Video Machine Vision under Compression* | Nhánh không-thời gian, virtual video codec, tối ưu rate–distortion–accuracy, đánh giá codec chuẩn | Video Swin Lite, predictive-entropy proxy có dự đoán frame trước, loss task, H.264/H.265 | Chưa hiện thực tracking GOT-10k trong V9; không nhận >15% là kết quả V9 |
| RPP, arXiv:2301.10455 | Adaptive DCT, bảo toàn thành phần tần số cao quan trọng, metric cảm nhận | `adaptive_dct_loss`, compact MS-SSIM, LPIPS tùy chọn | Đây là hiện thực lấy cảm hứng, không phải reproduction chính thức |
| DINOv2 | Đặc trưng thị giác tổng quát tự giám sát | teacher DINOv2 đóng băng; cosine loss trên CLS/patch token | DINOv2 không chạy trên Jetson trong đường pre/post mặc định |
| FiLM | Điều biến affine theo điều kiện | QP embedding sinh gamma/beta ở preprocessor, proxy và postprocessor | Không tuyên bố FiLM tự nó tối ưu bitrate |
| Video Swin | Attention theo cửa sổ không-thời gian, cửa sổ dịch chuyển | Video Swin Lite không giảm chiều thời gian, residual RGB identity-init | Không tuyên bố trùng kiến trúc Video Swin gốc |
| `proxy_v3`, `proxy_v4`, `film_deeper3d`, `video_swin` | Hạ tầng dữ liệu, analyzer đóng băng, codec cache, FiLM 3-D, Video Swin Lite, BD-rate | Giữ toàn bộ pipeline V4 và thêm sandwich/post/DINO/human loss/Jetson export | Các mục tiêu hoặc checkpoint cũ không tự động trở thành kết quả V9 |

## 3. Đường truyền và đường gradient

Với video nguồn `x`, QP `q`, preprocessor `P_theta`, codec chuẩn `C_q`, proxy đóng băng `C_hat`, và postprocessor `R_phi`:

```text
z       = P_theta(x, q)                 neural-code video
y_real  = C_q(z)                        reconstruction thật
r_real  = bits(C_q(z)) / số pixel       BPP thật
x_hat   = R_phi(y_real, q)              video phục hồi
```

Khi huấn luyện, giá trị forward và gradient được tách rõ:

```text
y_ST = y_proxy + stopgrad(y_real - y_proxy)
r_ST = r_proxy + stopgrad(r_real - r_proxy)
```

Do đó `value(y_ST) = y_real`, `value(r_ST) = r_real`, nhưng đạo hàm theo `z` đến từ proxy. Codec thật, proxy, task analyzer và DINOv2 đều đóng băng. Chỉ `theta` và `phi` được cập nhật.

## 4. Kiến trúc

### Preprocessor

`VideoSwinLitePreprocessor` nhận tensor `[B,T,3,H,W]`. Conv3D patch embedding chỉ giảm không gian; bốn Swin block xen kẽ regular/shifted window trao đổi thông tin không-thời gian. QP được chuẩn hóa rồi đi qua MLP FiLM. Đầu RGB zero-init tạo ánh xạ đồng nhất ở bước đầu, giảm nguy cơ làm hỏng anchor.

### Predictive-entropy proxy

Proxy tách I-frame signal và temporal prediction residual, dùng analysis/synthesis Conv3D có QP-FiLM, STE quantization và factorized Laplace likelihood để ước lượng entropy. Calibration dương theo QP khớp thang BPP codec thật nhưng không được phép đảo dấu gradient entropy. Proxy cần vượt audit trên dữ liệu codec-cache và real-codec gradient probe trước khi dùng.

### Postprocessor

`FiLM3DPostprocessor` là residual U-Net 3-D nhỏ gồm depthwise-separable Conv3D, một mức down/up không gian, và QP-FiLM. Không giảm chiều thời gian. Đầu RGB zero-init nên checkpoint mới cũng chính xác là identity. Kiến trúc này nhằm giảm chi phí khi xuất TensorRT, nhưng độ trễ thực tế phải đo trên Jetson.

## 5. Hàm mục tiêu đa tiêu chí

```text
L = w_r L_rate + w_t L_task + w_s L_DINO + w_h L_human
```

Trong đó:

- `L_rate = BPP_real / BPP_anchor(q)` để cân bằng các QP.
- `L_task` là cross-entropy từ analyzer video đóng băng.
- `L_DINO` là `1 - cosine` giữa đặc trưng CLS/patch của video phục hồi và video nguồn qua DINOv2 đóng băng.
- `L_human` gồm Charbonnier, compact three-scale MS-SSIM, LPIPS tùy chọn, sai khác temporal gradient, và adaptive-DCT trên neural code.

Adaptive-DCT chỉ kéo các hệ số tần số cao yếu hơn trung bình block về 0, trong khi không phạt trực tiếp các hệ số mạnh đại diện cạnh/chuyển động. Đây là surrogate lấy cảm hứng từ RPP; cần ablation riêng để xác định đóng góp.

## 6. Quyết định về đầu vào task

Mặc định task analyzer nhận video **sau postprocessor** (`--task-input postprocessed`) để một stream phục vụ cả người lẫn máy. `--task-input codec` là ablation trong đó máy đọc reconstruction trực tiếp, còn postprocessor chỉ phục vụ người. Hai cấu hình phải được báo cáo tách biệt vì chúng có chi phí triển khai và gradient khác nhau.

## 7. Triển khai Jetson

Preprocessor chạy trước encoder phần cứng, postprocessor chạy sau decoder phần cứng. Hai mạng được xuất ONNX riêng để không giả định rằng H.264/H.265 khả vi hoặc nằm trong TensorRT graph. DINOv2/proxy chỉ dùng khi huấn luyện; analyzer chỉ dùng nếu ứng dụng VCM thực sự thực hiện inference tại edge.

Kết luận về thời gian thực chỉ hợp lệ khi đo đồng thời pre + codec + post + task trên cùng Orin NX, cùng power mode, JetPack/TensorRT, resolution, GOP/preset, precision, batch, warm-up và số lần lặp.

## 8. Tiêu chí chấp nhận trước khi viết Results

1. Codec proxy đạt audit tái tạo/rate và gradient-direction trên validation cố định.
2. Đánh giá cuối chỉ dùng bitstream H.264/H.265 thật; proxy không xuất hiện trong forward.
3. So sánh paired trên cùng video/QP/seed giữa anchor, pre-only, và sandwich.
4. Báo cáo task BD-rate cùng khoảng tin cậy bootstrap; không chỉ chọn checkpoint tốt nhất trên test.
5. Báo cáo PSNR, MS-SSIM, LPIPS hoặc VMAF cùng task metric để thấy trade-off người–máy.
6. Đo p50/p95 latency, FPS, memory và VDD_IN power trực tiếp trên Orin NX.
7. Chạy đủ ablation để tách đóng góp pre, post, DINO, DCT, QP-FiLM và real-forward STE.

## 9. Các giới hạn hiện tại

- Code hiện thực đầy đủ action recognition Kinetics-style; tracking GOT-10k vẫn là hướng mở rộng, chưa phải capability hiện tại.
- Compact three-scale MS-SSIM là surrogate huấn luyện nội bộ, không thay thế implementation metric chuẩn dùng trong bảng cuối.
- `torch.onnx.export` tạo graph pre/post; việc TensorRT hỗ trợ đầy đủ operator Video Swin phải được xác nhận trên phiên bản JetPack cụ thể.
- Không có dữ liệu/checkpoint/Jetson trong workspace lúc đóng gói, vì vậy manuscript chưa có bảng kết quả của V9.
