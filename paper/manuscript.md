# Adaptive Video Preprocessing Techniques for Optimizing Video Coding for Machines (VCM) on NVIDIA Jetson Orin NX

Anonymous Author(s) — identities pending human approval

PRE-RESULTS MANUSCRIPT DRAFT — DO NOT SUBMIT

## Abstract

Video coding for machines (VCM) must preserve features required by downstream analysis while respecting bitrate, human-viewing quality, and edge-compute constraints. We present a standards-compatible adaptive video preprocessing framework that places trainable neural pre- and postprocessors around a frozen H.264/AVC or H.265/HEVC codec. The preprocessor is a compact QP-conditioned Video Swin network; the postprocessor is an identity-initialized FiLM-3D residual network. During optimization, both decoded pixels and bitrate in the forward pass come from the real codec, while a frozen predictive-entropy proxy supplies gradients through the non-differentiable coding operation. A composite objective combines measured rate, frozen task loss, frozen DINOv2 feature consistency, and human-oriented Charbonnier, multiscale structural, optional LPIPS, temporal, and adaptive-DCT terms. The proxy and DINOv2 are training-only, and the two wrappers can be exported independently to TensorRT around the hardware codec on NVIDIA Jetson Orin NX. This manuscript defines the method, falsifiable hypotheses, ablations, and a reproducible measurement protocol. No new rate–accuracy or Jetson measurements are claimed because the required dataset, trained checkpoint, and target hardware were not present at manuscript generation time.

Keywords—video coding for machines, neural preprocessing, standard codec, straight-through estimation, DINOv2, Video Swin Transformer, FiLM, Jetson Orin NX.

## 1. Introduction

Edge cameras increasingly transmit video for recognition rather than viewing alone. Conventional H.264/AVC and H.265/HEVC codecs are standardized, widely deployed, and hardware accelerated, but their native rate–distortion objectives do not directly model a downstream neural task. Replacing them with a learned codec can make the complete system differentiable, yet may sacrifice bitstream compatibility and mature hardware implementations. A practical alternative is to retain the codec and learn a transformation of its input.

Prior work provides complementary pieces of this solution. Lu et al. [1] learn a quantization-adaptive image preprocessor and show that using the real codec in the forward pass while using a proxy for backpropagation avoids a train–test mismatch. Sandwiched Compression [2] surrounds a standard codec with jointly trained neural wrappers and demonstrates that a codec can transport a learned neural code optimized for a metric unlike its native distortion. Zhao et al. [3] extend task-aware preprocessing to video using spatial/temporal processing and a differentiable virtual codec. Rate-Perception Optimized Preprocessing (RPP) [4] introduces an adaptive DCT term and perceptual training strategy. DINOv2 [5] offers broadly useful frozen visual features; FiLM [6] offers a compact mechanism to condition computation on codec QP; and Video Swin Transformer [7] supplies an efficient local spatiotemporal attention pattern.

These components are not sufficient by simple concatenation. A postprocessor can improve human quality while shifting the input distribution seen by the machine model. A codec proxy can provide gradients whose values differ materially from deployment. A perceptual term can preserve texture that is costly but irrelevant to the task, while a task loss can erase information required by people or by an unseen task. Finally, an architecture that improves rate–accuracy on a desktop GPU may violate latency or power constraints on an embedded device.

We therefore define a single, auditable framework whose key design rule is strict separation of forward truth from backward approximation. Real H.264/H.265 reconstructions and elementary-stream byte counts determine every training loss value and every evaluation result. The learned codec proxy contributes only a Jacobian. The human and machine objectives are explicit rather than collapsed into an undocumented score, and deployment excludes all training-only teachers.

The contributions of this work are:

1. A video codec sandwich combining a QP-FiLM Video Swin Lite preprocessor, a frozen standard codec, and a compact QP-FiLM 3-D postprocessor, both wrappers initialized as exact identity mappings.
2. A forward-real-codec/proxy-backward estimator for both decoded pixels and BPP, adapted from image preprocessing to a predictive video entropy proxy.
3. A joint VCM objective that balances measured rate, supervised task fidelity, DINOv2 semantic consistency, and human-oriented spatial, temporal, perceptual, and adaptive-DCT losses.
4. A deployment split that exports only the pre/post wrappers to ONNX/TensorRT around Jetson hardware codecs, together with a locked accuracy, latency, memory, and power protocol.
5. An open research implementation with gradient-invariant tests and an evidence ledger that prevents prior-work values from being presented as new results.

## 2. Related Work

### 2.1 Standard-compatible preprocessing for machines

Lu et al. [1] place a neural preprocessing module before a non-differentiable image codec. Their published ablation reports approximately 14.6% bitrate reduction for a proxy-forward configuration and 20.3% when the forward path uses real BPG while the proxy supplies the backward path, at equal object-detection accuracy. These are prior-work image results, not measurements of our video system. Their public implementation confirms an in-place value replacement pattern equivalent to a straight-through bridge. Zhao et al. [3] formulate video preprocessing with spatial and temporal branches and a virtual codec that represents prediction, transform/quantization, and entropy effects. Their reported savings motivate video modeling, but are not transferred to this work as empirical evidence.

### 2.2 Neural codec sandwiches and perception

Guleryuz et al. [2] jointly train pre/post networks around standard codecs through differentiable proxies. Their examples show that the transported signal can function as a neural code and can optimize LPIPS, VMAF, or a non-native signal domain. We retain their two-sided architecture but specialize it for machine-video semantics, temporal consistency, QP adaptation, and embedded deployment. RPP [4] uses an adaptive DCT objective, image-quality assessment, high-order degradation, and a lightweight network for perceptual coding. Our DCT term follows the principle of suppressing weak high-frequency coefficients while preserving strong coefficients; it is not claimed to be a line-for-line reproduction of RPP.

### 2.3 Semantic and conditional representations

DINOv2 [5] learns general-purpose visual features without task-specific labels. We use a frozen DINOv2 model as a framewise semantic teacher so that the restored stream is constrained beyond the selected action classifier. FiLM [6] computes featurewise affine modulation from a conditioning variable. Here, QP is the conditioning variable and is injected throughout the preprocessor, proxy, and postprocessor. Video Swin [7] restricts attention to shifted local windows. Our smaller variant preserves temporal resolution and reconstructs dense RGB, rather than performing classification itself.

## 3. Method

### 3.1 System definition

Let x in [0,1]^(T×3×H×W) be a source clip and q a codec quantization parameter. The trainable preprocessor P_theta produces a standards-compatible RGB neural code z = P_theta(x,q). A frozen standard codec C_q emits reconstruction y = C_q(z) and a bitstream of length b bits. The trainable postprocessor R_phi reconstructs x_hat = R_phi(y,q). By default, a frozen task analyzer A consumes x_hat; an ablation routes y directly to A.

The deployment graph is:

```text
x -> P_theta(.,q) -> H.264/H.265 encoder -> bitstream
  -> H.264/H.265 decoder -> R_phi(.,q) -> human and/or machine consumer
```

P_theta is a Video Swin Lite residual network. It uses a spatial Conv3D patch embedding, alternating regular and shifted 3-D windows, QP-FiLM modulation, normalization, and a ConvTranspose3D RGB head. Time is never downsampled. R_phi is a shallow 3-D residual U-Net with depthwise-separable blocks, one spatial down/up stage, and QP-FiLM. The two RGB heads are zero initialized, so both new networks are exact identities before learning.

### 3.2 Real-forward, proxy-backward codec bridge

The standard codec is non-differentiable. Let C_hat_psi(z,q) and r_hat_psi(z,q) be frozen differentiable proxies for reconstructed video and BPP. Let y_real and r_real be the real decoded video and elementary-stream BPP. Training uses:

```text
y_ST = y_proxy + stopgrad(y_real - y_proxy)
r_ST = r_proxy + stopgrad(r_real - r_proxy).
```

Consequently, value(y_ST)=y_real and value(r_ST)=r_real, while gradients with respect to z are those of the proxy. This estimator is not an unbiased gradient of the discrete codec. Its role is pragmatic and must be audited. We require proxy reconstruction/rate agreement and a directional test: a small input update that decreases proxy rate should reduce real measured BPP more often than a predeclared acceptance threshold on held-out samples.

The predictive proxy encodes an I-frame signal and previous-frame prediction residuals with QP-conditioned 3-D analysis transforms. A straight-through quantizer feeds a factorized Laplace likelihood; entropy is calibrated by a positive QP-dependent affine mapping. A synthesis transform integrates the decoded residuals. The proxy, codec, task analyzer, and DINOv2 weights are frozen during wrapper training.

### 3.3 Multi-objective optimization

For a codec/QP-specific anchor BPP r_anchor(q), the objective is

```text
L = w_r (r_ST / r_anchor(q)) + w_t L_task
    + w_s L_DINO + w_h L_human.
```

L_task is cross-entropy from the frozen video analyzer. L_DINO is a weighted cosine distance between frozen DINOv2 CLS and patch tokens for x_hat and x. Source features are detached; gradients for the restored clip pass through the fixed teacher network.

The human-view term is

```text
L_human = a L_Charbonnier + b (1 - MS-SSIM_3)
        + c LPIPS + d L_temporal + e L_DCT.
```

MS-SSIM_3 is a compact three-scale differentiable training surrogate; final tables must use a validated external implementation. L_temporal is the L1 distance between adjacent-frame differences. For L_DCT, the neural-code frames are partitioned into orthonormal 8×8 DCT blocks. Coefficients on or above a selected zig-zag frequency diagonal are compared with their detached per-block mean magnitude. Only weaker coefficients are pulled toward zero, retaining strong edge or motion detail. LPIPS is optional because it increases training cost and dependency weight.

The normalized rate term prevents high-rate QPs from dominating merely by scale. All weights are nonnegative and recorded in the checkpoint. The primary configuration uses the postprocessed clip for both task and human losses; routing task loss before postprocessing is a required ablation.

### 3.4 Training and checkpoint selection

Training has three stages. First, real codec caches are generated for fixed train/validation samples and QPs. Second, the predictive proxy is distilled against decoded pixels and BPP, with paired rate-delta supervision and real-codec gradient probes. Third, the pre/post wrappers are optimized while all other networks remain frozen. The default checkpoint rule uses validation loss only until the first mathematically valid task BD-rate is observed. That first valid estimate replaces the fallback checkpoint; thereafter an undefined BD-rate cannot replace a valid incumbent. The checkpoint stores the optimizer, scaler, selection state, random-number-generator states, proxy hash, and codec configuration so a V10 resume reproduces the interrupted trajectory rather than silently restarting parts of it.

## 4. Experimental Protocol

### 4.1 Dataset, task, and codecs

The implemented primary task is Kinetics-400 action recognition [8] with a frozen video analyzer. Each example contains 16 RGB frames sampled at stride 2 and resized/cropped to 128×128. Official train/validation identities should be used; when unavailable, the code generates one deterministic stratified validation split and persists its seed. The final scientific forward path uses FFmpeg `libx264` and `libx265` at QP {30,32,35,37,40,42,45}. Hardware-codec measurements on Jetson form a separate implementation stratum.

For every video, codec, and QP, evaluation writes an anchor, pre-only, and full-sandwich record with a stable sample identity. BPP is computed from elementary-stream bytes divided by T×H×W. Machine metrics are Top-1 and Top-5. Human/reconstruction metrics are PSNR, validated MS-SSIM, LPIPS, and VMAF [9] where available. The primary aggregate PSNR is computed from the mean video MSE, not by averaging PSNR values in decibels. Task BD-rate uses Top-1 as quality; perceptual BD-rate is reported separately. The primary BD-rate uses a monotone quality envelope, while the raw-curve result is retained as a sensitivity diagnostic. A BD-rate is undefined when the selected curves lack sufficient distinct points or an overlapping quality range.

### 4.2 Ablations and uncertainty

Required ablations remove the postprocessor, DINO loss, adaptive-DCT loss, or QP-FiLM; replace Video Swin with the retained CNN preprocessor; route the machine before postprocessing; and replace real-forward training with proxy-forward training. Experiments use three seeds. Paired bootstrap resampling operates over video identity with 10,000 samples and a fixed seed. The valid-resample count must accompany the 95% interval when some resampled BD-rate curves are undefined.

### 4.3 Jetson Orin NX protocol

The pre/post networks are exported independently to ONNX and converted to TensorRT at a recorded precision. The H.264/H.265 encoder and decoder remain NVIDIA hardware codec elements. The report must identify the exact Orin NX memory variant, carrier board, JetPack/L4T, CUDA, cuDNN, TensorRT, power mode, clocks, thermal range, codec parameters, precision, and engine hashes. After at least 30 warm-up iterations, at least 200 iterations measure pre, encode, decode, post, task, and end-to-end p50/p95 latency. `tegrastats` provides total-module VDD_IN samples; results include power and energy/frame rather than inferring watts from latency.

## 5. Results

No V10 empirical result is available in the present workspace. Table 1 is a schema, not a result table. It must be populated only from held-out real-codec artifacts satisfying the locked protocol.

Table 1. Primary result schema; all V10 cells are intentionally unmeasured.

| Codec | Method | Task BD-rate (%) | PSNR BD-rate (%) | MS-SSIM BD-rate (%) | Top-1 at lowest rate | 95% CI |
|---|---|---:|---:|---:|---:|---|
| H.264 | Anchor | reference | reference | reference | NOT MEASURED | — |
| H.264 | Pre-only | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| H.264 | Sandwich | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| H.265 | Anchor | reference | reference | reference | NOT MEASURED | — |
| H.265 | Pre-only | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| H.265 | Sandwich | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Table 2. Jetson result schema; all cells are intentionally unmeasured.

| Precision | Pre p50/p95 (ms) | Codec p50/p95 (ms) | Post p50/p95 (ms) | End-to-end FPS | Peak memory | VDD_IN (W) | Energy/frame |
|---|---|---|---|---:|---:|---:|---:|
| FP16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

The source papers report useful context: Lu et al. [1] report approximately −20.3% versus −14.6% in their real-forward and proxy-forward image configurations; Zhao et al. [3] report more than 15% video bitrate saving; RPP [4] reports 16.27% average bitrate saving; and Sandwiched Compression [2] reports gains up to 30% in selected adaptations. These results have different tasks, codecs, data, and objectives. They are not targets, baselines, or substitutes for V10 measurements.

## 6. Discussion

The proposed system preserves codec compatibility at the cost of an approximate gradient. The exact forward values eliminate one important form of proxy mismatch, but the update direction can still be poor. Proxy auditing is therefore part of the method, not a debugging convenience. Joint postprocessing also creates a deployment choice. If both people and machines consume x_hat, its latency is mandatory and the task network sees the perceptually restored distribution. If only people use x_hat, machines can consume y earlier, reducing machine latency but optimizing two outputs.

DINOv2 adds protection against overfitting to one analyzer, but it is framewise and may miss motion semantics. The supervised video loss and temporal reconstruction loss partly address this limitation. Conversely, adaptive DCT encourages compressibility but could suppress subtle task signals. The corresponding ablations and per-class analysis are needed to determine whether the objectives cooperate or conflict.

The two-sided sandwich is also a system contract: the decoder side must have the matching postprocessor checkpoint. Pre-only operation remains valuable where receivers cannot be updated. Reporting both variants makes the compatibility trade-off visible.

## 7. Limitations, Ethics, and Reproducibility

This release has not been trained or measured on Kinetics-400 or Jetson Orin NX in the supplied environment. It must not be advertised as achieving a bitrate, accuracy, real-time, or power improvement. Action recognition is implemented; video tracking is not. The compact MS-SSIM function is suitable as a differentiable training term but not as a standards-grade reporting implementation. TensorRT conversion is version dependent and must be verified on the target JetPack release.

Task-aware compression may discard information that is irrelevant to a selected model but meaningful to people, future models, safety review, accessibility, or forensic analysis. Human-oriented losses reduce but do not eliminate this risk. Deployments should retain an auditable policy for source retention, downstream task changes, and demographic/per-class failure analysis. Kinetics and other video datasets also require license, privacy, and content review.

The repository contains source code, tests, an experiment protocol, a Jetson runbook, and claim/source manifests. The current test suite verifies identity initialization, codec freezing, real-forward equality, proxy-gradient flow, DINO freezing with input gradients, DCT gradients, metric behavior, inherited proxy invariants, and the V10 checkpoint/evaluation regressions.

## 8. Conclusion

We introduced a standards-compatible adaptive video codec sandwich for VCM. The design combines local spatiotemporal preprocessing, exact real-codec forward values, proxy-only gradients, lightweight postprocessing, task-specific and DINOv2 semantic supervision, and human-oriented perceptual constraints. Its deployment graph isolates pre/post TensorRT engines from a frozen hardware codec and its protocol makes rate–accuracy and edge-efficiency claims falsifiable. The implementation is ready for the missing empirical stage; the manuscript remains a pre-results draft until the declared experiments and human review are complete.

## References

[1] G. Lu, X. Ge, T. Zhong, Q. Hu, and J. Geng, “Preprocessing Enhanced Image Compression for Machine Vision,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 12, pp. 13556–13568, 2024, doi:10.1109/TCSVT.2024.3441049.

[2] O. G. Guleryuz et al., “Sandwiched Compression: Repurposing Standard Codecs with Neural Network Wrappers,” arXiv:2402.05887, 2024.

[3] F. Zhao, M. Guo, S. Zhao, J. Li, L. Zhang, and X. Xie, “A Preprocessing Framework for Video Machine Vision under Compression,” arXiv:2512.15331, 2025.

[4] C. Ma, Z. Wu, C. Cai, P. Zhang, Y. Wang, L. Zheng, C. Chen, and Q. Zhou, “Rate-Perception Optimized Preprocessing for Video Coding,” arXiv:2301.10455, 2023.

[5] M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” arXiv:2304.07193, 2023.

[6] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual Reasoning with a General Conditioning Layer,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[7] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, “Video Swin Transformer,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3202–3211, 2022.

[8] W. Kay et al., “The Kinetics Human Action Video Dataset,” arXiv:1705.06950, 2017.

[9] Z. Li et al., “VMAF: The Journey Continues,” Netflix Technology Blog, 2018.

[10] T. Wiegand, G. J. Sullivan, G. Bjøntegaard, and A. Luthra, “Overview of the H.264/AVC Video Coding Standard,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 13, no. 7, pp. 560–576, 2003.

[11] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the High Efficiency Video Coding (HEVC) Standard,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649–1668, 2012.

[12] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.
