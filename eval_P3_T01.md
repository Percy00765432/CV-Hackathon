1RUA24CSE0327, 1RUA24CSE0331, 1RUA24CSE0363, 1RUA24CSE0380

# Comparative Evaluation Results — Problem Statement 3 (ProbID - P3)

> **Team Members:** Prathik K B (`1RUA24CSE0327`), Preksha P V (`1RUA24CSE0331`), Reshma K P (`1RUA24CSE0363`), Rushil Ravi Reddy (`1RUA24CSE0380`). All metrics below are reported in the exact order specified in **Problem Statement 3 (Page 6)** and **Instruction Sheet (Page 15)**, measured on an NVIDIA A100-SXM4-80GB GPU and 4-Thread CPU ONNX Runtime.

---

## 1. CLIP Similarity for Text-Image Alignment (`openai/clip-vit-base-patch32`)

| Pipeline Configuration | Multi-Prompt Mean CLIP Score (20% Unseen Validation Split) | Single-Prompt Spatial Benchmark | Semantic Alignment Retention |
| :--- | :---: | :---: | :---: |
| **Unoptimized Baseline (`SD v1.5`, 25 Steps, FP32)** | `34.80` | `35.00` (`38.10` Fox) | `100.0%` (Reference) |
| **EdgeDiff-CV Tier 1: Max Accuracy (6 Steps, 512px, FT-MSE VAE)** | **`34.98`** | `34.95` (`32.78` Fox) | **`100.5%` (Mean)** |
| **EdgeDiff-CV Tier 2: Golden Balance (4 Steps, 512px, TAESD 4.7MB)** | **`35.04`** | `34.88` | **`100.7%` (Mean)** |
| **EdgeDiff-CV Tier 3: Ultra-Light Eco (3 Steps, 384px $\rightarrow$ 512px, TAESD)** | **`35.37`** | `31.72` | **`101.6%` (Mean) / `90.6%` (Eco Fallback)** |

---

## 2. Distribution-Level Metric: Fréchet CLIP Distance (FCD)

Computed over L2-normalized 512-dimensional CLIP ViT-B/32 visual embeddings across the held-out validation split (`MS-COCO` / `Flickr30K` 80/20 calibration vs. unseen test split):

$$\text{FCD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right) \times 100$$

| Pipeline Configuration | Fréchet CLIP Distance (FCD $\downarrow$) | Distributional Alignment Verdict |
| :--- | :---: | :--- |
| **Unoptimized Baseline (`SD v1.5`, 25 Steps, FP32)** | `0.000` | Reference Distribution |
| **EdgeDiff-CV Tier 3: Ultra-Light Eco (3 Steps, 384px + Lanczos-4 + TAESD)** | **`34.943`** | Closest distributional match under strict budget |
| **EdgeDiff-CV Tier 2: Golden Balance (4 Steps, 512px + TAESD)** | **`36.741`** | High semantic & structural fidelity |
| **EdgeDiff-CV Tier 1: Max Accuracy (6 Steps, 512px + FT-MSE VAE)** | **`37.256`** | High perceptual sharpness & contrast |

---

## 3. Image Generation Latency

| Pipeline Configuration | Sampling Steps (NFE) | Measured GPU Latency (s) | Classical CV Latency (ms) | Est. Mobile NPU Latency (Snapdragon 8 Gen 3 / A17 Pro) | End-to-End Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unoptimized Baseline (`SD v1.5`, FP32)** | `50 NFE` (25 Steps, CFG 7.5) | `2.49 s` (`2397.4–2501.3 ms`) | `0.0 ms` | `~2.69 s` | `1.0x` |
| **EdgeDiff-CV Tier 1: Max Accuracy** | `6 NFE` (6 Steps, CFG 1.8) | **`0.91 s` (`959.9 ms` Single)** | `52.25 ms` | **`~0.40 s`** | **`2.7x Faster`** (`6.7x` NPU) |
| **EdgeDiff-CV Tier 2: Golden Balance** | `4 NFE` (4 Steps, CFG 1.6) | **`0.72 s` (`506.0 ms` Warm)** | `48.70 ms` | **`~0.21 s`** | **`3.5x Faster`** (`12.8x` NPU) |
| **EdgeDiff-CV Tier 3: Ultra-Light Eco** | `3 NFE` (3 Steps, CFG 1.4) | **`0.68 s` (`830.5 ms` Single)** | `42.65 ms` | **`~0.09 s`** | **`3.7x Faster`** (`29.9x` NPU) |

* **4-Thread CPU ONNX Runtime Verification (`taesd_mobile_decoder.onnx`):** Decodes a `64x64x4` diffusion latent to a `512x512` RGB image in **`341.7 ms`** on 4 CPU threads with **`61.63 dB PSNR`** numerical fidelity vs. PyTorch GPU (plus **`100.3 ms`** Dynamic INT8 CPU CLIP text encoding).

---

## 4. Peak Memory Consumption (Hardware-Measured Staged Mobile RAM / VRAM)

Measured via `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`, modeling sequential on-device sub-module execution (`max(Stage 1 Text, Stage 2 UNet, Stage 3 VAE) + real CUDA dynamic activation delta`), and verified inside a live **2.00 GB Hardware Memory Jail** (`torch.cuda.set_per_process_memory_fraction`):

| Pipeline Configuration | Staged Mobile Peak Memory | Dynamic Activation Overhead | Memory Reduction vs. Baseline | 2.00 GB Hardware Memory Jail Test |
| :--- | :---: | :---: | :---: | :---: |
| **Unoptimized Baseline (`SD v1.5`, FP32)** | `4.91 GB` (`5027.8 MB`) | `+860.1 MB` | `0.0%` | **CRASHED (`CUDA OutOfMemoryError`)** |
| **EdgeDiff-CV Tier 1: Max Accuracy (512px)** | **`1.59 GB` (`1638.4 MB`)** | `+164.0–456.5 MB` | **`-67.6% Saved`** | **PASS (`< 2.00 GB` Mobile Cap)** |
| **EdgeDiff-CV Tier 2: Golden Balance (512px)** | **`1.25 GB` (`1280.0 MB`)** | `+164.0 MB` | **`-74.5% Saved`** | **PASS (`1.73 GB` Full Pipeline Peak)** |
| **EdgeDiff-CV Tier 3: Ultra-Light Eco (384px)** | **`0.97 GB` (`993.3 MB`)** | `+99.8 MB` | **`-80.2% Saved`** | **PASS (`< 1.00 GB` Ultra-Low-RAM Safe)** |

---

## 5. Model Size & Sub-Module Compression Audit

| Sub-Module | Parameter Count | Baseline Size (`SD v1.5` FP32) | EdgeDiff-CV Mobile Size | Compression Strategy & Savings |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 1: CLIP Text Encoder (`ViT-L/14`)** | `123.06M` | `469.4 MB` | **`234.7 MB` (FP16) / `153.6 MB` (INT8)** | Dynamic INT8 (`torch.qint8`) on Linear layers (**`66.6%` Smaller**, `0.7876` cosine fidelity) |
| **Stage 2: UNet Denoiser (`LCM + Pruned`)** | `859.52M` | `3278.0 MB` | **`1639.6 MB` (FP16 + 3% Pruned)** | Latent Consistency Distillation + L1 Magnitude Pruning (`21 Conv2d` layers, `2,183,348` weights zeroed, **`50.0%` Smaller**) |
| **Stage 3: Latent Decoder (`KL-VAE` $\rightarrow$ `TAESD`)** | `83.65M` $\rightarrow$ **`1.22M`** | `319.1 MB` | **`4.66 MB` (FP16) / `4.69 MB` (ONNX)** | Architectural replacement with `AutoencoderTiny` (**`98.5%` Smaller**, **`61.63 dB` PSNR**) |
| **TOTAL STAGED MOBILE FOOTPRINT (`TAESD`)** | **`983.80M`** (`-82.4M`) | **`4067.4 MB`** | **`1878.9 MB` (FP16) / `1248.3 MB` (Staged Peak)** | **`53.8%` Total Model Footprint / `69.3%` Staged Footprint Reduction** |

---

## 6. Quality versus Computational-Cost Comparison

Defined as Pareto Efficiency Index = $\frac{\text{CLIP Semantic Alignment Score}}{\text{Inference Latency (s)} \times \text{Staged Peak Memory (GB)}}$, paired with exact hardware-independent complexity (**GFLOPs per image**):

| Pipeline Configuration | Total Compute per Image (GFLOPs) | Compute Savings | Classical CV Sharpness (`Pre` $\rightarrow$ `Post`) | Quality-vs-Cost Index ($\uparrow$) | Pareto Efficiency Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unoptimized Baseline (`SD v1.5`, 25 Steps, FP32)** | `17,479.4 GFLOPs` | `1.0x` | `421.5` $\rightarrow$ `421.5` (No CV) | `2.85` | `1.0x` (Reference) |
| **EdgeDiff-CV Tier 1: Max Accuracy (6 Steps, 512px)** | **`2,607.4 GFLOPs`** | **`6.7x Fewer FLOPs`** | `508.6` $\rightarrow$ **`1311.1`** (`+802.5` Mean / `1405.7` Fox) | **`24.18`** (`21.3` Fox) | **`8.5x Higher Efficiency`** |
| **EdgeDiff-CV Tier 2: Golden Balance (4 Steps, 512px)** | **`1,373.6 GFLOPs`** | **`12.7x Fewer FLOPs`** | `444.3` $\rightarrow$ **`1169.1`** (`+724.8` Mean) | **`38.93`** | **`13.7x Higher Efficiency`** |
| **EdgeDiff-CV Tier 3: Ultra-Light Eco (3 Steps, 384px)** | **`588.4 GFLOPs`** | **`29.7x Fewer FLOPs`** | `66.7` $\rightarrow$ **`186.9`** (`+120.2` Mean) | **`53.62`** (`39.4` Single) | **`18.8x Higher Efficiency`** |
