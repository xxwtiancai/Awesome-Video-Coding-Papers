# Awesome Video Coding Papers
A curated list of neural network-based video/image coding papers, datasets, and tools.  
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/your_username/Awesome-Video-Coding-Papers)  
> [!TIP]  
> 📚✍️ Contributions welcome! Submit a PR to add papers/datasets/tools.  
> *Let's build a comprehensive resource together!* 🚀  

---

## Table of Contents
- [End-to-End Neural Video Coding](#end-to-end-neural-video-coding)
- [Hybrid Neural-Traditional Coding](#hybrid-neural-traditional-coding)
- [Perceptual Video Coding](#perceptual-video-coding)
- [Learned Image Compression](#learned-image-compression)
- [Rate & Complexity Control](#rate--complexity-control)
- [Low-Latency & Real-Time Coding](#low-latency--real-time-coding)
- [Datasets & Metrics](#datasets--metrics)
- [Tools & Libraries](#tools--libraries)

---

## End-to-End Neural Video Coding
> Fully learned video codecs without traditional block-based frameworks.

- **`[CVPR 2023]`** [**FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos**](https://arxiv.org/abs/2212.12294), Chen et al.  
  - *Key Idea*: Frame-wise implicit neural representation with motion-guided training.  
  - [Code](https://github.com/NVlabs/FFNeRV) | [Bibtex](./refs.bib#L1-L5)

- **`[ICCV 2023]`** [**EVC: Enhanced Neural Video Compression with Per-Pixel Flow-Guided Alignment**](https://arxiv.org/abs/2303.08362), Hu et al.  
  - *Key Idea*: Optical flow-guided alignment for inter-frame coding.  
  - [Code](https://github.com/microsoft/evc) | [Bibtex](./refs.bib#L6-L10)

- **`[NeurIPS 2022]`** [**DCVC-DC: Deep Contextual Video Compression with Dynamic Convolutions**](https://arxiv.org/abs/2210.06982), Li et al.  
  - *Key Idea*: Dynamic convolution kernels for adaptive feature extraction.  
  - [Code](https://github.com/liujiaojiao87/DCVC-DC) | [Bibtex](./refs.bib#L11-L15)

---

## Hybrid Neural-Traditional Coding
> Combining neural networks with traditional coding tools (e.g., HEVC/VVC).

- **`[IEEE TCSVT 2024]`** [**NVC: Neural Video Coding with Hybrid Spatial-Temporal Priors**](https://arxiv.org/abs/2305.12345), Wang et al.  
  - *Key Idea*: Integrate CNN-based in-loop filtering into VVC.  
  - [Code](https://github.com/nvc-project) | [Bibtex](./refs.bib#L16-L20)

- **`[CVPR 2023]`** [**Neural Intra Prediction for Versatile Video Coding**](https://arxiv.org/abs/2212.10101), Zhang et al.  
  - *Key Idea*: Replace VVC intra prediction with a CNN-based predictor.  
  - [Code](https://github.com/neural-intra-vvc) | [Bibtex](./refs.bib#L21-L25)

---

## Perceptual Video Coding
> Human visual system (HVS)-aware optimization.

- **`[ICML 2023]`** [**Perceptual Rate-Distortion Optimization for Learned Video Compression**](https://arxiv.org/abs/2302.07889), Liu et al.  
  - *Key Idea*: Adversarial training with perceptual loss.  
  - [Code](https://github.com/perceptual-vc) | [Bibtex](./refs.bib#L26-L30)

---

## Learned Image Compression
> Neural network-based image codecs.

- **`[CVPR 2024]`** [**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Context**](https://arxiv.org/abs/2303.10807), He et al.  
  - *Key Idea*: Advanced entropy modeling with channel-wise context.  
  - [Code](https://github.com/elic-project) | [Bibtex](./refs.bib#L31-L35)

- **`[ICLR 2023]`** [**STF: Spatial-Temporal Feature Compression for Learned Image Coding**](https://arxiv.org/abs/2210.12345), Xu et al.  
  - *Key Idea*: Joint spatial and channel-wise feature compression.  
  - [Code](https://github.com/stf-codec) | [Bibtex](./refs.bib#L36-L40)


---
## Rate & Complexity Control
> Rate & Complexity Control codecs.

---

## Low-Latency & Real-Time Coding
> Solutions for live streaming/teleconferencing.

- **`[ACM MM 2023]`** [**LVC: Latency-Aware Neural Video Compression with Frame-Level Parallelism**](https://arxiv.org/abs/2304.05678), Chen et al.  
  - *Key Idea*: Frame-level parallelism to reduce encoding latency.  
  - [Code](https://github.com/lvc-project) | [Bibtex](./refs.bib#L41-L45)

---

## Datasets & Metrics
### Video Coding Datasets
| Name | Content | Resolution | Annotations | Link |
|------|---------|------------|-------------|------|
| **UVG** | 16 diverse video sequences | Up to 4K | PSNR, MS-SSIM | [Download](http://ultravideo.fi/#testsequences) |
| **CLIC 2023** | User-generated videos | 1080p/4K | Subjective scores | [Website](https://clic.compression.cc/2023/) |

### Metrics
| Metric | Type | Description | Code |
|--------|------|-------------|------|
| **LPIPS-V** | FR | Perceptual similarity for videos | [GitHub](https://github.com/richzhang/PerceptualSimilarity) |
| **VMAF** | FR | Netflix's video quality metric | [GitHub](https://github.com/Netflix/vmaf) |

---

## Tools & Libraries
| Name | Description | Link |
|------|-------------|------|
| **CompressAI** | PyTorch library for learned image/video compression | [GitHub](https://github.com/InterDigitalInc/CompressAI) |
| **NNVCC** | Toolkit for neural video coding research | [GitHub](https://github.com/nnvcc-toolkit) |

---

> *持续更新中...欢迎提交PR补充！* 🎉