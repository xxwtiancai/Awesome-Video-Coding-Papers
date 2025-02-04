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

### **2014**
- **`[CoRR 2014]`** [**Auto-encoders: reconstruction versus compression**](https://arxiv.org/abs/1403.7752) 
  - *Key Idea*: Theoretical exploration of autoencoders for compression by minimizing code length.  
  - Code: Not Provided

---

### **2016**
- **`[CoRR 2016]`** [**Towards Conceptual Compression**](https://arxiv.org/abs/1604.08772) 
  - *Key Idea*: Recurrent VAE for latent space learning in conceptual compression.  
  - Code: Not Provided

- **`[ICLR 2016]`** [**Autoencoding beyond pixels using a learned similarity metric**](https://arxiv.org/abs/1512.09300)
  - *Key Idea*: KL divergence-based VAE for image generation.  
  - Code: Not Provided

- **`[ICLR 2016]`** [**Variable Rate Image Compression with Recurrent Neural Networks**](https://arxiv.org/abs/1511.06085) 
  - *Key Idea*: Adaptive RNN-based compression on Kodak dataset.  
  - Code: [https://github.com/mr-mikmik/VRIC-RNN](https://github.com/mr-mikmik/VRIC-RNN)

---

### **2017**
- **`[CVPR 2017]`** [**Full Resolution Image Compression with Recurrent Neural Networks**](https://arxiv.org/abs/1608.05148)  
  - *Key Idea*: Combines RNN and entropy encoding for high MS-SSIM.  
  - Code: [https://github.com/1zb/pytorch-image-comp-rnn](https://github.com/1zb/pytorch-image-comp-rnn)

- **`[DCC 2017]`** [**Semantic Perceptual Image Compression using Deep Convolution Networks**](https://arxiv.org/abs/1612.08712)
  - *Key Idea*: Semantic-aware compression with CNN.  
  - Code: [https://github.com/iamaaditya/image-compression-cnn](https://github.com/iamaaditya/image-compression-cnn)

---

### **2018**
- **`[CVPR 2018]`** [**Variational Autoencoder for Low Bit-rate Image Compression**](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w50/Zhou_Variational_Autoencoder_for_CVPR_2018_paper.pdf)
  - *Key Idea*: Nonlinear encoder + uniform quantizer for low-bitrate compression.  
  - Code: Not Provided | Bibtex: See [Ref 47]

- **`[NIPS 2018]`** [**Joint Autoregressive and Hierarchical Priors for Learned Image Compression**](https://arxiv.org/abs/1809.02736)
  - *Key Idea*: Hybrid autoregressive and hierarchical priors for entropy modeling.  
  - [Code](https://github.com/thekoshkina/learned_image_compression)
- **`[CVPR 2018]`** [**Deep Image Compression via End-to-End Learning**](https://arxiv.org/abs/1806.01496)
  - *Key Idea*: End-to-end CNN framework with CLIC 2018 dataset.  
  - Code: Not Provided
- **`[NIPS 2018]`** [**Non-local Recurrent Network for Image Restoration**](https://arxiv.org/abs/1806.02919) 
  - *Key Idea*: Integrates non-local operations with RNN for artifact reduction.  
  - [Code](https://github.com/Ding-Liu/NLRN)
---

### **2019**
- **`[ICCV 2019]`** [**Generative adversarial networks for extreme learned image compression**](https://arxiv.org/abs/1804.02958) 
  - *Key Idea*: GAN-based compression targeting <0.1 bpp with selective generative modules.  
  - Code: [GitHub](https://github.com/Justin-Tan/generative-compression)

---

### **2020**
- **`[WACV 2020]`** [**CompressNet: Generative Compression at Extremely Low Bitrates**](https://arxiv.org/abs/2006.08003) 
  - *Key Idea*: Combines MSE, adversarial, and layer-wise losses for ultra-low bpp.  
  - [Code](https://github.com/shubham14/CompressNet)

- **`[ICIP 2020]`** [**Channel-Wise Autoregressive Entropy Models for Learned Image Compression**](https://arxiv.org/abs/2007.08739)
  - *Key Idea*: Channel-wise autoregressive entropy model for ultra-low bpp (0.01921).  
  - [Code](https://github.com/tokkiwa/minnen2020)

---

### **2021**
- **`[TIP 2021]`** [**End-to-End Learnt Image Compression via Non-Local Attention Optimization**](https://ieeexplore.ieee.org/document/9359473)  
  - *Key Idea*: Non-local attention modules (NLAM) for global context modeling.  
  - Code: [GitHub](https://github.com/NJUVISION/NIC) | Bibtex: See [Ref 48]

- **`[TCSVT 2021]`** [**Learned Block-based Hybrid Image Compression**](https://arxiv.org/abs/2012.09550)
  - *Key Idea*: Block-based partitioning to address OOM issues in high-resolution images.  
  - Code: Not Provided

---

### **2022**
- **`[CVPR 2022 Oral]`** [**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Context**](https://arxiv.org/abs/2203.10886)  
  - *Key Idea*: Advanced entropy modeling with channel-wise and spatial context grouping.  
  - [Code](https://github.com/VincentChandelier/ELiC-ReImplemetation) | Bibtex: See [Ref ELIC]

### **2023**

### **2024**
- **`[CVPR 2024]`** [**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Context**](https://arxiv.org/abs/2303.10807)
  - *Key Idea*: Advanced entropy modeling with channel-wise context.  
  - [Code](https://github.com/elic-project)
- **`[ICLR 2024]`** [**Towards Image Compression with Perfect Realism at Ultra-Low Bitrates**](https://arxiv.org/abs/2310.10325)
  - *Key Idea*: Using Stable Diffusion v2.1 (Rombach et al., CVPR 2022) as latent diffusion model and hence refer to our work as PerCo (SD)
  - [Code](https://github.com/Nikolai10/PerCo)



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