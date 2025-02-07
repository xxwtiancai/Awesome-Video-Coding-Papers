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
- [Frame Intra Prediction](#frame-intra-prediction)
- [Frame Inter Prediction](#frame-inter-prediction)
- [Transform Coding](#transform-coding)
- [Quantization](#quantization)
- [Entropy Coding](#entropy-coding)
- [Loop Filtering](#loop-filtering)
- [Rate-Distortion Optimization](#rate-distortion-optimization)
- [Rate Control](#rate-control)
- [Complexity Control](#complexity-control)
- [Datasets](#datasets)
- [Tools](#tools)
### Frame Intra Prediction
- **`[TCSVT 2024]`** [**STRANet: soft-target and restriction-aware neural network for efficient VVC intra coding**](https://ieeexplore.ieee.org/abstract/document/10599317) [CODE](https://github.com/cppppp/STRANet)


- **`[ISCAS 2023]`** [**Ultra-Lightweight CNN Based Fast Intra Prediction for VVC Screen Content Coding**](https://doi.org/10.1109/ISCAS46773.2023.10181706) [CODE]()
- **`[TCSVT 2023]`** [**Machine Learning Based Efficient QT-MTT Partitioning Scheme for VVC Intra Encoders**](https://ieeexplore.ieee.org/abstract/document/10004946) [CODE](https://alexandretissier.github.io/QTMTT_VVC/)
- **`[TCSVT 2023]`** [**Deep Multi-Task Learning Based Fast Intra-Mode Decision for Versatile Video Coding**](https://ieeexplore.ieee.org/document/10083102) [CODE]()
- **`[TIP 2023]`** [**Partition Map Prediction for Fast Block Partitioning in VVC Intra-Frame Coding**](https://ieeexplore.ieee.org/abstract/document/10102791) [CODE](https://github.com/AolinFeng/PMP-VVC-TIP2023)


- **`[TMM 2022]`** [**Fast Intra Mode Decision Algorithm for Versatile Video Coding**](https://dl.acm.org/doi/10.1109/TMM.2021.3052348) [CODE]()
- **`[TMM 2022]`** [**Efficient VVC Intra Prediction Based on Deep Feature Fusion and Probability Estimation**](https://ieeexplore.ieee.org/abstract/document/9899414) [CODE]()
- **`[VCIP 2022]`** [**Autoencoder-based intra prediction with auxiliary feature**](https://doi.org/10.1109/VCIP56404.2022.10008846) [CODE]()
- **`[VCIP 2022]`** [**Neural Frank-Wolfe Policy Optimization for Region-of-Interest Intra-Frame Coding with HEVC/H.265**](https://doi.org/10.1109/VCIP56404.2022.10008853) [CODE]()
- **`[PCS 2022]`** [**Effective VVC Intra Prediction Based on Ensemble Learning**](https://doi.org/10.1109/PCS56426.2022.10018067) [CODE]()
- **`[TCSVT 2022]`** [**HG-FCN: Hierarchical Grid Fully Convolutional Network for Fast VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/9691378) [CODE]()
- **`[TCSVT 2022]`** [**Configurable Fast Block Partitioning for VVC Intra Coding Using Light Gradient Boosting Machine**](https://ieeexplore.ieee.org/document/9524713) [CODE]()
- **`[DCC 2022]`** [**Graph-based Transform based on 3D Convolutional Neural Network for Intra-Prediction of Imaging Data**](https://doi.org/10.1109/DCC52660.2022.00029) [CODE]()
- **`[2022]`** [**An efficient low-complexity block partition scheme for VVC intra coding**](https://link.springer.com/article/10.1007/s11554-021-01174-z) [CODE](https://github.com/csust-sonie/fastVVC_RTIP_2107)


- **`[PCS 2021]`** [**Contour-based Intra Coding Using Gaussian Processes and Neural Networks**](https://doi.org/10.1109/PCS50896.2021.9477500) [CODE]()
- **`[VCIP 2021]`** [**Learning-Based Complexity Reduction Scheme for VVC Intra-Frame Prediction**](https://doi.org/10.1109/VCIP53242.2021.9675394) [CODE]()
- **`[ICIP 2021]`** [**Intra To Inter: Towards Intra Prediction for Learning-Based Video Coders Using Optical Flow**](https://doi.org/10.1109/ICIP42928.2021.9506274) [CODE]()
- **`[TIP 2021]`** [**DeepQTMT: A Deep Learning Approach for Fast QTMT-Based CU Partition of Intra-Mode VVC**](https://doi.org/10.1109/TIP.2021.3083447) [CODE]()
- **`[DCC 2021]`** [**Fast Partitioning for VVC Intra-Picture Encoding with a CNN Minimizing the Rate-Distortion-Time Cost**](https://ieeexplore.ieee.org/abstract/document/9418763) [CODE]()

- **`[TMM 2020]`** [**Fast Multi-Type Tree Partitioning for Versatile Video Coding Using a Lightweight Neural Network**](https://ieeexplore.ieee.org/document/9277576) [CODE]()
- **`[ICIP 2020]`** [**CNN Oriented Complexity Reduction Of VVC Intra Encoder**](https://ieeexplore.ieee.org/abstract/document/9190797) [CODE]()
- **`[ICIP 2020]`** [**Chroma Intra Prediction With Attention-Based CNN Architectures**](https://doi.org/10.1109/ICIP40778.2020.9191050) [CODE]()
- **`[ICIP 2020]`** [**Multi-Mode Intra Prediction for Learning-Based Image Compression**](https://doi.org/10.1109/ICIP40778.2020.9191108) [CODE]()
- **`[ICIP 2020]`** [**Optimized Convolutional Neural Networks for Video Intra Prediction**](https://doi.org/10.1109/ICIP40778.2020.9190713) [CODE]()
- **`[VCIP 2020]`** [**Introducing Latent Space Correlation to Conditional Autoencoders for Intra Prediction**](https://doi.org/10.1109/VCIP49819.2020.9301806) [CODE]()
- **`[VCIP 2020]`** [**Fully Neural Network Mode Based Intra Prediction of Variable Block Size**](https://doi.org/10.1109/VCIP49819.2020.9301842) [CODE]()
- **`[ISCAS 2020]`** [**Fast partitioning decision scheme for versatile video coding intra-frame prediction**](https://ieeexplore.ieee.org/document/9180980) [CODE]()
- **`[TCSVT 2020]`** [**Low complexity CTU partition structure decision and fast intra mode decision for versatile video coding**](https://ieeexplore.ieee.org/abstract/document/8664144) [CODE]()
- **`[Multimedia Systems 2020]`** [**Fast CU partition decision for H.266/VVC based on the improved DAG-SVM classifier model**](https://link.springer.com/article/10.1007/s00530-020-00688-z) [CODE]()
- **`[Multimedia Tools and Applications 2020]`** [**A fast CU size decision algorithm for VVC intra prediction based on support vector machine**](https://link.springer.com/article/10.1007/s11042-020-09401-8) [CODE]()



- **`[TBC 2019]`** [**Fast HEVC Intra Mode Decision Based on RDO Cost Prediction**](https://ieeexplore.ieee.org/abstract/document/8401532) [CODE]()
- **`[VCIP 2019]`** [**Adaptive CU Split Decision with Pooling-variable CNN for VVC Intra Encoding**](https://ieeexplore.ieee.org/abstract/document/8965679) [CODE]()
- **`[TCSVT 2019]`** [**Video Compression Using Generalized Binary Partitioning, Trellis Coded Quantization, Perceptually Optimized Encoding, and Advanced Prediction and Transform Coding**](https://doi.org/10.1109/TCSVT.2019.2945918) [CODE]()
- **`[TCSVT 2019]`** [**Multi-scale Convolutional Neural Network Based Intra Prediction for Video Coding**](https://doi.org/10.1109/TCSVT.2019.2934681) [CODE]()
- **`[TCSVT 2019]`** [**CNN-based Intra-Prediction for Lossless HEVC**](https://doi.org/10.1109/TCSVT.2019.2940092) [CODE]()
- **`[TIP 2019]`** [**Context-adaptive neural network-based prediction for image compression**](https://doi.org/10.1109/TIP.2019.2934565) [CODE]()
- **`[TCSVT 2019]`** [**A deep convolutional neural network approach for complexity reduction on intra-mode HEVC**](https://ieeexplore.ieee.org/document/8361836) [CODE](https://github.com/wolverinn/HEVC-CU-depths-prediction-CNN)
- **`[TMM 2019]`** [**Generative Adversarial Network-Based Intra Prediction for Video Coding**](https://doi.org/10.1109/TMM.2019.2924591) [CODE]()
- **`[TMM 2019]`** [**Enhanced Intra Prediction for Video Coding by Using Multiple Neural Networks**](https://doi.org/10.1109/TMM.2019.2963620) [CODE]()
- **`[JVCIR 2019]`** [**Intra Mode Prediction for H. 266/FVC Video Coding based on Convolutional Neural Network**](https://doi.org/10.1016/j.jvcir.2019.102686) [CODE]()
- **`[ISCAS 2019]`** [**CNN-Based Bi-Prediction Utilizing Spatial Information for Video Coding**](https://doi.org/10.1109/ISCAS.2019.8702552) [CODE]()
- **`[ICASSP 2019]`** [**Nonlinear Prediction of Multidimensional Signals via Deep Regression with Applications to Image Coding**](https://doi.org/10.1109/ICASSP.2019.8683863) [CODE]()
- **`[ICASSP 2019]`** [**Convolutional neural networks for video intra prediction using cross-component adaptation**](https://doi.org/10.1109/ICASSP.2019.8682846) [CODE]()
- **`[DCC 2019]`** [**Deep learning based angular intra-prediction for lossless HEVC video coding**](https://doi.org/10.1109/DCC.2019.00091) [CODE]()
- **`[DCC 2019]`** [**Intra picture prediction for video coding with neural networks**](https://doi.org/10.1109/DCC.2019.00053) [CODE]()
- **`[ICIP 2019]`** [**Look-Ahead Prediction Based Coding Unit Size Pruning for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/8803421) [CODE]()


- **`[TIP 2018]`** [**Fully Connected Network-Based Intra Prediction for Image Coding**](https://ieeexplore.ieee.org/abstract/document/8319436) [CODE]()


- **`[ICME 2017]`** [**A deep convolutional neural network approach for complexity reduction on intra-mode HEVC**](https://ieeexplore.ieee.org/document/8019316) [CODE](https://github.com/wolverinn/HEVC-CU-depths-prediction-CNN)


- **`[PCS 2016]`** [**Deep learning-based intra prediction mode decision for HEVC**](https://ieeexplore.ieee.org/abstract/document/7906399) [CODE]()

### Frame Inter Prediction
### Transform Coding
### Quantization
### Entropy Coding
### Loop Filtering
### Rate-Distortion Optimization
### Complexity Control
- **`[ICIP 2020]`** [**Complexity Analysis Of Next-Generation VVC Encoding And Decoding**](https://ieeexplore.ieee.org/abstract/document/9190983) [CODE]()
Complexity Analysis Of Next-Generation VVC Encoding And Decoding
### Datasets
### Tools



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