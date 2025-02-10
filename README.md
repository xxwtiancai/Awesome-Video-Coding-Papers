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
- **`[VCIP 2024]`** [**Vectorized Angular Intra Prediction for Practical VVC Encoding**](https://ieeexplore.ieee.org/abstract/document/10849848) [CODE]()
- **`[Multimedia Tools and Applications 2024]`** [**Fast CU partition strategy based on texture and neighboring partition information for Versatile Video Coding Intra Coding**](https://link.springer.com/article/10.1007/s11042-023-16601-5) [CODE]()
- **`[TCSVT 2024]`** [**STRANet: soft-target and restriction-aware neural network for efficient VVC intra coding**](https://ieeexplore.ieee.org/abstract/document/10599317) [CODE](https://github.com/cppppp/STRANet)


- **`[ISCAS 2023]`** [**Ultra-Lightweight CNN Based Fast Intra Prediction for VVC Screen Content Coding**](https://doi.org/10.1109/ISCAS46773.2023.10181706) [CODE]()
- **`[TCSVT 2023]`** [**Machine Learning Based Efficient QT-MTT Partitioning Scheme for VVC Intra Encoders**](https://ieeexplore.ieee.org/abstract/document/10004946) [CODE](https://alexandretissier.github.io/QTMTT_VVC/)
- **`[TCSVT 2023]`** [**Deep Multi-Task Learning Based Fast Intra-Mode Decision for Versatile Video Coding**](https://ieeexplore.ieee.org/document/10083102) [CODE]()
- **`[TIP 2023]`** [**Partition Map Prediction for Fast Block Partitioning in VVC Intra-Frame Coding**](https://ieeexplore.ieee.org/abstract/document/10102791) [CODE](https://github.com/AolinFeng/PMP-VVC-TIP2023)
- **`[TIP 2023]`** [**Adaptive Chroma Prediction Based on Luma Difference for H.266/VVC**](https://ieeexplore.ieee.org/abstract/document/10316257) [CODE]()


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
- **`[ICMEW 2021]`** [**Open-Source: Attention-Based Neural Networks For Chroma Intra Prediction In Video Coding**](https://ieeexplore.ieee.org/document/9455958) [CODE](https://github.com/bbc/intra-chroma-attentionCNN)


- **`[TMM 2020]`** [**Fast Multi-Type Tree Partitioning for Versatile Video Coding Using a Lightweight Neural Network**](https://ieeexplore.ieee.org/document/9277576) [CODE]()
- **`[ICIP 2020]`** [**CNN Oriented Complexity Reduction Of VVC Intra Encoder**](https://ieeexplore.ieee.org/abstract/document/9190797) [CODE]()
- **`[ICIP 2020]`** [**Chroma Intra Prediction With Attention-Based CNN Architectures**](https://doi.org/10.1109/ICIP40778.2020.9191050) [CODE]()
- **`[ICIP 2020]`** [**Multi-Mode Intra Prediction for Learning-Based Image Compression**](https://doi.org/10.1109/ICIP40778.2020.9191108) [CODE]()
- **`[ICIP 2020]`** [**Optimized Convolutional Neural Networks for Video Intra Prediction**](https://doi.org/10.1109/ICIP40778.2020.9190713) [CODE]()
- **`[VCIP 2020]`** [**Introducing Latent Space Correlation to Conditional Autoencoders for Intra Prediction**](https://doi.org/10.1109/VCIP49819.2020.9301806) [CODE]()
- **`[VCIP 2020]`** [**Fully Neural Network Mode Based Intra Prediction of Variable Block Size**](https://doi.org/10.1109/VCIP49819.2020.9301842) [CODE]()
- **`[ISCAS 2020]`** [**Fast partitioning decision scheme for versatile video coding intra-frame prediction**](https://ieeexplore.ieee.org/document/9180980) [CODE]()
- **`[TCSVT 2020]`** [**Low complexity CTU partition structure decision and fast intra mode decision for versatile video coding**](https://ieeexplore.ieee.org/abstract/document/8664144) [CODE]()
- **`[TCSVT 2020]`** [**Deep Learning-Based Chroma Prediction for Intra Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/8664144) [CODE](https://ieeexplore.ieee.org/document/9247080)
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
- **`[ICMEW 2024]`** [**Adaptive Intra Period Size for Deep Learning-Based Screen Content Video Coding**](https://ieeexplore.ieee.org/abstract/document/10645479) [CODE](https://openi.pcl.ac.cn/OpenDatasets/PKU-SCV)
- **`[Pattern Recognition 2024]`** [**IBVC: Interpolation-driven B-frame video compression**](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002164) [CODE](https://github.com/ruhig6/IBVC)
- **`[TIP 2024]`** [**Spatio-Temporal Convolutional Neural Network for Enhanced Inter Prediction in Video Coding**](https://ieeexplore.ieee.org/document/10648618) [CODE]()
- **`[MMsys 2024]`** [**Inter-Frame Parallelization in an Open Optimized VVC Encoder**](https://dl.acm.org/doi/abs/10.1145/3625468.3647624) [CODE]()



- **`[AAAI 2023]`** [**Video Compression Artifact Reduction by Fusing Motion Compensation and Global Context in a Swin-CNN Based Parallel Architecture**](https://ojs.aaai.org/index.php/AAAI/article/view/25458) [CODE](https://github.com/WilliammmZ/AutoVQE)
- **`[ICASSP 2023]`** [**Learned Video Coding with Motion Compensation Mixture Model**](https://doi.org/10.1109/ICASSP49357.2023.10094757) [CODE]()
- **`[DCC 2023]`** [**Butterfly: Multiple Reference Frames Feature Propagation Mechanism for Neural Video Compression**](https://doi.org/10.1109/DCC55655.2023.00028) [CODE]()


- **`[TCSVT 2022]`** [**Neural Network-Based Enhancement to Inter Prediction for Video Coding**](https://doi.org/10.1109/TCSVT.2021.3063165) [CODE]()
- **`[TCSVT 2022]`** [**Deep Affine Motion Compensation Network for Inter Prediction in VVC**](https://doi.org/10.1109/TCSVT.2021.3107135) [CODE]()
- **`[TIP 2022]`** [**Neural Reference Synthesis for Inter Frame Coding**](https://doi.org/10.1109/TIP.2021.3134465) [CODE]()
- **`[VCIP 2022]`** [**Deep Reference Frame Interpolation based Inter Prediction Enhancement for Versatile Video Coding**](https://doi.org/10.1109/VCIP56404.2022.10008890) [CODE]()
- **`[VCIP 2022]`** [**Deep Reference Frame Interpolation based Inter Prediction Enhancement for Versatile Video Coding**](https://doi.org/10.1109/VCIP56404.2022.10008890) [CODE]()
- **`[ICIP 2022]`** [**Intra-Inter Prediction for Versatile Video Coding Using a Residual Convolutional Neural Network**](https://ieeexplore.ieee.org/document/9897324) [CODE]()


- **`[DCC 2021]`** [**Bi-Prediction Enhancement with Deep Frame Prediction Network for Versatile Video Coding**](https://doi.org/10.1109/DCC50243.2021.00054) [CODE]()
- **`[DCC 2021]`** [**Deformable Convolution Network based Invertibility-Driven Interpolation Filter for HEVC**](https://doi.org/10.1109/DCC50243.2021.00069) [CODE]()
- **`[ICIP 2021]`** [**Deep Video Compression for Interframe Coding**](https://doi.org/10.1109/ICIP42928.2021.9506275) [CODE]()
- **`[VCIP 2021]`** [**Deep Inter Prediction via Reference Frame Interpolation for Blurry Video Coding**](https://doi.org/10.1109/VCIP53242.2021.9675429) [CODE]()
- **`[PCS 2021]`** [**Switchable Motion Models for Non-Block-Based Inter Prediction in Learning-Based Video Coding**](https://doi.org/10.1109/PCS50896.2021.9477475) [CODE]()
- **`[ICCV 2021]`** [**Extending Neural P-frame Codecs for B-frame Coding**](https://doi.org/10.1109/ICCV48922.2021.00661) [CODE]()
- **`[TCSVT 2021]`** [**Motion Vector Coding and Block Merging in the Versatile Video Coding Standard**](https://ieeexplore.ieee.org/document/9502124) [CODE]()
- **`[TCSVT 2021]`** [**Subblock-Based Motion Derivation and Inter Prediction Refinement in the Versatile Video Coding Standard**](https://ieeexplore.ieee.org/document/9499051) [CODE]()

- **`[TCSVT 2020]`** [**Convolutional Neural Network Based Bi-prediction Utilizing Spatial and Temporal Information in Video Coding**](https://doi.org/10.1109/TCSVT.2019.2954853) [CODE]()
- **`[TCSVT 2020]`** [**A Robust Quality Enhancement Method Based on Joint Spatial-Temporal Priors for Video Coding**](https://doi.org/10.1109/TCSVT.2020.3019919) [CODE]()
- **`[TCSVT 2020]`** [**Compression Priors Assisted Convolutional Neural Network for Fractional Interpolation**](https://doi.org/10.1109/TCSVT.2020.3011197) [CODE]()
- **`[TCSVT 2020]`** [**Deep Network-Based Frame Extrapolation With Reference Frame Alignment**](https://doi.org/10.1109/TCSVT.2020.2995243) [CODE]()
- **`[TCSVT 2020]`** [**A Distortion-Aware Multi-task Learning Framework for Fractional Interpolation in Video Coding**](https://doi.org/10.1109/TCSVT.2020.3028330) [CODE]()
- **`[TCSVT 2020]`** [**Neural Video Coding Using Multiscale Motion Compensation and Spatiotemporal Context Model**](https://doi.org/10.1109/TCSVT.2020.3035680) [CODE]()
- **`[TIP 2020]`** [**Optical Flow Based Co-Located Reference Frame for Video Compression**](https://doi.org/10.1109/TIP.2020.3014723) [CODE]()
- **`[CVPR Workshops 2020]`** [**Joint Motion and Residual Information Latent Representation for P-Frame Coding**](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w7/da_Silva_Joint_Motion_and_Residual_Information_Latent_Representation_for_P-Frame_Coding_CVPRW_2020_paper.pdf) [CODE]()
- **`[CVPR Workshops 2020]`** [**P-Frame Coding Proposal by NCTU: Parametric Video Prediction Through Backprop-Based Motion Estimation**](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w7/Ho_P-Frame_Coding_Proposal_by_NCTU_Parametric_Video_Prediction_Through_Backprop-Based_CVPRW_2020_paper.pdf) [CODE]()
- **`[ISCAS 2020]`** [**Memory-Augmented Auto-Regressive Network for Frame Recurrent Inter Prediction**](https://doi.org/10.1109/ISCAS45731.2020.9180452) [CODE]()
- **`[ICIP 2020]`** [**Interpreting CNN for Low Complexity Learned Sub-pixel Motion Compensation in Video Coding**](https://doi.org/10.1109/ICIP40778.2020.9191193) [CODE]()
- **`[ICIP 2020]`** [**Deep Virtual Reference Frame Generation For Multiview Video Coding**](https://doi.org/10.1109/ICIP40778.2020.9191112) [CODE]()
- **`[VCIP 2020]`** [**Deep Inter Coding with Interpolated Reference Frame for Hierarchical Coding Structure**](https://doi.org/10.1109/VCIP49819.2020.9301769) [CODE]()
- **`[ICMEW 2020]`** [**CNN-Based Inter Prediction Refinement for AVS3**](https://ieeexplore.ieee.org/document/9106017) [CODE]()



- **`[TCSVT 2019]`** [**Deep frame prediction for video coding**](https://doi.org/10.1109/TCSVT.2019.2924657) [CODE]()
- **`[TCSVT 2019]`** [**A Multi-Scale Position Feature Transform Network for Video Frame Interpolation**](https://doi.org/10.1109/TCSVT.2019.2939143) [CODE]()
- **`[TIP 2019]`** [**Three-Zone Segmentation-Based Motion Compensation for Video Compression**](https://doi.org/10.1109/TIP.2019.2910382) [CODE]()
- **`[TMM 2019]`** [**Deep reference generation with multi-domain hierarchical constraints for inter prediction**](https://doi.org/10.1109/TMM.2019.2961504) [CODE]()
- **`[ISCAS 2019]`** [**Switch mode based deep fractional interpolation in video coding**](https://doi.org/10.1109/ISCAS.2019.8702522) [CODE]()
- **`[ISCAS 2019]`** [**CNN-Based Bi-Prediction Utilizing Spatial Information for Video Coding**](https://ieeexplore.ieee.org/document/8702552) [CODE]()
- **`[DCC 2019]`** [**Deep frame interpolation for video compression**](https://doi.org/10.1109/DCC.2019.00068) [CODE]()
- **`[ICIP 2019]`** [**Advanced cnn based motion compensation fractional interpolation**](https://doi.org/10.1109/ICIP.2019.8804199) [CODE]()
- **`[PCS 2019]`** [**HEVC Inter Coding using Deep Recurrent Neural Networks and Artificial Reference Pictures**](https://doi.org/10.1109/PCS48520.2019.8954497) [CODE]()
- **`[PCS 2019]`** [**An Extended Skip Strategy for Inter Prediction**](https://doi.org/10.1109/PCS48520.2019.8954532) [CODE]()
- **`[PCS 2019]`** [**Recent development of AVS video coding standard: AVS3**](https://ieeexplore.ieee.org/document/8954503) [CODE]()


- **`[TCSVT 2018]`** [**Enhanced Bi-Prediction With Convolutional Neural Network for High-Efficiency Video Coding**](https://ieeexplore.ieee.org/document/8493529) [CODE]()
- **`[TCSVT 2018]`** [**Triple-Frame-Based Bi-Directional Motion Estimation for Motion-Compensated Frame Interpolation**](https://doi.org/10.1109/TCSVT.2018.2840842) [CODE]()
- **`[TCSVT 2018]`** [**Weighted Convolutional Motion-Compensated Frame Rate Up-Conversion Using Deep Residual Network**](https://doi.org/10.1109/TCSVT.2018.2885564) [CODE]()



### Transform Coding
- **`[TBC 2024]`** [**Fast Transform Kernel Selection Based on Frequency Matching and Probability Model for AV1**](https://ieeexplore.ieee.org/abstract/document/10479536) [CODE]()
- **`[PCS 2024]`** [**Nonlinear Transform Coding for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10566436) [CODE]()
- **`[DCC 2024]`** [**Decoder-side Secondary Transform Derivation for Video Coding beyond AVS3**](https://ieeexplore.ieee.org/abstract/document/10533834) [CODE]()
- **`[DCC 2024]`** [**Construction of Fast Data-driven Transforms for Image Compression via Multipath Coordinate Descent on Orthogonal Matrix Manifold**](https://ieeexplore.ieee.org/abstract/document/10533826) [CODE]()
- **`[ICASSP 2024]`** [**Adaptive Secondary Transform Sets for Video Coding Beyond AV1**](https://ieeexplore.ieee.org/abstract/document/10447533) [CODE]()
- **`[ICCCAS 2024]`** [**A Fast Transform Algorithm for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/9825469) [CODE]()
- **`[ACM TOM 2024]`** [**Graph Based Cross-Channel Transform for Color Image Compression**](https://dl.acm.org/doi/abs/10.1145/3631710) [CODE]()
- **`[TCSVT 2024]`** [**Revisiting All-Zero Block Detection for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10378712) [CODE]()
- **`[TCSVT 2023]`** [**Learning-Based Early Transform Skip Mode Decision for VVC Screen Content Coding**](https://ieeexplore.ieee.org/abstract/document/10068735) [CODE]()
- **`[ISCAS 2023]`** [**High-Throughput Design for a Multi-Size DCT-II Targeting the AV1 Encoder**](https://ieeexplore.ieee.org/abstract/document/10181828) [CODE]()
- **`[ISCAS 2022]`** [**A High-Throughput Design for the H.266/VVC Low-Frequency Non-Separable Transform**](https://ieeexplore.ieee.org/abstract/document/9937597) [CODE]()
- **`[IEEE Journal of Selected Topics in Signal Processing 2021]`** [**Nonlinear Transform Coding**](https://ieeexplore.ieee.org/document/9242247) [CODE]()
- **`[ICIP 2021]`** [**Machine-Learning Based Secondary Transform for Improved Image Compression in JPEG2000**](https://doi.org/10.1109/ICIP42928.2021.9506122) [CODE]()
- **`[DCC 2021]`** [**Graph Based Transforms based on Graph Neural Networks for Predictive Transform Coding**](https://doi.org/10.1109/DCC50243.2021.00079) [CODE]()
- **`[PCS 2021]`** [**Combined neural network-based intra prediction and transform selection**](https://doi.org/10.1109/PCS50896.2021.9477455) [CODE]()
- **`[TCSVT 2021]`** [**Transform Coding in the VVC Standard**](https://ieeexplore.ieee.org/abstract/document/9449858) [CODE]()
- **`[ICIP 2020]`** [**Augmenting JPEG2000 With Wavelet Coefficient Prediction**](https://doi.org/10.1109/ICIP40778.2020.9190969) [CODE]()
- **`[VCIP 2020]`** [**Deep Learning-Based Nonlinear Transform for HEVC Intra Coding**](https://doi.org/10.1109/VCIP49819.2020.9301790) [CODE]()
- **`[PCS 2019]`** [**Low frequency non-separable transform**](https://ieeexplore.ieee.org/document/8954507) [CODE]()



### Quantization
- **`[TBC 2023]`** [**Complexity-Efficient Dependent Quantization for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10093119) [CODE]()

- **`[ICIP 2022]`** [**Learning Frequency-Specific Quantization Scaling in VVC for Standard-Compliant Task-Driven Image Coding**](https://doi.org/10.1109/ICIP46576.2022.9897987) [CODE](https://github.com/FAU-LMS/VCM_scaling_lists)

- **`[DCC 2021]`** [**JQF: Optimal JPEG Quantization Table Fusion by Simulated Annealing on Texture Images and Predicting Textures**](https://doi.org/10.1109/DCC50243.2021.00041) [CODE]()

- **`[VCIP 2020]`** [**Towards Quantized DCT Coefficients Restoration for Compressed Images**](https://doi.org/10.1109/VCIP49819.2020.9301794) [CODE]()
- **`[DCC 2020]`** [**Deep Learning-based Image Compression with Trellis Coded Quantization**](https://doi.org/10.1109/DCC47342.2020.00009) [CODE]()
- **`[ECCV 2020]`** [**Task-Aware Quantization Network for JPEG Image Compression**](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650307.pdf) [CODE]()
- **`[DCC 2019]`** [**Hard-Decision Quantization Algorithm Based on Deep Learning in Intra Video Coding**](https://doi.org/10.1109/DCC.2019.00119) [CODE]()

### Entropy Coding
- **`[IET Image Processing 2024]`** [**Dynamic estimator selection for double-bit-range estimation in VVC CABAC entropy coding**](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12980) [CODE]()
- **`[TCSVT 2021]`** [**Quantization and Entropy Coding in the Versatile Video Coding (VVC) Standard**](https://ieeexplore.ieee.org/abstract/document/9399502) [CODE]()
- **`[TCSVT 2019]`** [**Convolutional Neural Network-Based Arithmetic Coding for HEVC Intra-Predicted Residues**](https://ieeexplore.ieee.org/document/8756025) [CODE]()


### In-Loop Filtering
- **`[DCC 2024]`** [**Residual Block Fusion in Low Complexity Neural Network-Based In-loop Filtering for Video Compression**](https://ieeexplore.ieee.org/abstract/document/10533825) [CODE]()

- **`[ICIP 2024]`** [**IN-Loop Filter for Object Mask Coding in Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10647608) [CODE]()
- **`[PCS 2024]`** [**Swin Transformer-Based In-Loop Filter for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10566453) [CODE]()
- **`[ACM TOM 2024]`** [**A Reconfigurable Framework for Neural Network Based Video In-Loop Filtering**](https://dl.acm.org/doi/abs/10.1145/3640467) [CODE]()
- **`[VCIP 2024]`** [**PFT-ILF: In-loop Filter with Partition Feature Transform for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10849854) [CODE]()
- **`[TIP 2024]`** [**Combining Progressive Rethinking and Collaborative Learning: A Deep Framework for In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/9394795) [CODE](https://dezhao-wang.github.io/PRN-v2/)
- **`[VCIP 2024]`** [**Fast Adaptive Loop Filter Algorithm Based on the Optimization of Class Merging**](https://ieeexplore.ieee.org/abstract/document/10849852) [CODE]()
- **`[VCIP 2024]`** [**In-Loop Filtering via Trained Look-Up Tables**](https://ieeexplore.ieee.org/abstract/document/10849824) [CODE]()
- **`[ICIP 2024]`** [**IN-Loop Filter for Object Mask Coding in Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10647608) [CODE]()
- **`[ICIP 2024]`** [**NN-Based In-Loop Filtering With Inputs Transformed**](https://ieeexplore.ieee.org/abstract/document/10647805) [CODE]()
- **`[TCSVT 2024]`** [**Neural Network Based Multi-Level In-Loop Filtering for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10577261) [CODE]()
- **`[TCSVT 2024]`** [**Screen Content-Aware Video Coding through Non-Local Model embedded with Intra-Inter In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10704652) [CODE]()
- **`[TCSVT 2024]`** [**Area-Efficient Processing Elements-Based Adaptive Loop Filter Architecture With Optimized Memory for VVC**](https://ieeexplore.ieee.org/abstract/document/10136842) [CODE]()
- **`[Pattern Recognition 2024]`** [**DWT-SALF: Subband Adaptive Neural Network Based In-Loop Filter for VVC Using Cyclic DWT**](https://link.springer.com/chapter/10.1007/978-3-031-78395-1_14) [CODE]()
- **`[Pattern Recognition 2024]`** [**Progressive Learning Based on QP Distance for Enhancing HOP In-Loop Filter**](https://link.springer.com/chapter/10.1007/978-3-031-78395-1_10) [CODE]()
- **`[ISCAS 2023]`** [**Towards Next Generation Video Coding: from Neural Network Based Predictive Coding to In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10181462) [CODE]()
- **`[ICME 2023]`** [**Meta-ILF: In-Loop Filter with Customized Weights For VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10219609) [CODE]()
- **`[ICML 2023]`** [**Low Complexity Neural Network-Based In-loop Filtering with Decomposed Split Luma-Chroma Model for Video Compression**](https://openreview.net/pdf?id=ZkkjPbx5KG) [CODE]()
- **`[TCSVT 2023]`** [**Lightweight Multiattention Recursive Residual CNN-Based In-Loop Filter Driven by Neuron Diversity**](https://ieeexplore.ieee.org/abstract/document/10109188) [CODE]()
- **`[TCSVT 2023]`** [**Deep In-Loop Filtering via Multi-Domain Correlation Learning and Partition Constraint for Multiview Video Coding**](https://ieeexplore.ieee.org/abstract/document/9915617) [CODE]()
- **`[ICIG 2023]`** [**Content-Adaptive Block Clustering for Improving VVC Adaptive Loop Filtering**](https://link.springer.com/chapter/10.1007/978-3-031-46311-2_34) [CODE]()
- **`[ACM TOM 2023]`** [**iDAM: Iteratively Trained Deep In-loop Filter with Adaptive Model Selection**](https://dl.acm.org/doi/abs/10.1145/3529107) [CODE]()
- **`[ICIP 2023]`** [**Lightweight CNN-Based in-Loop Filter for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10223094) [CODE]()
- **`[TMM 2023]`** [**Joint Rate-Distortion Optimization for Video Coding and Learning-Based In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10216325) [CODE]()
- **`[DCC 2023]`** [**A Low Complexity Convolutional Neural Network with Fused CP Decomposition for In-Loop Filtering in Video Coding**](https://doi.org/10.1109/DCC55655.2023.00032) [CODE]()
- **`[IFTC 2023]`** [**Temporal Dependency-Oriented Deep In-Loop Filter for VVC**](https://link.springer.com/chapter/10.1007/978-981-97-3626-3_9) [CODE]()
- **`[TMM 2022]`** [**Textural and Directional Information Based Offset In-Loop Filtering in AVS3**](https://ieeexplore.ieee.org/abstract/document/9868111) [CODE]()
- **`[ICASSP 2022]`** [**Low-Complexity Multi-Model CNN in-Loop Filter for AVS3**](https://ieeexplore.ieee.org/abstract/document/9746146) [CODE]()
- **`[CVPR 2022]`** [**Perceptual In-Loop Filter for Image and Video Compression**](https://openaccess.thecvf.com/content/CVPR2022W/CLIC/html/Wang_Perceptual_In-Loop_Filter_for_Image_and_Video_Compression_CVPRW_2022_paper.html) [CODE]()
- **`[ACM TOM 2022]`** [**NR-CNN: Nested-Residual Guided CNN In-loop Filtering for Video Coding**](https://dl.acm.org/doi/abs/10.1145/3502723) [CODE]()
- **`[ICME 2022]`** [**Neural Network Based in-Loop Filter with Constrained Memory**](https://ieeexplore.ieee.org/abstract/document/9859910) [CODE]()
- **`[TIP 2022]`** [**Deformable Wiener Filter for Future Video Coding**](https://ieeexplore.ieee.org/document/9948451) [CODE]()
- **`[TIP 2022]`** [**QA-filter: A QP-adaptive convolutional neural network filter for video coding**](https://ieeexplore.ieee.org/document/9750961) [CODE]()
- **`[TCE 2022]`** [**Adaptive Loop Filter Hardware Design for 4K ASIC VVC Decoders**](https://ieeexplore.ieee.org/document/9691469) [CODE]()
- **`[VCIP 2022]`** [**Multi-stage Locally and Long-range Correlated Feature Fusion for Learned In-loop Filter in VVC**](https://doi.org/10.1109/VCIP56404.2022.10008834) [CODE]()
- **`[ICIP 2022]`** [**Switchable CNN-Based Same-Resolution and Super-Resolution In-Loop Restoration for Next Generation Video Codecs**](https://ieeexplore.ieee.org/document/9897763) [CODE]()
- **`[ICIP 2022]`** [**Non-Separable Filtering with Side-Information and Contextually-Designed Filters for Next Generation Video Codecs**](https://ieeexplore.ieee.org/document/9898053) [CODE]()
- **`[ICIP 2022]`** [**Adaptive Loop Filter with a CNN-Based Classification**](https://ieeexplore.ieee.org/document/9897666) [CODE]()
- **`[PCS 2022]`** [**Efficient HW Design of Adaptive Loop Filter for 4k ASIC VVC Encoder**](https://ieeexplore.ieee.org/abstract/document/10018078) [CODE]()
- **`[PCS 2022]`** [**Multi-Stage Spatial and Frequency Feature Fusion using Transformer in CNN-Based In-Loop Filter for VVC**](https://doi.org/10.1109/PCS56426.2022.10017998) [CODE]()
- **`[PCS 2022]`** [**Optimize neural network based in-loop filters through iterative training**](https://doi.org/10.1109/PCS56426.2022.10018057) [CODE]()
- **`[PCS 2022]`** [**Generalized deblocking filter for AVM**](https://ieeexplore.ieee.org/abstract/document/10018081) [CODE]()
- **`[ISCAS 2022]`** [**An Attention Based CNN with Temporal Hierarchical Deployment for AVS3 Inter In-loop Filtering**](https://doi.org/10.1109/ISCAS48785.2022.9937718) [CODE]()
- **`[ISCAS 2022]`** [**Complexity Reduction of Learned In-Loop Filtering in Video Coding**](https://doi.org/10.1109/ISCAS48785.2022.9937777) [CODE]()
- **`[ISCAS 2022]`** [**A QP-adaptive Mechanism for CNN-based Filter in Video Coding**](https://doi.org/10.1109/ISCAS48785.2022.9937233) [CODE]()
- **`[ISCAS 2022]`** [**Joint Luma and Chroma Multi-Scale CNN In-loop Filter for Versatile Video Coding**](https://doi.org/10.1109/ISCAS48785.2022.9937419) [CODE]()
- **`[DCC 2022]`** [**Joint Rate Distortion Optimization with CNN-based In-Loop Filter For Hybrid Video Coding**](https://doi.org/10.1109/DCC52660.2022.00073) [CODE]()
- **`[DCC 2022]`** [**Parametric Non-local In-loop Filter for Future Video Coding**](https://doi.org/10.1109/DCC52660.2022.00085) [CODE]()
- **`[DCC 2022]`** [**An Improved Multi-reference Frame Loop Filter Algorithm Based on Transformer for VVC**](https://doi.org/10.1109/DCC52660.2022.00078) [CODE]()
- **`[JVCIR 2022]`** [**PTR-CNN for in-loop filtering in video coding**](https://doi.org/10.1016/j.jvcir.2022.103615) [CODE]()
- **`[TCSVT 2022]`** [**One-for-all: An efficient variable convolution neural network for in-loop filter of VVC**](https://ieeexplore.ieee.org/document/9455379) [CODE]()
- **`[SPIC 2021]`** [**Deep learning based HEVC in-loop filter and noise reduction**](https://www.sciencedirect.com/science/article/abs/pii/S0923596521001946) [CODE]()
- **`[SPIC 2021]`** [**A progressive CNN in-loop filtering approach for inter frame coding**](https://www.sciencedirect.com/science/article/abs/pii/S0923596521000321) [CODE]()
- **`[TIP 2021]`** [**Adaptive deep reinforcement learning-based in-loop filter for VVC**](https://ieeexplore.ieee.org/document/9446562) [CODE]()
- **`[TCSVT 2021]`** [**VVC In-Loop Filters**](https://ieeexplore.ieee.org/abstract/document/9399506) [CODE]()
- **`[CVPR 2021]`** [**Deep Learning Based Spatial-Temporal In-Loop Filtering for Versatile Video Coding**](https://openaccess.thecvf.com/content/CVPR2021W/CLIC/papers/Pham_Deep_Learning_Based_Spatial-Temporal_In-Loop_Filtering_for_Versatile_Video_Coding_CVPRW_2021_paper.pdf) [CODE]()
- **`[DCC 2021]`** [**Multi-Density Convolutional Neural Network for In-Loop Filter in Video Coding**](https://doi.org/10.1109/DCC50243.2021.00010) [CODE]()
- **`[DCC 2021]`** [**An Efficient QP Variable Convolutional Neural Network Based In-loop Filter for Intra Coding**](https://doi.org/10.1109/DCC50243.2021.00011) [CODE]()
- **`[DCC 2021]`** [**3D-CVQE: An Effective 3D-CNN Quality Enhancement for Compressed Video Using Limited Coding Information**](https://doi.org/10.1109/DCC50243.2021.00050) [CODE]()
- **`[DCC 2021]`** [**Spatial-Temporal Fusion Convolutional Neural Network for Compressed Video Enhancement in HEVC**](https://doi.org/10.1109/DCC50243.2021.00066) [CODE]()
- **`[DCC 2021]`** [**Video Enhancement Network Based on Max-Pooling and Hierarchical Feature Fusion**](https://doi.org/10.1109/DCC50243.2021.00067) [CODE]()
- **`[PCS 2021]`** [**Bonnineau C, Hamidouche W, Travers J F, et al. Multitask Learning for VVC Quality Enhancement and Super-Resolution**](https://doi.org/10.1109/PCS50896.2021.9477492) [CODE]()
- **`[PCS 2021]`** [**Bordes P, Galpin F, Dumas T, et al. Revisiting the Sample Adaptive Offset post-filter of VVC with Neural-Networks**](https://doi.org/10.1109/PCS50896.2021.9477457) [CODE]()
- **`[PCS 2021]`** [**Nasiri F, Hamidouche W, Morin L, et al. Model Selection CNN-based VVC QualityEnhancement**](https://doi.org/10.1109/PCS50896.2021.9477473) [CODE]()
- **`[PCS 2021]`** [**Convolutional neural network-based post-filtering for compressed YUV420 images and video**](https://doi.org/10.1109/PCS50896.2021.9477486) [CODE]()
- **`[VCIP 2021]`** [**Distortion-based Neural Network for Compression Artifacts Reduction in VVC**](https://doi.org/10.1109/VCIP53242.2021.9675413) [CODE]()
- **`[ICIP 2021]`** [**Convolutional Neural Network Based In-Loop Filter For VVC Intra Coding**](https://doi.org/10.1109/ICIP42928.2021.9506027) [CODE]()
- **`[DCC 2021]`** [**Flow-Guided Temporal-Spatial Network for HEVC Compressed Video Quality Enhancement**](https://doi.org/10.1109/DCC50243.2021.00064) [CODE]()
- **`[DCC 2021]`** [**Densely Connected Unit based Loop Filter for Short Video Coding**](https://doi.org/10.1109/DCC50243.2021.00076) [CODE]()
- **`[ICIP 2021]`** [**Enhancing quality for VVC compressed videos by jointly exploiting spatial details and temporal structure**](https://doi.org/10.1109/ICIP.2019.8804469) [CODE]()
- **`[ICIP 2021]`** [**Partition tree guided progressive rethinking network for in-loop filtering of HEVC**](https://doi.org/10.1109/ICIP.2019.8803253) [CODE]()
- **`[ICIP 2021]`** [**Deep Enhancement for 3D HDR Brain Image Compression**](https://doi.org/10.1109/ICIP.2019.8803781) [CODE]()
- **`[ICIP 2021]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Development Of New Fractal And Non-Fractal Deep Residual Networks For Deblocking Of Jpeg Decompressed Images**](https://doi.org/10.1109/ICIP40778.2020.9191030) [CODE]()
- **`[ICIP 2021]`** [**Xia J, Wen J. Asymmetric Convolutional Residual Network for AV1 Intra in-Loop Filtering**](https://doi.org/10.1109/ICIP40778.2020.9190743) [CODE]()
- **`[ICIP 2021]`** [**Li B, Liang J, Wang Y. Compression Artifact Removal with Stacked Multi-Context Channel-Wise Attention Network**](https://doi.org/10.1109/ICIP.2019.8803448) [CODE]()
- **`[ICIP 2021]`** [**Kim T, Lee H, Son H, et al. SF-CNN: A Fast Compression Artifacts Removal via Spatial-To-Frequency Convolutional Neural Networks**](https://doi.org/10.1109/ICIP.2019.8803503) [CODE]()
- **`[ICIP 2021]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Deep Jpeg Image Deblocking Using Residual Maxout Units**](https://doi.org/10.1109/ICIP.2019.8803374) [CODE]()
- **`[ICME 2020]`** [**Multi-Gradient Convolutional Neural Network Based In-Loop Filter For Vvc**](https://ieeexplore.ieee.org/document/9102826) [CODE]()
- **`[TCSVT 2020]`** [**A switchable deep learning approach for in-loop filtering in video coding**](https://ieeexplore.ieee.org/document/8801877) [CODE]()
- **`[VCIP 2020]`** [**Ma H, Liu D, Wu F. Improving Compression Artifact Reduction via End-to-End Learning of Side Information**](https://doi.org/10.1109/VCIP49819.2020.9301805) [CODE]()
- **`[VCIP 2020]`** [**A Mixed Appearance-based and Coding Distortion-based CNN Fusion Approach for In-loop Filtering in Video Coding**](https://ieeexplore.ieee.org/abstract/document/9301895) [CODE]()
- **`[VCIP 2020]`** [**Nasiri F, Hamidouche W, Morin L, et al. Prediction-Aware Quality Enhancement of VVC Using CNN**](https://doi.org/10.1109/VCIP49819.2020.9301884) [CODE]()
- **`[VCIP 2020]`** [**Yue J, Gao Y, Li S, et al. A Mixed Appearance-based and Coding Distortion-based CNN Fusion Approach for In-loop Filtering in Video Coding**](https://doi.org/10.1109/VCIP49819.2020.9301895) [CODE]()
- **`[VCIP 2019]`** [**Yu Y, Yang X, Chen J, et al. Deep Learning Based In-Loop Filter for Video Coding**](https://doi.org/10.1109/VCIP47243.2019.8965980) [CODE]()
- **`[ICIP 2020]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Development Of New Fractal And Non-Fractal Deep Residual Networks For Deblocking Of Jpeg Decompressed Images**](https://doi.org/10.1109/ICIP40778.2020.9191030) [CODE]()
- **`[ICIP 2020]`** [**Li B, Liang J, Wang Y. Compression Artifact Removal with Stacked Multi-Context Channel-Wise Attention Network**](https://doi.org/10.1109/ICIP40778.2020.9191106) [CODE]()
- **`[ICIP 2020]`** [**Kim T, Lee H, Son H, et al. SF-CNN: A Fast Compression Artifacts Removal via Spatial-To-Frequency Convolutional Neural Networks**](https://doi.org/10.1109/ICIP40778.2020.9191106) [CODE]()
- **`[ICIP 2020]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Deep Jpeg Image Deblocking Using Residual Maxout Units**](https://doi.org/10.1109/ICIP40778.2020.9191030) [CODE]()
- **`[ICIP 2019]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Development Of New Fractal And Non-Fractal Deep Residual Networks For Deblocking Of Jpeg Decompressed Images**](https://doi.org/10.1109/ICIP.2019.8803374) [CODE]()
- **`[ICIP 2019]`** [**Li B, Liang J, Wang Y. Compression Artifact Removal with Stacked Multi-Context Channel-Wise Attention Network**](https://doi.org/10.1109/ICIP.2019.8803448) [CODE]()
- **`[ICIP 2019]`** [**Kim T, Lee H, Son H, et al. SF-CNN: A Fast Compression Artifacts Removal via Spatial-To-Frequency Convolutional Neural Networks**](https://doi.org/10.1109/ICIP.2019.8803503) [CODE]()
- **`[ICIP 2019]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Deep Jpeg Image Deblocking Using Residual Maxout Units**](https://doi.org/10.1109/ICIP.2019.8803374) [CODE]()
- **`[ICIP 2019]`** [**Gao S, Xiong Z. Deep Enhancement for 3D HDR Brain Image Compression**](https://doi.org/10.1109/ICIP.2019.8803781) [CODE]()
- **`[ICIP 2019]`** [**Meng X, Deng X, Zhu S, et al. Enhancing quality for VVC compressed videos by jointly exploiting spatial details and temporal structure**](https://doi.org/10.1109/ICIP.2019.8804469) [CODE]()
- **`[ICIP 2019]`** [**Partition tree guided progressive rethinking network for in-loop filtering of HEVC**](https://doi.org/10.1109/ICIP.2019.8803253) [CODE]()
- **`[ICIP 2019]`** [**Esmaeilzehi A, Ahmad M O, Swamy M N S. Deep Jpeg Image Deblocking Using Residual Maxout Units**](https://doi.org/10.1109/ICIP.2019.880337







### Rate-Distortion Optimization
- **`[TIP 2023]`** [**Toward the Achievable Rate-Distortion Bound of VVC Intra Coding: A Beam Search-Based Joint Optimization Scheme**](https://ieeexplore.ieee.org/abstract/document/10304583) [CODE]()
- **`[TMM 2023]`** [**Joint Rate-Distortion Optimization for Video Coding and Learning-Based In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10216325) [CODE]()
- **`[TBC 2023]`** [**A Quality-of-Experience-Aware Framework for Versatile Video Coding-Based Video Transmission**](https://ieeexplore.ieee.org/abstract/document/9990901) [CODE]()
- **`[TIP 2020]`** [**Rate distortion optimization: A joint framework and algorithms for random access hierarchical video coding**](https://ieeexplore.ieee.org/document/9216484) [CODE]()





### Rate Control
- **`[JVCIP 2024]`** [**Exploring the rate-distortion-complexity optimization in neural image compression☆**](https://www.sciencedirect.com/science/article/abs/pii/S1047320324002505) [CODE]()
- **`[ISCAS 2024]`** [**Rate Control for Slimmable Video Codec using Multilayer Perceptron**](https://ieeexplore.ieee.org/abstract/document/10558606) [CODE]()
- **`[TVT 2024]`** [**An Efficient Neural Network Based Rate Control for Intra-frame in Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10777536) [CODE]()
- **`[PCS 2024]`** [**Multi-Agent Reinforcement Learning based Bit Allocation for Gaming Video Coding**](https://ieeexplore.ieee.org/abstract/document/10566434) [CODE]()
- **`[TIP 2024]`** [**λ-Domain Rate Control via Wavelet-Based Residual Neural Network for VVC HDR Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10736428) [CODE](https://github.com/TJU-Videocoding/WRNN.git)
- **`[TCSVT 2024]`** [**Recent Advances in Rate Control: From Optimization to Implementation and Beyond**](https://ieeexplore.ieee.org/abstract/document/10155441) [CODE]()
- **`[TCSVT 2024]`** [**Content-Adaptive Rate Control Method for User-Generated Content Videos**](https://ieeexplore.ieee.org/abstract/document/10734399) [CODE]()
- **`[TMM 2024]`** [**Content-Adaptive Rate-Distortion Modeling for Frame-Level Rate Control in Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/9190983) [CODE]()
- **`[TBC 2024]`** [**Spatial Coupling Strategy and Improved BFGS-Based Advanced Rate Control for VVC**](https://ieeexplore.ieee.org/abstract/document/10819255) [CODE]()

- **`[VCIP 2023]`** [**Reinforcement Learning-based Frame-level Bit Allocation for VVC**](https://ieeexplore.ieee.org/document/10402665) [CODE]()
- **`[MMSP 2023]`** [**Rate control for VVC intra coding with simplified cubic rate-distortion model**](https://ieeexplore.ieee.org/document/10337686) [CODE]()
- **`[TIP 2023]`** [**Joint decision tree and visual feature rate control optimization for VVC UHD coding**](https://ieeexplore.ieee.org/document/9979046) [CODE]()
- **`[TCSVT 2023]`** [**λ -domain VVC rate control based on Nash equilibrium**](https://ieeexplore.ieee.org/document/9996433) [CODE]()
- **`[TCSVT 2023]`** [**Neural Network Based Rate Control for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10082986) [CODE]()
- **`[TCSVT 2023]`** [**Efficient Rate Control in Versatile Video Coding With Adaptive Spatial–Temporal Bit Allocation and Parameter Updating**](https://ieeexplore.ieee.org/document/9963963) [CODE]()
- **`[TCSVT 2023]`** [**A CTU-Level Screen Content Rate Control for Low-Delay Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10041774) [CODE]()
- **`[TBC 2023]`** [**Precise Encoding Complexity Control for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/9829846) [CODE]()
- **`[VCIP 2023]`** [**Rate Control Optimization for Joint Geometry and Attribute Coding of LiDAR Point Clouds**](https://ieeexplore.ieee.org/abstract/document/10402779) [CODE]()
- **`[TIP 2023]`** [**SUR-Driven Video Coding Rate Control for Jointly Optimizing Perceptual Quality and Buffer Control**](https://ieeexplore.ieee.org/abstract/document/10266980) [CODE]()
- **`[Information Sciences 2023]`** [**Rate distortion optimization with adaptive content modeling for random-access versatile video coding**](https://www.sciencedirect.com/science/article/abs/pii/S0020025523009106?via%3Dihub) [CODE]()


- **`[MMSP 2022]`** [**Reinforcement Learning based Low Delay Rate Control for HEVC Region of Interest Coding**](https://ieeexplore.ieee.org/abstract/document/9950043) [CODE]()
- **`[TBC 2022]`** [**Optimum Quality Control Algorithm for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/9705090) [CODE]()
- **`[TIP 2022]`** [**Joint Decision Tree and Visual Feature Rate Control Optimization for VVC UHD Coding**](https://ieeexplore.ieee.org/abstract/document/9979046) [CODE]()
- **`[TBC 2022]`** [**Perceptual quality consistency oriented CTU level rate control for HEVC intra coding**](https://ieeexplore.ieee.org/document/9586556) [CODE]()
- **`[TIP 2022]`** [**Learning-Based Rate Control for Video-Based Point Cloud Compression**](https://ieeexplore.ieee.org/document/9720075) [CODE]()
- **`[ArXiv 2022]`** [**MuZero with Self-competition for Rate Control in VP9 Video Compression**](https://arxiv.org/abs/2202.06626) [CODE]()
- **`[ArXiv 2022]`** [**λ-domain VVC Rate Control Based on Game Theory**](https://arxiv.org/abs/2205.03595) [CODE]()
- **`[MIPR 2022]`** [**Machine-Learning Based High Efficiency Rate Control for AV1**](https://ieeexplore.ieee.org/abstract/document/9874608) [CODE]()

- **`[Multimedia Tools and Applications 2021]`** [**New bufferless rate control for high efficiency video coding**](https://link.springer.com/article/10.1007/s11042-021-11055-z) [CODE]()
- **`[ACM MM 2021]`** [**Game theory-driven rate control for 360-degree video coding**](https://dl.acm.org/doi/10.1145/3474085.3475657) [CODE]()
- **`[TIP 2021]`** [**Multi-Objective Optimization of Quality in VVC Rate Control for Low-Delay Video Coding**](https://ieeexplore.ieee.org/document/9405471) [CODE]()
- **`[TCC 2021]`** [**Hybrid Distortion-Based Rate-Distortion Optimization and Rate Control for H.265/HEVC**](https://ieeexplore.ieee.org/abstract/document/9377475) [CODE]()
- **`[DCC 2021]`** [**A Viewport-Adaptive Rate Control Approach for Omnidirectional Video Coding**](https://ieeexplore.ieee.org/abstract/document/9418788) [CODE]()
- **`[DCC 2021]`** [**A Dual-Critic Reinforcement Learning Framework for Frame-Level Bit Allocation in HEVC/H.265**](https://ieeexplore.ieee.org/abstract/document/9418757) [CODE](http://mapl.nctu.edu.tw/RL_Rate_Control/)(The code page may be out of date, you can access the code by mail me [xiongweixiao@foxmail.com](xiongweixiao@foxmail.com.))
- **`[ICME 2021]`** [**Machine learning-based rate distortion modeling for VVC/H.266 intra-frame**](https://ieeexplore.ieee.org/document/9428378) [CODE]()
- **`[TCSVT 2021]`** [**Rate control for predictive transform screen content video coding based on RANSAC**](https://ieeexplore.ieee.org/document/9306869) [CODE]()
- **`[TCSVT 2021]`** [**High Efficiency Rate Control for Versatile Video Coding Based on Composite Cauchy Distribution**](https://ieeexplore.ieee.org/document/9467266) [CODE]()
- **`[TBC 2021]`** [**A Bit Allocation Method Based on Inter-View Dependency and Spatio-Temporal Correlation for Multi-View Texture Video Coding**](https://ieeexplore.ieee.org/abstract/document/9234533) [CODE]()

- **`[TII 2021]`** [**Consistent Quality Oriented Rate Control in HEVC Via Balancing Intra and Inter Frame Coding**](https://ieeexplore.ieee.org/abstract/document/9428610) [CODE]()

- **`[VCIP 2020]`** [**Neural Rate Control for Video Encoding using Imitation Learning**](https://arxiv.org/abs/2012.05339) [CODE]()
- **`[TCSVT 2020]`** [**Global rate-distortion optimization-based rate control for HEVC HDR coding**](https://ieeexplore.ieee.org/document/8933031) [CODE]()
- **`[TBC 2020]`** [**A novel rate control scheme for video coding in HEVC-SCC**](https://ieeexplore.ieee.org/document/8931257) [CODE]()
- **`[TIP 2020]`** [**Rate Control for Video-Based Point Cloud Compression**](https://ieeexplore.ieee.org/document/9080532) [CODE]()
- **`[TMM 2020]`** [**Rate Control Method Based on Deep Reinforcement Learning for Dynamic Video Sequences in HEVC**](https://doi.org/10.1109/TMM.2020.2992968) [CODE]()
- **`[DCC 2020]`** [**A Rate Control Scheme for HEVC Intra Coding Using Convolution Neural Network (CNN)**](https://ieeexplore.ieee.org/document/9105692) [CODE]()
- **`[ICIP 2020]`** [**Yoco: Light-Weight Rate Control Model Learning**](https://ieeexplore.ieee.org/document/9190880) [CODE]()
- **`[ICASSP 2020]`** [**Intra Frame Rate Control for Versatile Video Coding with Quadratic Rate-Distortion Modelling**](https://ieeexplore.ieee.org/abstract/document/9054633) [CODE]()
- **`[Multimedia Tools and Applications 2020]`** [**FJND-based fuzzy rate control of scalable video for streaming applications**](https://link.springer.com/article/10.1007/s11042-019-08563-4) [CODE]()

- **`[TBC 2019]`** [**Data-driven rate control for rate-distortion optimization in HEVC based on simplified effective initial QP learning**](https://ieeexplore.ieee.org/document/8456843) [CODE]()
- **`[TIP 2019]`** [**Optimize x265 rate control: An exploration of lookahead in frame bit allocation and slice type decision**](https://ieeexplore.ieee.org/document/8579235) [CODE]()
- **`[PCS 2019]`** [**Predicting Rate Control Target Through A Learning Based Content Adaptive Model**](https://ieeexplore.ieee.org/document/8954541) [CODE]()


### Complexity Control
- **`[JVCIP 2024]`** [**Exploring the rate-distortion-complexity optimization in neural image compression☆**](https://www.sciencedirect.com/science/article/abs/pii/S1047320324002505) [CODE]()
- **`[VCIP 2024]`** [**Fast Machine Learning Aided Intra Mode Decision for Real-Time VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10849874) [CODE]()
- **`[SPL 2024]`** [**VVC Intra Coding Complexity Optimization Based on Early Skipping of the Secondary Transform**](https://ieeexplore.ieee.org/abstract/document/10382653) [CODE]()
- **`[PCS 2022]`** [**Performance-Complexity Analysis of Adaptive Loop Filter with a CNN-based Classification**](https://ieeexplore.ieee.org/abstract/document/10018032) [CODE]()
- **`[IEEE Open Journal of Circuits and Systems 2021]`** [**AV1 and VVC Video Codecs: Overview on Complexity Reduction and Hardware Design**](https://ieeexplore.ieee.org/abstract/document/9536216) [CODE]()
- **`[TCSVT 2021]`** [**VVC Complexity and Software Implementation Analysis**](https://ieeexplore.ieee.org/abstract/document/9399488) [CODE]()
- **`[ICIP 2020]`** [**Complexity Analysis Of Next-Generation VVC Encoding And Decoding**](https://ieeexplore.ieee.org/abstract/document/9190983) [CODE]()
- **`[TIP 2020]`** [**Accelerate CTU Partition to Real Time for HEVC Encoding With Complexity Control**](https://ieeexplore.ieee.org/document/9126122) [CODE]()

### Datasets

### Tools
- **`[TCSVT 2021]`** [**Overview of the Screen Content Support in VVC: Applications, Coding Tools, and Performance**](https://ieeexplore.ieee.org/abstract/document/9408666) [CODE]()


### Survey
- **`[Proceedings of the IEEE 2021]`** [**Developments in International Video Coding Standardization After AVC, With an Overview of Versatile Video Coding (VVC)**](https://ieeexplore.ieee.org/document/9328514) [CODE]()





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