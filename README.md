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
- **`[DCC 2024]`** [**Decoder-side Secondary Transform Derivation for Video Coding beyond AVS3**](https://ieeexplore.ieee.org/abstract/document/10533834) [CODE]()
- **`[DCC 2024]`** [**Construction of Fast Data-driven Transforms for Image Compression via Multipath Coordinate Descent on Orthogonal Matrix Manifold**](https://ieeexplore.ieee.org/abstract/document/10533826) [CODE]()
- **`[ICASSP 2024]`** [**Adaptive Secondary Transform Sets for Video Coding Beyond AV1**](https://ieeexplore.ieee.org/abstract/document/10447533) [CODE]()
- **`[ICCCAS 2024]`** [**A Fast Transform Algorithm for VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/9825469) [CODE]()
- **`[ACM TOM 2024]`** [**Graph Based Cross-Channel Transform for Color Image Compression**](https://dl.acm.org/doi/abs/10.1145/3631710) [CODE]()
- **`[TCSVT 2024]`** [**Revisiting All-Zero Block Detection for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10378712) [CODE]()
- **`[TCSVT 2023]`** [**Learning-Based Early Transform Skip Mode Decision for VVC Screen Content Coding**](https://ieeexplore.ieee.org/abstract/document/10068735) [CODE]()
- **`[ISCAS 2023]`** [**High-Throughput Design for a Multi-Size DCT-II Targeting the AV1 Encoder**](https://ieeexplore.ieee.org/abstract/document/10181828) [CODE]()
- **`[ISCAS 2022]`** [**A High-Throughput Design for the H.266/VVC Low-Frequency Non-Separable Transform**](https://ieeexplore.ieee.org/abstract/document/9937597) [CODE]()
- **`[ICIP 2021]`** [**Machine-Learning Based Secondary Transform for Improved Image Compression in JPEG2000**](https://doi.org/10.1109/ICIP42928.2021.9506122) [CODE]()
- **`[DCC 2021]`** [**Graph Based Transforms based on Graph Neural Networks for Predictive Transform Coding**](https://doi.org/10.1109/DCC50243.2021.00079) [CODE]()
- **`[PCS 2021]`** [**Combined neural network-based intra prediction and transform selection**](https://doi.org/10.1109/PCS50896.2021.9477455) [CODE]()
- **`[TCSVT 2021]`** [**Transform Coding in the VVC Standard**](https://ieeexplore.ieee.org/abstract/document/9449858) [CODE]()
- **`[ICIP 2020]`** [**Augmenting JPEG2000 With Wavelet Coefficient Prediction**](https://doi.org/10.1109/ICIP40778.2020.9190969) [CODE]()
- **`[VCIP 2020]`** [**Deep Learning-Based Nonlinear Transform for HEVC Intra Coding**](https://doi.org/10.1109/VCIP49819.2020.9301790) [CODE]()



### Quantization
- **`[TBC 2023]`** [**Complexity-Efficient Dependent Quantization for Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/10093119) [CODE]()

- **`[ICIP 2022]`** [**Learning Frequency-Specific Quantization Scaling in VVC for Standard-Compliant Task-Driven Image Coding**](https://doi.org/10.1109/ICIP46576.2022.9897987) [CODE]()

- **`[DCC 2021]`** [**JQF: Optimal JPEG Quantization Table Fusion by Simulated Annealing on Texture Images and Predicting Textures**](https://doi.org/10.1109/DCC50243.2021.00041) [CODE]()

- **`[VCIP 2020]`** [**Towards Quantized DCT Coefficients Restoration for Compressed Images**](https://doi.org/10.1109/VCIP49819.2020.9301794) [CODE]()
- **`[DCC 2020]`** [**Deep Learning-based Image Compression with Trellis Coded Quantization**](https://doi.org/10.1109/DCC47342.2020.00009) [CODE]()
- **`[ECCV 2020]`** [**Task-Aware Quantization Network for JPEG Image Compression**](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650307.pdf) [CODE]()
- **`[DCC 2019]`** [**Hard-Decision Quantization Algorithm Based on Deep Learning in Intra Video Coding**](https://doi.org/10.1109/DCC.2019.00119) [CODE]()

### Entropy Coding
### In-Loop Filtering
- **`[TMM 2023]`** [**Joint Rate-Distortion Optimization for Video Coding and Learning-Based In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10216325) [CODE]()

### Rate-Distortion Optimization
- **`[TMM 2023]`** [**Joint Rate-Distortion Optimization for Video Coding and Learning-Based In-Loop Filtering**](https://ieeexplore.ieee.org/abstract/document/10216325) [CODE]()

### Rate Control
- **`[TCSVT 2024]`** [**Content-Adaptive Rate Control Method for User-Generated Content Videos**](https://ieeexplore.ieee.org/abstract/document/10734399) [CODE]()
- **`[TMM 2024]`** [**Content-Adaptive Rate-Distortion Modeling for Frame-Level Rate Control in Versatile Video Coding**](https://ieeexplore.ieee.org/abstract/document/9190983) [CODE]()

- **`[TMM 2020]`** [**Rate Control Method Based on Deep Reinforcement Learning for Dynamic Video Sequences in HEVC**](https://doi.org/10.1109/TMM.2020.2992968) [CODE]()

### Complexity Control
- **`[VCIP 2024]`** [**Fast Machine Learning Aided Intra Mode Decision for Real-Time VVC Intra Coding**](https://ieeexplore.ieee.org/abstract/document/10849874) [CODE]()
- **`[SPL 2024]`** [**VVC Intra Coding Complexity Optimization Based on Early Skipping of the Secondary Transform**](https://ieeexplore.ieee.org/abstract/document/10382653) [CODE]()


- **`[TCSVT 2021]`** [**VVC Complexity and Software Implementation Analysis**](https://ieeexplore.ieee.org/abstract/document/9399488) [CODE]()
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