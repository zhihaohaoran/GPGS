# GP-GS: Gaussian Processes for Enhanced 3D Gaussian Splatting



**This is the anonymous code repository for ICML2025 submission paper: GP-GS: Gaussian Processes for Enhanced 3D Gaussian Splatting **

## 😊️ Pipeline

![teaser](assets/overview.png)

# GP-GS: Gaussian Processes for Enhanced Gaussian Splatting

## Overview
GP-GS is a novel framework that enhances the initialization of 3D Gaussian Splatting (3DGS) by leveraging Multi-Output Gaussian Processes (MOGP). It improves the rendering quality of novel view synthesis by densifying sparse point clouds reconstructed via Structure-from-Motion (SfM). The method is particularly effective in complex regions with densely packed objects and challenging lighting conditions.

## Pipeline
Our pipeline consists of the following steps:

1. **Multi-View Image Input**: We start with multi-view images and extract per-view depth maps using depth estimation models (e.g., Depth Anything).
2. **SfM Reconstruction**: Sparse point clouds are generated from the input images using Structure-from-Motion (SfM).
3. **Point Cloud Densification**: 
   - MOGP is trained to take pixel coordinates and depth values as input and predict dense 3D points with position and color information.
   - A Matérn kernel is used to model smooth spatial variations, and the parameters are optimized via gradient updates.
4. **Uncertainty-Based Filtering**:
   - High-variance noisy points are filtered out based on a variance-based thresholding strategy to ensure structured densification.
5. **Gaussian Initialization and Optimization**:
   - The densified points are used to initialize 3D Gaussians, which undergo further optimization to improve geometric accuracy.
6. **Novel View Rendering**:
   - The optimized 3D Gaussians are used for efficient rasterization-based rendering to synthesize high-quality novel views.
## Local setup
conda env create --file environment.yml
conda activate GP-GS
### Running
**MOGP**:
```shell
python MOGP/top_four_contribution.py #Find the image from a perspective that contributes most to SfM points cloud.
python MOGP/mogp_train.py #Training MOGP model
python MOGP/predict.py #Predict high quality dense points cloud.
```
**MOGP for 3D gaussians Initialization**:
```shell
python MOGP/rewrite_images_sfm.py
python MOGP/write_points3d.py
```
**3DGS***:
```shell
python train.py -s <scene path>
```
**Render and Evaluation**:
```shell
python render.py -m <model path>
python metrics.py -m <model path>
```


```
