# Historical Photo Restoration Pipeline

A modular deep learning pipeline designed to restore and colorize degraded historical photographs. This project utilizes a divide-and-conquer architecture, separating spatial restoration (super-resolution and denoising) from chromatic restoration (colorization) to achieve high-fidelity results on vintage media.

## Overview

Historical photographs often suffer from low resolution, heavy film grain, and a lack of color. Applying a single monolithic model to solve all these issues simultaneously often yields suboptimal results. This pipeline addresses the problem in two distinct, optimized stages:

1. **Spatial Restoration:** Powered by Real-ESRGAN to upscale the image 2x, remove noise, and recover fine details.
2. **Chromatic Restoration:** Powered by DeOldify to predict plausible color channels (A/B in CIELAB space) and bring the image to life.

## Model Architecture

- **Real-ESRGAN:** Utilizes a Residual-in-Residual Dense Block (RRDB) generator and a U-Net discriminator. It is trained on a high-order degradation model to handle real-world noise and compression artifacts.
- **DeOldify (Artistic Model):** Utilizes a U-Net architecture with a ResNet-34 backbone. It employs a "NoGAN" training strategy, combining the vivid colors of a GAN with the stability of a perceptual loss function.

## Prerequisites and Installation

This project is optimized for Python 3.10 and requires a CUDA-enabled GPU (tested on an NVIDIA RTX 3050 Ti, 4GB VRAM).

**1. Create a Conda Environment**
```bash
conda create -n restore python=3.10 -y
conda activate restore
```

**2. Install Dependencies**
Install PyTorch with CUDA support first, then install the remaining requirements:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Download Model Weights**
Due to GitHub file size limits, the model weights must be downloaded manually and placed in their respective folders:
Download RealESRGAN_x4plus.pth and place it in the weights/ directory.
Download ColorizeArtistic_gen.pth and place it in the DeOldify/models/ directory.

**4. Place your degraded, black-and-white photos (.jpg, .jpeg, or .png) into the input_photos/ directory.**

**5. Run the pipeline**
```bash
python run_pipeline.py
```

**6. The script will output the 2x upscaled images into upscaled_photos/ and the final colorized results into colorized_photos/.**

## Result
The pipeline significantly improves PSNR and SSIM metrics compared to frame-wise processing, while ensuring temporal consistency is not an issue for static photographs.

## Reference
Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. ICCV.
Antic, J. (2019). DeOldify: A deep learning based project for colorizing and restoring old images and videos.
Wan, Z., et al. (2022). Bringing Old Films Back to Life. CVPR.

## Project Author
Spandhan Reddy
