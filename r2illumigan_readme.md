# R2IllumiGAN: Region-Aware Residual Retinex Decomposition GAN for Low-Light Image Enhancement

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)

**R2IllumiGAN** is a deep learning architecture designed for low-light image enhancement using region-aware residual Retinex decomposition and multi-domain adversarial learning. Built on top of PyTorch, it effectively decomposes low-light images into illumination and reflectance components, suppresses noise, and restores natural colors and fine details.

---

## 📌 Features & Key Innovations

* **Retinex-Based Deep Architecture**: Decomposes input low-light images ($I_l$) into high-quality normal-light outputs ($I_n$) using progressive feature refinement.
* **Dual-Branch Refinement**: Separates illumination enhancement and reflectance recovery into specialized streams for balanced exposure and detail preservation.
* **Triple-Domain Discriminator**: Evaluates generated images across three distinct domains:
  1. **RGB Domain**: Ensures global structural and color fidelity.
  2. **Illumination Domain**: Verifies natural luminance distribution.
  3. **Gradient Domain**: Preserves high-frequency texture and edge information.
* **Noise-Aware Residual Suppression**: Filters out sensor noise amplified during brightness enhancement.
* **Perceptual & Edge-Aware Losses**: Utilizes pre-trained VGG16 features alongside custom gradient losses to produce crisp, artifact-free visual results.

---

## 🏗️ Architecture Overview

The generator network (`R2IllumiGANGenerator`) consists of seven interconnected functional sub-modules:

1. **Shallow Feature Encoder (`ShallowFeatureEncoder`)**: Extracts initial shallow features from low-light input images.
2. **Residual Retinex Decomposition (`ResidualRetinexDecomposition`)**: Decomposes shallow features into baseline illumination and reflectance maps.
3. **Gradient-Guided Reflectance Recovery (`GradientGuidedReflectanceRecovery`)**: Enhances edge details and textures in the reflectance map using structural gradients.
4. **Illumination Residual Branch (`IlluminationResidualBranch`)**: Learns residual illumination maps to correct uneven exposure.
5. **Adaptive Illumination Enhancement (`AdaptiveIlluminationEnhancement`)**: Dynamically adjusts brightness levels using spatial attention mechanisms.
6. **Noise-Aware Residual Suppression (`NoiseAwareResidualSuppression`)**: Removes high-frequency noise induced by severe low-light conditions.
7. **RGB Residual Refinement (`RGBResidualRefinement`)**: Blends illumination and reflectance features back into the final high-resolution RGB image.

---

## 📂 Dataset Structure

The repository is pre-configured for the **LOL (Low-Light) Dataset**. Structure your dataset directory as follows:

```text
dataset/
└── lol_dataset/
    ├── our485/          # Training split
    │   ├── low/        # Low-light images (input)
    │   └── high/       # Normal-light ground truth
    └── eval15/          # Evaluation split
        ├── low/
        └── high/
```

If using zipped dataset archives in Google Colab/Kaggle environments:
* Extract your archive to `/content/unzipped_lol` or update `LOCAL_UNZIP_DIR` in the notebook script.

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ and PyTorch installed with GPU support.

```bash
# Clone the repository
git clone https://github.com/your-username/R2IllumiGAN.git
cd R2IllumiGAN

# Install dependencies
pip install torch torchvision numpy pillow opencv-python tqdm
```

### Hardware Requirements
* **GPU**: NVIDIA GPU with at least 8 GB VRAM (16 GB Recommended for batch size $\ge 8$).
* **CUDA**: 11.7 or later.

---

## 🚀 Quick Start & Training

### 1. Data Preparation & Loading
The custom PyTorch `Dataset` handles random cropping and data augmentation:

```python
from torch.utils.data import DataLoader
from r2illumigan import LocalRuntimeLOLDataset

train_dataset = LocalRuntimeLOLDataset(
    root_dir="/path/to/lol_dataset/our485",
    crop_size=256,
    is_train=True
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
```

### 2. Initializing Model & Discriminator

```python
import torch
from r2illumigan import R2IllumiGANGenerator, TripleDomainDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Generator and Discriminator
generator = R2IllumiGANGenerator().to(device)
discriminator = TripleDomainDiscriminator().to(device)
```

### 3. Running Training Loop

Run the provided Jupyter Notebook (`R2IllumiGAN_Training.ipynb`) sequentially:
1. **Cell 1**: Environment setup, GPU verification, and dataset extraction.
2. **Cell 2**: Dataset definition, Generator architecture, and Discriminator definitions.
3. **Cell 3**: Optimizer setup (AdamW), Perceptual VGG Loss definition, and progressive training loop with checkpoint saving.

---

## 📊 Loss Functions & Optimization

R2IllumiGAN is trained end-to-end using a weighted combination of losses:

$$\mathcal{L}_{Total} = \lambda_{rec} \mathcal{L}_{1} + \lambda_{vgg} \mathcal{L}_{VGG} + \lambda_{grad} \mathcal{L}_{Grad} + \lambda_{adv} \mathcal{L}_{GAN}$$

* **$\mathcal{L}_{1}$ Reconstruction Loss**: Enforces pixel-level similarity between generated and ground-truth images.
* **$\mathcal{L}_{VGG}$ Perceptual Loss**: Measures high-level feature similarity extracted from VGG16 layers (`relu1_2`, `relu2_2`, `relu3_3`).
* **$\mathcal{L}_{Grad}$ Gradient Loss**: Preserves sharp edges by matching spatial derivatives.
* **$\mathcal{L}_{GAN}$ Multi-Domain Adversarial Loss**: Evaluates realism across RGB, Illumination, and Gradient domains using LSGAN / Vanilla GAN objectives.

---

## 🧪 Evaluation Metrics

Model performance can be evaluated on the `eval15` validation set using standard image quality metrics:
* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index Measure)
* **LPIPS** (Learned Perceptual Image Patch Similarity)

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Citation & Acknowledgments

If you find this work useful in your research, please consider citing:

```bibtex
@article{r2illumigan2026,
  title={R2IllumiGAN: Region-Aware Residual Retinex Decomposition GAN for Low-Light Image Enhancement},
  author={Your Name},
  journal={GitHub Repository},
  year={2026}
}
```