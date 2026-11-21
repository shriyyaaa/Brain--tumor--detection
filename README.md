# Brain Tumor MRI Classification with Transfer Learning

Deep learning project that classifies brain MRI scans into four categories (**glioma, meningioma, pituitary tumor, no tumor**) by fine-tuning an ImageNet-pretrained CNN. Grad-CAM heatmaps show which regions of the scan influenced each prediction.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/brain_tumor_detection_colab.ipynb)

> **Disclaimer:** This project is for education and research only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Demo

<!-- Add after training: a Grad-CAM image from predict.py -->
![Grad-CAM example](assets/gradcam_example.png)

## Overview

- **Task:** 4-class image classification on brain MRI
- **Approach:** transfer learning with ResNet50 or EfficientNet-B0 (PyTorch / torchvision)
- **Training strategy:**
  1. Freeze the pretrained backbone and train only the new classifier head
  2. Unfreeze the last blocks and fine-tune at a lower learning rate
- **Regularization:** data augmentation (flip, rotation, brightness/contrast), dropout, label smoothing, weight decay
- **Explainability:** Grad-CAM overlays

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle), a combination of the Figshare, SARTAJ and Br35H datasets.

| Split | Use |
|---|---|
| `Training/` | 85% train, 15% stratified validation |
| `Testing/` | held out, used only for the final evaluation |

Expected layout:

```
data/
  Training/  glioma/ meningioma/ notumor/ pituitary/
  Testing/   glioma/ meningioma/ notumor/ pituitary/
```

## Project structure

```
.
├── brain_tumor_detection_colab.ipynb   # end-to-end notebook (train, evaluate, Grad-CAM)
├── train.py                            # training + evaluation script
├── predict.py                          # single-image prediction + Grad-CAM
├── requirements.txt
├── assets/                             # images used in this README
└── outputs/                            # created by train.py (model, plots, metrics)
```

## Getting started

**Option 1: Google Colab (easiest).** Click the Colab badge above, set the runtime to GPU, and run all cells.

**Option 2: Local**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# after downloading the dataset into data/
python train.py --data_dir data --arch resnet50
python predict.py --image path/to/scan.jpg --checkpoint outputs/best_model.pt
```

Useful training options: `--arch {resnet50,efficientnet_b0}`, `--epochs_head`, `--epochs_ft`, `--lr_head`, `--lr_ft`, `--batch_size`, `--img_size`.

## Results

Evaluated on the held-out `Testing/` set (1,600 images, 400 per class) with ResNet50, using the default two-phase training setup (5 epochs head-only, then 10 epochs fine-tuning).

| Model | Test accuracy | Macro F1 |
|---|---|---|
| ResNet50 | 94.00% | 0.9389 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.9849 | 0.8175 | 0.8934 | 400 |
| meningioma | 0.9244 | 0.9475 | 0.9358 | 400 |
| notumor | 0.8869 | 1.0000 | 0.9401 | 400 |
| pituitary | 0.9779 | 0.9950 | 0.9864 | 400 |

**Takeaways**

- Pituitary tumors are detected almost perfectly (F1 0.986), and meningioma is solid (F1 0.936).
- **Glioma is the weak spot:** precision is high (0.985) but recall is only 0.818, so about 18% of gliomas are missed. The very high no-tumor recall (1.00) paired with lower no-tumor precision (0.887) suggests that many of the missed gliomas are being labeled as "no tumor". The confusion matrix below shows exactly where they go.
- In a medical setting, missing a tumor is the costlier error, so glioma recall is the number to improve first (for example with class-weighted loss, more fine-tuning, or a stronger backbone).

![Confusion matrix](outputs/confusion_matrix.png)
![Training curves](outputs/training_curves.png)

## Limitations

- Trained on a single public dataset; performance on scans from other scanners, hospitals or protocols is untested.
- Some images may come from the same patient across train and test splits, which can inflate accuracy.
- 2D slices only; no 3D context and no patient metadata.
- Not clinically validated.

## Future work

- External validation on a separate dataset
- Additional backbones (DenseNet, ConvNeXt, ViT)
- Patient-level data splitting
- Gradio / Streamlit web demo
- Calibration and uncertainty estimates

## Acknowledgements

- Dataset: Masoud Nickparvar, Brain Tumor MRI Dataset (Kaggle)
- Pretrained weights: torchvision ImageNet models
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)

