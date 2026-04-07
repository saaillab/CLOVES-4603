# CLOVES-4603: Benchmarking Classical Texture Features and Fine-Tuned Deep Models for Clove Quality Grading

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/)
[![Workshop](https://img.shields.io/badge/Workshop-Vision%20for%20Agriculture%20V4A-green)](https://www.agriculture-vision.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Innocent Nyalala, Patrick Vincent Ndowo  
**Affiliation:** SAAIL Lab, IIT Madras Zanzibar Campus, Tanzania  
**Venue:** 7th Workshop on Vision for Agriculture (V4A), CVPR 2026, Denver, USA

---

## Abstract

We introduce CLOVES-4603, the first public benchmark dataset for automated clove quality grading, comprising 4,603 standardized 224×224 RGB images spanning four commercial grades collected at the Zanzibar State Trading Corporation (ZSTC). We benchmark classical texture pipelines (HOG+SVM, GLCM+GLRLM+LBP fusion) against fine-tuned deep learning models including ResNet-50, MobileNetV3, and a frozen DINOv2 vision transformer. ResNet-50 achieves 99.67% accuracy on the test set, while our texture-fusion SVM reaches 92.94%, providing a strong and deployment-friendly classical baseline. This benchmark is designed to inform real-world deployment trade-offs between accuracy, latency, and computational cost in agricultural settings where infrastructure is limited.

---

## Key Results

| Model | Accuracy | Notes |
|---|---|---|
| ResNet-50 (fine-tuned) | 99.67% | Best overall |
| MobileNetV3-Large (fine-tuned) | 99.35% | Lightweight option |
| DINOv2 ViT-B/14 (frozen) | 94.90% | Zero-shot baseline |
| GLCM+GLRLM+LBP SVM | 92.94% | Classical baseline |
| HOG+SVM | 32.68% | Weak baseline |

---

## Dataset

CLOVES-4603 contains 4,603 images across four official ZSTC commercial grades:

| Grade | Description |
|---|---|
| Grade 1 | Attractive golden/saffron color, max 3% mpeta |
| Grade 2 | Faded/slightly blackish, max 7% mpeta |
| Grade 3 | More faded color, max 20% mpeta |
| Grade 4 | Primarily mpeta cloves, >20% mpeta |

> Dataset download link coming soon via Zenodo.

---

## Repository Structure

```
CLOVES-4603/
├── data/                  # Dataset placeholder (see Zenodo link above)
├── src/
│   ├── classical/         # Texture feature pipelines
│   ├── deep/              # Fine-tuning scripts
│   └── evaluation/        # Metrics and visualization
├── notebooks/             # Experiment notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/saaillab/CLOVES-4603
cd CLOVES-4603
pip install -r requirements.txt
```

---

## Citation

```bibtex
@inproceedings{nyalala2026cloves,
  title={CLOVES-4603: Benchmarking Classical Texture Features and Fine-Tuned Deep Models for Clove Quality Grading},
  author={Nyalala, Innocent and Ndowo, Patrick Vincent},
  booktitle={7th Workshop on Vision for Agriculture (V4A), CVPR},
  year={2026}
}
```

---

## About SAAIL Lab

SAAIL Lab (Sustainable AI for Agriculture and Intelligent Livelihoods) is based at IIT Madras Zanzibar Campus, Tanzania. We build responsible, locally grounded AI solutions for East Africa and the Global South.

🌍 [saaillab.github.io](https://saaillab.github.io)
