# Stroke-Seg: A Deep Learning-based Framework for Chinese Stroke Segmentation 🖌️

[![Paper Status](https://img.shields.io/badge/Paper-Under_Review-yellow)](https://ietresearch.onlinelibrary.wiley.com/journal/17519667)
[![GitHub stars](https://img.shields.io/github/stars/Rvosuke/Stroke-Seg.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/Rvosuke/Stroke-Seg/stargazers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📝 About

Stroke-Seg is a novel framework designed to improve the stroke segmentation of Chinese characters by tackling complex stroke characteristics. This repository contains the implementation of our method, which is currently under review at IET Image Processing.

![Stroke Segmentation Example](stroke_seg_example.png)

## 🔑 Key Features

- 🧠 Incorporates a Prior Knowledge Vector for enhanced character understanding
- 🏷️ Multi-label output strategy for handling intersecting strokes
- 💪 Compatible with various network architectures (CNNs and Transformers)
- 📊 Dedicated 'Stroke Class' mechanism for balancing stroke distribution
- 🎯 Novel BDLoss strategy for capturing both global and local stroke features

## 📊 Results

The dataset used in this project, BCSS (Brush Calligraphy Stroke Segmentation), is available in a separate repository: [Rvosuke/BCSS](https://github.com/Rvosuke/BCSS)

Our framework demonstrates excellent generalization ability and compatibility with various architectures. Compared to existing methods, Stroke-Seg achieves remarkable improvements in True Stroke Rate while requiring less computational cost.

## 📖 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{bai2024strokeseg,
  title={Stroke-Seg: A Deep Learning-based Framework for Chinese Stroke Segmentation},
  author={Bai, Zeyang and Gong, Xinyu and Nie, Haitao and Xie, Bin},
  journal={IET Image Processing},
  year={2024},
  publisher={IET},
  note={Under Review}
}
```