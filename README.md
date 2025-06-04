# 🔬 [2024] LGDisplay-DGU-Cooperation

## 📌 Project Overview

- **Title**: Development of a Multi-Illumination-Based Visual Inspection Algorithm  
- **Objective**: Defect Classification in Display Products  
- **Partner**: LG Display × Dongguk University (Industry-Academia Collaboration Project)  

## 🏆 Achievements

- ✅ **Defect Classification Accuracy**:  
  - 98.5% (192/195 images correctly classified on test set)

- ✅ **Normal Classification Accuracy**:  
  - 51.3% (157/306 images correctly classified on test set)

- ✅ **Paper in Progress**:  
  - Research on *ProtoDC-Net: ~~* has been submitted and is under review.

- ✅ **Blog Post**: https://johyeongseob.tistory.com/57


---

## 🧠 Code Description

This repository implements multiple ensemble models and deep learning backbones for defect classification under multi-illumination settings.

### 📁 `Ensemble/` Directory

- `model1_ensemble.py`  
  → Ensemble result script for Model 1  
- `model2_ensemble.py`  
  → Ensemble result script for Model 2  
- `model3_ensemble.py`  
  → Ensemble result script for Model 3  
- `model_ensemble.py`  
  → Final ensemble result integrating all models  
- `model_train.py`  
  → Training script for individual models  
- `model_evaluation.py`  
  → Evaluation script for each trained model  

### 📁 `models/` Directory

- `PretrainedSqueezeNet.py`  
  → SqueezeNet model pretrained on ImageNet-1k  
- `SENet.py`  
  → Implementation of the SENet (Squeeze-and-Excitation Network)

### 🧾 Other Core Scripts

- `Classifier.py`  
  → Contains the classification logic  
  - Uses `num_classes` parameter to switch model architecture

- `DataLoader.py`  
  → Prepares image batches for training and testing  
  - Uses `target` parameter to adjust for different model configurations

---

## 📂 Dataset

*This project uses a private dataset provided by LG Display and is not publicly available.*

---

이 README는 GitHub에서 바로 사용 가능해.  
필요하다면 실행 방법(`Usage`), 요구 라이브러리(`Requirements`), 논문 링크 등도 추가해줄 수 있어. 원할 경우 알려줘!
