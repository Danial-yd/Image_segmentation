# Semantic Segmentation on Open Images V7

## Project Overview

This repository contains an implementation of semantic segmentation models for multi-class object detection in the **Open Images V7** dataset. We compare a custom-designed **U-Net** architecture against a modified **DeepLabV3-ResNet101** model, focusing on three target classes: **Person**, **Car**, **Dog**, and **Background**.

---

## Key Components

### ✅ **Dataset Processing**
- Created multi-class segmentation masks from Open Images' instance annotations
- Preprocessed **6,100 samples** to **256×256** resolution
- Implemented class-specific masking:  
  - `Background = 0`  
  - `Person = 1`  
  - `Dog = 2`  
  - `Car = 3`  

### **Model Architecture**
- **Custom U-Net**: Lightweight encoder-decoder structure with dropout regularization  
- **DeepLabV3-ResNet101**: Pretrained model adapted for 4-class segmentation  

### **Training & Evaluation**
- 8-epoch training using **SGD optimizer** (`lr = 0.01`)
- Pixel-wise **CrossEntropyLoss** implementation
- Comprehensive per-class metrics: **Precision**, **Recall**, **F1-Score**
- Support-weighted **accuracy** calculation

---

## 📊 Key Findings

| **Metric**        | **Custom U-Net** | **DeepLabV3**   | **Improvement** |
|-------------------|------------------|------------------|------------------|
| **Accuracy**       | 42%              | 58%              | **+38%**         |
| **Weighted F1**    | 39%              | 52%              | **+33%**         |
| **Inference Speed**| 23ms/img         | 41ms/img         | **-78%**         |

---

## Class-Specific Insights

- 🐕 **Dog** class showed **0 performance** due to extreme data imbalance  
- 👥 **Person** detection achieved **highest precision** at **50%**  
- 🚗 **Car** segmentation showed **most balanced performance**  
- 🌆 **Background** class dominated **recall metrics** at **61%**


