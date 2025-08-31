# 🧠 SMART Brain Tumour Detection System using Vision Transformers  

A deep learning system for automated **brain tumour detection** from MRI images using **Vision Transformers (ViT)**.  
Deployed with a **Gradio interface** on Hugging Face Spaces.  

---

## ✨ Features
- ✅ **Vision Transformers (ViT-Base/16)** fine-tuned for MRI classification  
- ✅ **Strict dataset separation** (independent training & testing sources)  
- ✅ **K-Fold cross-validation** for robust evaluation  
- ✅ **High accuracy (94–97%)** across multiple evaluation settings  
- ✅ **Multi-class classification**: *Pituitary, Glioma, Meningioma, No Tumour*  
- ✅ **User-friendly interface** via Gradio  

---

## 📊 Results
| Model Version | Dataset Setting        | Accuracy (%) | Remarks |
|---------------|------------------------|--------------|---------|
| Version 1     | Random Split           | 95.36        | Unseen training samples |
| Version 2     | Independent Dataset    | 96.21        | Separate test dataset |
| Version 2     | K-Fold Cross Validation| 96.0 (mean)  | Robustness & generalization |

---

## 📂 Dataset Sources
- **Dataset 1 (MRI, Testing):** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing)  
- **Dataset 2 (CT & MRI, Testing):** [Brain Tumor Multimodal Image Dataset](https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri/code)  

---

## 🛠️ Methodology
1. **Preprocessing**
   - Resize images → `224 × 224`  
   - Normalize pixel values → `[0,1]`  
   - Structured labels into 4 classes  

2. **Model**
   - Pre-trained **ViT-Base/16 (224)**  
   - Optimizer: **Adam** (`lr=3e-5`)  
   - Loss: **CrossEntropyLoss**  
   - Training Epochs: **5**  

3. **Evaluation**
   - Accuracy, Confusion Matrix, Validation Curves  
   - **5-Fold Cross Validation**  

---

## 🚀 Deployment
- **Interactive Demo:** [Hugging Face Space](https://huggingface.co/spaces/rgb95/smartBraintumourDetection)  
- **Source Code:** [GitHub Repository](https://github.com/ibrahimamam1/Brain-Tumour-Detection.git)  

---

## 👨‍💻 My Contribution (Kazi Shakkhar Rahman – 210041240)
As part of this project, I was responsible for:  
- ✅ **Strict dataset separation** → ensuring completely independent training/testing sources  
- ✅ **Implementing the Vision Transformer (ViT) model**  
- ✅ **Model training, evaluation, and documentation**  
- ✅ **Visualization** (Confusion matrix, training/loss curves)  
- ✅ **Creating testing workflow** → load trained model without retraining for quick evaluation  

---

## 👥 Team Contributions
- **Kazi Shakkhar Rahman (210041240):** Data collection, preprocessing, ViT model implementation, evaluation, documentation  
- **Ibrahima Mamoudu (210041259):** K-Fold cross-validation, parameter tuning, attention maps, deployment  
- **Ahmed Albreem (210041258):** Gradio interface design, real-time system development, bug fixes  

---

## ⚠️ Challenges
- Avoiding dataset leakage with strict separation  
- Handling imbalanced MRI classes  
- High computational cost of ViTs  
- Parameter tuning across folds  
- Deploying under limited compute (Hugging Face free tier)  

---

