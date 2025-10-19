# Skin Cancer Classification using EfficientNetB0

##  Overview
This project uses deep learning (EfficientNetB0) to classify skin lesions from the HAM10000 dataset.  
The goal is to help early skin cancer detection through AI-based image analysis.

## Dataset
The HAM10000 dataset contains over 10,000 dermatoscopic images of different types of skin lesions.  
Each image is labeled with one of seven diagnosis classes such as melanoma, nevus, or basal cell carcinoma.

##  Project Pipeline
1. Data loading and cleaning  
2. Label encoding and class balancing  
3. Data augmentation using Keras  
4. Model building with EfficientNetB0  
5. Training and fine-tuning  
6. Evaluation on test set

## Tools & Libraries
Python, TensorFlow, Keras
Transfer Learning (EfficientNetB0 pretrained on ImageNet)
Data Augmentation & Class Balancing
Stratified Train/Val/Test Split
Fine-tuning


## Results
- **Test Accuracy:** 66.9%
- **Loss:** 1.93
- The model shows reasonable performance considering class imbalance and limited dataset size.

## Key Learnings
- Learned how to apply transfer learning effectively.
- Understood the importance of careful fine-tuning and class balancing.
- Improved skills in model evaluation and visualization.

Example of predictions:

Deep-Learnign-Projeacts/Skin-Cancer-Detection/images/ISIC_0024373.jpg


