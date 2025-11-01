# Real-Time Tomato Ripeness Detection

## Overview
This project uses **deep learning (CNN)** to classify the ripeness of tomatoes from images in real-time.  
The goal is to help farmers and food suppliers monitor tomato ripeness using an AI-based image analysis tool.

## Dataset
The dataset used in this project comes from **Kaggle**:  
[Tomato Maturity Detection](https://www.kaggle.com/datasets/sujaykapadnis/tomato-maturity-detection-and-quality-grading?utm_source=chatgpt.com)  

It contains labeled images of tomatoes categorized as **Mature** or **Immature**.  
Images are preprocessed and resized to 224x224 pixels for model input.

## Project Pipeline
1. Data loading and preprocessing  
2. Image normalization and resizing  
3. Building and fine-tuning a CNN model using TensorFlow/Keras  
4. Training and evaluation on the dataset  
5. Deploying the model using **Gradio** for real-time inference

## Tools & Libraries
- Python  
- TensorFlow / Keras  
- Gradio for an interactive web interface  
- PIL & NumPy for image processing  

## Results
- **Test Accuracy:** 99.83%  
- **Test Loss:** 0.0108  
- **Test Precision:** 100.00%  
- **Test Recall:** 99.67%  

The model performs exceptionally well, providing accurate predictions for tomato ripeness.

## Key Learnings
- Learned to build and fine-tune CNN models for image classification.  
- Understood the importance of preprocessing and normalization for high accuracy.  
- Gained experience deploying ML models using **Gradio** for real-time usage.  

## Live Demo
Try the interactive demo here:  
[Gradio App Link](https://huggingface.co/spaces/Hesham-vision/Tomato-Ripeness-Detection)

