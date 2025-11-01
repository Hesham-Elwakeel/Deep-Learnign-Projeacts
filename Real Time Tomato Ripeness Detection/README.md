# Real-Time Tomato Ripeness Detection

## Overview
This project uses **deep learning (Transfer Learning with MobileNetV2)** to classify the ripeness of tomatoes from images in real-time.  
The goal is to help farmers and food suppliers monitor tomato ripeness using an AI-based image analysis tool.

## Dataset
The dataset used in this project comes from **Kaggle**:  
[Tomato Maturity Detection](https://www.kaggle.com/datasets/sujaykapadnis/tomato-maturity-detection-and-quality-grading?utm_source=chatgpt.com)  

**Dataset Details:**
- **Total Images:** 4,000 (Augmented Dataset)
- **Classes:** 
  - **Immature**: 2,000 images
  - **Mature**: 2,000 images
- **Split:**
  - Training: 70% (2,800 images)
  - Validation: 15% (600 images)
  - Testing: 15% (600 images)
- **Image Size:** Resized to 224x224 pixels for model input

## Project Pipeline
1. **Data Loading & Exploration**
   - Mounted Google Drive and accessed the dataset
   - Explored data distribution and visualized sample images
   
2. **Data Preprocessing**
   - Split data into train/validation/test sets (70/15/15)
   - Applied data augmentation (rotation, flipping, zoom, shear)
   - Normalized pixel values to [0, 1]

3. **Model Architecture**
   - **Base Model:** MobileNetV2 (pre-trained on ImageNet)
   - **Transfer Learning Approach:**
     - Phase 1: Froze base model, trained custom layers
     - Phase 2: Fine-tuned the last 30 layers of the base model
   - **Custom Layers:**
     - Global Average Pooling
     - Dense layer (128 units, ReLU)
     - Dropout (0.3, 0.2)
     - Output layer (1 unit, Sigmoid for binary classification)

4. **Training Process**
   - **Phase 1:** 25 epochs with frozen base model
   - **Phase 2:** 15 epochs of fine-tuning
   - **Optimizer:** Adam (LR: 0.0001 → 0.00001)
   - **Loss Function:** Binary Crossentropy
   - **Callbacks:**
     - Early Stopping (patience=5)
     - Model Checkpoint (save best model)
     - ReduceLROnPlateau (factor=0.5, patience=3)

5. **Evaluation & Visualization**
   - Generated confusion matrix
   - Plotted training/validation curves
   - Tested predictions on sample images

6. **Deployment**
   - Deployed using **Gradio** on Hugging Face Spaces
   - Real-time inference with a user-friendly interface

## Tools & Libraries
- **Python**
- **TensorFlow** / Keras for model building
- **Gradio** for interactive web interface
- **PIL & NumPy** for image processing
- **Matplotlib & Seaborn** for visualization
- **Google Colab** for training
- **Hugging Face Spaces** for deployment

## Results
- **Test Accuracy:** 99.83%  
- **Test Loss:** 0.0108  
- **Test Precision:** 100.00%  
- **Test Recall:** 99.67%  

The model performs exceptionally well, achieving near-perfect classification of tomato ripeness.

## Challenges Faced & Solutions

### 1. **Data Organization Issues**
**Problem:** After extracting the dataset from Kaggle, the data was nested in multiple folders, making it difficult to access.

**Solution:** 
- Created a Python script to explore folder structure using `os.walk()`
- Identified the correct path to Augment Dataset
- Organized data into clear train/validation/test splits

---

### 2. **Session Disconnection on Google Colab**
**Problem:** Google Colab session disconnected during long training, losing all variables and progress.

**Solution:**
- Implemented automatic model checkpointing using `ModelCheckpoint` callback
- Saved models with timestamps to Google Drive
- Created a "reload cell" to quickly restore model and settings after disconnection
- Used `EarlyStopping` to prevent unnecessarily long training

---

### 3. **File Path Errors During Data Splitting**
**Problem:** `FileNotFoundError` when trying to copy images to train/val/test folders due to:
- Special characters in filenames
- Missing files
- Incorrect path handling

**Solution:**
- Added file existence checks before copying: `if os.path.exists(src)`
- Filtered only image files using extensions: `.jpg`, `.jpeg`, `.png`
- Implemented try-except blocks for error handling
- Added progress logging to track which files failed

---

### 4. **Undefined Variables After Session Loss**
**Problem:** Variables like `test_generator`, `CLASS_NAMES`, `IMG_SIZE` were undefined after session restart.

**Solution:**
- Created a dedicated cell to reload all necessary configurations
- Recreated data generators from saved split folders
- Defined all constants (IMG_SIZE, CLASS_NAMES) in a single cell
- Made the code modular to easily re-run specific sections

---

### 5. **Model Loading Issues on Hugging Face**
**Problem:** Multiple deployment errors on Hugging Face Spaces:
- `ModuleNotFoundError: No module named 'cv2'`
- Build errors due to incompatible TensorFlow versions
- Model file not found errors

**Solution:**
- **Removed OpenCV dependency:** Replaced `cv2` with PIL for image processing
- **Optimized requirements.txt:**
```txt
  tensorflow-cpu==2.13.0
  gradio==3.50.2
  numpy==1.24.3
  Pillow==10.0.0
```
- **Simplified model loading:** Added `compile=False` flag and recompiled manually
- **Used Gradio Blocks:** Created custom layout for better UI/UX

---

### 6. **Output Not Displaying in Gradio Interface**
**Problem:** After uploading an image, the output panel remained empty.

**Solution:**
- Changed from `gr.Interface` to `gr.Blocks` for better control
- Added explicit Submit button with clear event handlers
- Implemented both manual submission and auto-submission on image upload
- Added default message: "⏳ Waiting for image upload..."

---

### 7. **Binary Classification Output Confusion**
**Problem:** Initially used `np.argmax()` for binary classification, which is incorrect for single sigmoid output.

**Solution:**
- Changed output interpretation for binary classification:
```python
  pred_prob = model.predict(img)[0][0]
  if pred_prob > 0.5:
      pred_class = 'Mature'
  else:
      pred_class = 'Immature'
```
- Updated confidence calculation accordingly

---

### 8. **Model Download Issues**
**Problem:** Couldn't download model because filename included timestamp, not the generic `model.h5`.

**Solution:**
- Created a script to automatically find the latest model:
```python
  finetuned_models = [f for f in model_files if 'finetuned' in f.lower()]
  latest_model = sorted(finetuned_models)[-1]
```
- Added option to save model with simple name: `tomato_final_model.h5`

---

## Key Learnings
- Learned to build and fine-tune **Transfer Learning models** using MobileNetV2
- Understood the importance of **data preprocessing, normalization, and augmentation** for achieving high accuracy
- Gained experience in **handling session disconnections** and implementing robust checkpointing
- Mastered **debugging deployment issues** on cloud platforms (Hugging Face Spaces)
- Learned to create **user-friendly ML interfaces** using Gradio
- Developed problem-solving skills for **file path management** in nested datasets
- Understood the difference between **multi-class and binary classification** output handling
- Gained experience in **optimizing dependencies** for cloud deployment

## Future Improvements
- Add multi-class classification for tomato quality (Fresh/Rotten)
- Implement object detection to identify multiple tomatoes in one image
- Create a mobile app using TensorFlow Lite
- Build a REST API using FastAPI for integration with other systems
- Add a database to log predictions and track usage statistics


## Live Demo
Try the interactive demo here:  
[Gradio App Link](https://huggingface.co/spaces/Hesham-vision/Tomato-Ripeness-Detection)

