# Real-Time Tomato Ripeness Detection

A **deep learning project** that detects the ripeness of tomatoes in real-time using a Convolutional Neural Network (CNN) and provides an interactive web interface.

---

## 🛠 Technologies Used
- **Python**  
- **TensorFlow / Keras**  
- **Gradio**  
- **PIL / NumPy**  

---

## 🔹 How It Works
1. Upload an image of a tomato.  
2. The app preprocesses the image (resizes to 224x224 and normalizes).  
3. A fine-tuned CNN model predicts the ripeness: **Mature** or **Immature**.  
4. The prediction and confidence score are displayed on the interface.

---

## 🚀 Live Demo
Try the live demo here:  
[Gradio App Link](https://huggingface.co/spaces/Hesham-vision/Tomato-Ripeness-Detection)

---

## 💡 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
