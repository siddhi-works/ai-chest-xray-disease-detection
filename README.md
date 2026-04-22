# AI-Based Chest X-Ray Disease Detection

## Overview
This project implements a deep learning-based system for classifying chest X-ray images into multiple disease categories including COVID-19, Viral Pneumonia, Lung Opacity, and Normal. The objective is to assist in early detection of respiratory conditions using computer vision techniques.

The model is built using TensorFlow and Keras, based on transfer learning with a pre-trained ResNet50 architecture.

---

## Dataset
Name: COVID-19 Radiography Dataset  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database  

### Structure
```
dataset/
│── COVID/
│── Lung_Opacity/
│── Viral Pneumonia/
│── Normal/
```

---

## Tech Stack
- Python  
- TensorFlow, Keras  
- NumPy  
- OpenCV  
- Flask  
- HTML, CSS  
- Google Colab  

---

## Methodology
- Image preprocessing: resizing (150×150), normalization, RGB conversion  
- Transfer learning using ResNet50  
- Freezing base layers to retain learned features  
- Adding custom classification layers (Dense + Dropout)  
- Multi-class classification using Softmax activation  

---

## Training
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Batch size: 32  
- Epochs: 15–25 (EarlyStopping applied)  

---

## Results
- Accuracy: ~88–92%  
- Evaluation Metrics:  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

The model demonstrates good performance in distinguishing between different chest conditions with stable convergence.

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Usage
```bash
git clone https://github.com/siddhi-works/ai-chest-xray-disease-detection.git
cd ai-chest-xray-disease-detection
python app.py
```

---

## Project Structure
```
│── models/
│── templates/
│── uploads/
│── app.py
│── class_names.pkl
│── README.md
│── requirements.txt
```

---

## Future Improvements
- Fine-tuning deeper layers of ResNet  
- Model explainability (Grad-CAM)  
- Cloud deployment (Render/AWS)  
- Improved UI/UX  
- Larger and more diverse dataset  

---

## Disclaimer
This project is for educational purposes only and is not a substitute for professional medical diagnosis.

---

## License
MIT License  

---

## Authors
Akriti Biswas  
Siddhi Kale  
