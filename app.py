from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load model + class names
model = load_model('models/final_model.h5', compile=False)
class_names = pickle.load(open("models/class_names.pkl", "rb"))

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Image processing
def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))

    # ResNet preprocessing
    image_input = preprocess_input(image_resized)
    image_input = np.expand_dims(image_input, axis=0)

    predictions = model.predict(image_input, verbose=0)
    predicted_index = np.argmax(predictions)

    confidence_score = predictions[0][predicted_index]
    predicted_label = class_names[predicted_index]

    return predicted_label, confidence_score


# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_label, confidence_score = process_image(file_path)

        return render_template(
            'result.html',
            filename=filename,
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )


@app.route('/camera')
def camera():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)