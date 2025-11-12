import tensorflow as tf
import numpy as np
import cv2
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

try:
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
        MODEL_PATH = os.path.join(PROJECT_ROOT, params['output']['model_file'])
        IMG_SIZE = params['model']['img_size']
except Exception as e:
    print(f"Warning: Could not load params.yaml for prediction. Error: {e}")
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'pneumonia_detector.keras')
    IMG_SIZE = 150

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully for prediction.")
except Exception as e:
    MODEL = None
    print(f"CRITICAL ERROR: Failed to load model from {MODEL_PATH}. Prediction will fail. {e}")

def preprocess_image(image_data):
    """
    Reads image data, resizes it, and normalizes it for the model.
    """
    np_img = np.frombuffer(image_data.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE) 
    
    if img is None:
        raise ValueError("Could not decode image data.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img = img.astype('float32') / 255.0

    return np.expand_dims(img, axis=0)

def predict_pneumonia(image_data):
    """
    Preprocesses the image and makes a prediction using the loaded model.
    Returns the predicted class and the confidence percentage.
    """
    if MODEL is None:
        return "Model Unavailable", 0.0, 0.0

    try:
        processed_image = preprocess_image(image_data)
        
        prediction = MODEL.predict(processed_image)
        
        probability_pneumonia = prediction[0][0]
        
        if probability_pneumonia >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = probability_pneumonia
        else:
            predicted_class = "NORMAL"
            confidence = 1.0 - probability_pneumonia
        
        confidence_percent = confidence * 100
        
        return predicted_class, confidence_percent, probability_pneumonia

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "Error Processing Image", 0.0, 0.0