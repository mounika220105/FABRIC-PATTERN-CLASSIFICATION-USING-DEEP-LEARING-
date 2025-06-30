import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/assets/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    # Load the trained model
    model = load_model('model_50.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# List of labels
labels = [
    'Chancellor Hall', 
    'Chancellor Tower', 
    'Clock Tower', 
    'Colorful Stairway',
    'DKP Baru', 
    'Library', 
    'Recital Hall', 
    'UMS Aquarium',
    'UMS Mosque'
]

# Helper function to predict
def get_model_prediction(image_path):
    try:
        logger.debug(f"Processing image: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None, None
            
        # Load and preprocess image
        img = load_img(image_path, target_size=(255, 255))
        x = img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        
        logger.debug(f"Input shape: {x.shape}")
        
        # Make prediction
        predictions = model.predict(x, verbose=0)
        logger.debug(f"Raw predictions shape: {predictions.shape}")
        logger.debug(f"Raw predictions: {predictions}")
        
        predicted_index = predictions.argmax()
        confidence = float(np.max(predictions))
        
        # Verify prediction index
        if predicted_index >= len(labels):
            logger.error(f"Prediction index {predicted_index} is out of range for labels {len(labels)}")
            # Map to valid range if needed
            predicted_index = predicted_index % len(labels)
            logger.info(f"Mapped to valid index: {predicted_index}")
            
        predicted_class = labels[predicted_index]
        
        logger.info(f"Prediction successful: {predicted_class} with confidence {confidence}")
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return None, None
        
# Add helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def home():
    return render_template("home.html")

# Predict page
@app.route('/predict_page')
def predict_page():
    return render_template("predict.html")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def prediction():
    try:
        logger.info("Starting prediction request")
        
        # Check if image was uploaded
        if 'image' not in request.files:
            logger.error("No image key in request.files")
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        logger.debug(f"Received file: {file.filename}")
        
        if file.filename == '':
            logger.error("Empty filename received")
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Create full upload path
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Upload path: {filepath}")
        
        # Save file
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            logger.info(f"File saved successfully at: {filepath}")
        except Exception as save_error:
            logger.error(f"Failed to save file: {str(save_error)}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Get prediction
        try:
            prediction, confidence = get_model_prediction(filepath)
            if prediction is None:
                logger.error("Model prediction returned None")
                return jsonify({'error': 'Model prediction failed'}), 500
                
            # Create response
            response = {
                'prediction': prediction,
                'confidence': f"{confidence * 100:.2f}%",
                'image_path': '/'.join(filepath.split(os.sep))
            }
            logger.info(f"Prediction successful: {response}")
            return jsonify(response)
            
        except Exception as pred_error:
            logger.error(f"Prediction failed: {str(pred_error)}")
            return jsonify({'error': 'Failed to process image'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Server error occurred'}), 500

# Add debug configuration
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

