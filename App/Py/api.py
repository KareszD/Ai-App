# api.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import threading
import sys
from JB.NeuralNetworkForFoliageDetection.MainPredict import Predictor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute path to the directory where api.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for uploads and results
UPLOAD_FOLDER = os.path.join( 'uploads')

RESULTS_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'out')
print(RESULTS_FOLDER)
# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Update Python path to include Predictor module


# Dictionaries to track progress and results
progress_status = {}
prediction_results = {}

def process_image(file_path, task_id):
    try:
        print(f"[Task {task_id}] Starting prediction for {file_path}")
        
        # Initialize Predictor instance
        predictor = Predictor(
            Path=file_path,
            OutputDir=RESULTS_FOLDER,
            classNum=6,                # Adjust based on your model
            patchSize=256,             # Adjust based on your model
            IsOutputPOI=False          # Adjust based on your needs
        )
        print(f"[Task {task_id}] Predictor initialized.")

        # Update progress
        progress_status[task_id] = 10
        print(f"[Task {task_id}] Progress set to 10%.")

        # Run prediction
        predictor.predict_identification(
            modelPath=os.path.join("Py\\JB\\NeuralNetworkForFoliageDetection\\FoldiKutya\\foldiKutya_500epoch_CLAHE_NoSelfContemination_SameFiledRes_v9mModel"),  # Ensure the model path is correct
            data_path=file_path,
            input_labels=['label1', 'label2'],  # Replace with your actual labels
            needs_splitting=False
        )
        print(f"[Task {task_id}] Prediction completed.")

        # Update progress to 100%
        progress_status[task_id] = 100
        prediction_results[task_id] = 'Prediction completed successfully.'
        print(f"[Task {task_id}] Progress set to 100%.")

    except Exception as e:
        progress_status[task_id] = -1  # Indicates error
        prediction_results[task_id] = str(e)
        print(f'[Task {task_id}] Error processing task: {e}')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Save the image with the task_id prefix
    filename = f'{task_id}_{image.filename}'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(file_path)

    image.save(file_path)
    # Initialize progress and results
    progress_status[task_id] = 0
    prediction_results[task_id] = 'Processing...'

    # Start a new thread to process the image
    thread = threading.Thread(target=process_image, args=(file_path, task_id))
    thread.start()

    return jsonify({'message': 'Image uploaded successfully', 'task_id': task_id}), 200

@app.route('/progress', methods=['GET'])
def get_progress():
    task_id = request.args.get('task_id')
    if task_id in progress_status:
        return jsonify({'task_id': task_id, 'progress': progress_status[task_id]}), 200
    else:
        return jsonify({'error': 'Invalid task ID'}), 400

@app.route('/result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    if task_id in prediction_results:
        result = prediction_results[task_id]
        return jsonify({'task_id': task_id, 'result': result}), 200
    else:
        return jsonify({'error': 'Result not available or invalid task ID'}), 400

@app.route('/results/<filename>', methods=['GET'])
def get_result_image(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/status', methods=['GET'])
def get_status():
    return "API is running.", 200

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
