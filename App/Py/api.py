from flask import Flask, request, jsonify
import os
import time
from threading import Thread
import sys

from JB.NeuralNetworkForFoliageDetection.MainPredict import Predictor

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store prediction progress
progress_status = {}
prediction_results = {}

def process_image(file_path, task_id):
    try:
        # Initialize your Predictor instance
        predictor = Predictor(
            Path=file_path,
            OutputDir='prediction_results',  # Specify your output directory
            classNum=6,                      # Adjust as needed
            patchSize=256,                   # Adjust as needed
            IsOutputPOI=False                # Adjust as needed
        )

        # Update progress status
        progress_status[task_id] = 0

        # Call the predict_identification method
        # You might need to adjust the arguments based on your implementation
        predictor.predict_identification(
            modelPath='App/Py/JB/NeuralNetworkForFoliageDetection/FoldiKutya/foldiKutya_100epoch_noFilter_yolov8_pose_BG_10%_normRes_trainOnPose_v2',  # Path to your model file
            data_path=file_path,
            input_labels=['label1', 'label2'],  # Replace with your actual labels
            needs_splitting=False               # Set to True if needed
        )

        # Update progress to 100% upon completion
        progress_status[task_id] = 100
        prediction_results[task_id] = 'Prediction completed successfully.'

    except Exception as e:
        # Handle exceptions and update progress status
        progress_status[task_id] = -1  # Indicates an error
        prediction_results[task_id] = str(e)
        print(f'Error processing task {task_id}: {e}')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate a unique task ID
    task_id = str(time.time()).replace('.', '')

    # Save the image to the upload folder
    filename = f'{task_id}_{image.filename}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)

    # Start a new thread to process the image
    thread = Thread(target=process_image, args=(file_path, task_id))
    thread.start()

    return jsonify({'message': 'Image uploaded successfully', 'task_id': task_id}), 200

@app.route('/progress', methods=['GET'])
def get_progress():
    task_id = request.args.get('task_id')
    if task_id in progress_status:
        progress = progress_status[task_id]
        return jsonify({'task_id': task_id, 'progress': progress})
    else:
        return jsonify({'error': 'Invalid task ID'}), 400

@app.route('/result', methods=['GET'])
def get_result():
    task_id = request.args.get('task_id')
    if task_id in prediction_results:
        result = prediction_results[task_id]
        return jsonify({'task_id': task_id, 'result': result})
    else:
        return jsonify({'error': 'Result not available or invalid task ID'}), 400

@app.route('/status', methods=['GET'])
def get_status():
    return "up"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
