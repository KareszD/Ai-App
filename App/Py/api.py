# api.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
from JB.NeuralNetworkForFoliageDetection.MainPredict import Predictor
import zipfile
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Define absolute paths for uploads and results
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'out')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global progress and result variables
global_progress = 0
global_result = None
processing = False  # Flag to indicate if processing is ongoing

def process_image(file_path, original_filename):
    global global_progress, global_result, processing
    try:
        processing = True
        global_progress = 10  # Starting progress

        print(f"Starting prediction for {file_path}")
        
        # Initialize Predictor instance
        predictor = Predictor(
            Path=file_path,
            OutputDir=RESULTS_FOLDER,
            classNum=6,                # Adjust based on your model
            patchSize=256,             # Adjust based on your model
            IsOutputPOI=False          # Adjust based on your needs
        )
        print("Predictor initialized.")

        # Step 1: Initialization
        global_progress = 10

        # Step 2: Running prediction
        predictor.predict_identification(
            modelPath=os.path.join("Py", "JB", "NeuralNetworkForFoliageDetection", "FoldiKutya", "foldiKutya_500epoch_CLAHE_NoSelfContemination_SameFiledRes_v9mModel"),
            data_path=file_path,
            input_labels=['label1', 'label2'],  # Replace with your actual labels
            needs_splitting=False
        )
        print("Prediction completed.")
        
        # Step 3: Creating ZIP file
        global_progress = 80

        base_filename = f"output"
        shapefile_components = [
            f"{base_filename}.shp",
            f"{base_filename}.dbf",
            f"{base_filename}.shx",
            # Add other components if necessary
        ]

        zip_filename = f"{base_filename}.zip"
        zip_filepath = os.path.join(os.path.dirname(RESULTS_FOLDER), zip_filename)
        print(zip_filepath)
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for component in shapefile_components:
                component_path = os.path.join(os.path.dirname(RESULTS_FOLDER),"out" ,component)
                print(component_path)
                if os.path.exists(component_path):
                    zipf.write(component_path, arcname=component)
                else:
                    print(f"Warning: {component} not found.")

        # Finalizing
        global_progress = 100
        global_result = zip_filename
        print(f"Progress set to 100%. Zip file created: {zip_filename}")
    except Exception as e:
        global_progress = -1  # Indicates error
        global_result = str(e)
        print(f'Error processing task: {e}')
    finally:
        processing = False

@app.route('/upload', methods=['POST'])
def upload_image():
    global processing, global_progress, global_result
    if processing:
        return jsonify({'error': 'Another upload is currently being processed. Please wait.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"Saving uploaded image to: {file_path}")

    image.save(file_path)

    # Initialize progress and result
    global_progress = 0
    global_result = 'Processing...'

    # Start a new thread to process the image
    thread = threading.Thread(target=process_image, args=(file_path, image.filename))
    thread.start()

    return jsonify({'message': 'Image uploaded successfully'}), 200

@app.route('/result', methods=['GET'])
def get_result():
    if global_result and isinstance(global_result, str) and global_result.endswith('.zip'):
        return jsonify({'result': 'Prediction completed successfully.', 'filename': global_result}), 200
    elif global_result:
        # If result is an error message
        return jsonify({'result': global_result}), 200
    else:
        return jsonify({'result': 'Processing not completed yet.'}), 200

@app.route('/results/<filename>', methods=['GET'])
def get_result_file(filename):
    #print(RESULTS_FOLDER)
    return send_from_directory(os.path.dirname(RESULTS_FOLDER), filename, as_attachment=True)

@app.route('/status', methods=['GET'])
def get_status():
    return "API is running.", 200

@app.route('/update_progress', methods=['POST'])
def update_progress():
    global global_progress

    try:
        # Parse the JSON data from the request
        data = request.get_json()
        
        # Ensure 'progress' is in the data
        if 'progress' not in data:
            return jsonify({'error': 'Missing "progress" in request data'}), 400
        global_progress = data['progress']
        
        # Perform any processing you need with the progress value
        print(f"Received progress update: {global_progress}")
        
        # Return a success response
        return jsonify({'message': 'Progress updated successfully'}), 200
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

@app.route('/all_progress', methods=['GET'])
def get_all_progress():
    return jsonify({'progress': global_progress, 'result': global_result}), 200

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
