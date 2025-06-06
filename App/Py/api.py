from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import threading
from JB.NeuralNetworkForFoliageDetection.MainPredict import Predictor
from JB.NeuralNetworkForFoliageDetection.LabelingTools import Label, Labels
#from JB.NeuralNetworkForFoliageDetection import MainPredict
#from JB.NeuralNetworkForFoliageDetection import LabelingTools
import zipfile
from pathlib import Path
import time
import torch
import encodings
labels = Labels(())
labels.ReadJSON("Data/labels.json")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'out')
print(RESULTS_FOLDER)
print(os.path.join((BASE_DIR), 'out'))
print(os.path.join(os.path.dirname(RESULTS_FOLDER),"out" ))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

global_progress = 0
global_result = None
processing = False  

def process_image(file_path, original_filename):
    global global_progress, global_result, processing
    try:
        processing = True
        global_progress = 10  

        print(f"Starting prediction for {file_path}")
        
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPUs available. Using CPU.")

        #print(f"Starting prediction for {file_path}")
        
        predictor = Predictor(
            Path=file_path,
            OutputDir=RESULTS_FOLDER,
            classNum=6,                
            patchSize=256,             
            IsOutputPOI=False         
        )
        print("Predictor initialized.")
       
        global_progress = 0
        
        start_time = time.time()
        print("Starting prediction...")
        
        predictor.predict_identification(
            modelPath=os.path.join(BASE_DIR, "JB", "NeuralNetworkForFoliageDetection", "FoldiKutya", "foldiKutya_500epoch_CLAHE_NoSelfContemination_SameFiledRes_v9mModel"),
            data_path=file_path,
            input_labels=labels.list, 
            needs_splitting=False
        )
        print("Prediction completed.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Prediction completed in {elapsed_time:.2f} seconds.")
                
        global_progress = 99

        base_filename = f"output"
        shapefile_components = [
            f"{base_filename}.shp",
            f"{base_filename}.dbf",
            f"{base_filename}.shx",
            
        ]

        zip_filename = f"{base_filename}.zip"
        zip_filepath = os.path.join((BASE_DIR), 'out', zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for component in shapefile_components:
                component_path = os.path.join((BASE_DIR), 'out',component)
                if os.path.exists(component_path):
                    zipf.write(component_path, arcname=component)
                else:
                    print(f"Warning: {component_path} not found.")
        
        global_progress = 100
        global_result = zip_filename
        print(f"Progress set to 100%. Zip file created: {zip_filename}")
    except Exception as e:
        global_progress = -1  
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
   
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"Saving uploaded image to: {file_path}")

    image.save(file_path)
  
    global_progress = 0
    global_result = 'Processing...'
   
    thread = threading.Thread(target=process_image, args=(file_path, image.filename))
    thread.start()

    return jsonify({'message': 'Image uploaded successfully'}), 200

@app.route('/results/<filename>', methods=['GET'])
def get_result_file(filename):
    return send_from_directory(
    os.path.join(BASE_DIR, 'out'), filename, as_attachment=True
)

@app.route('/status', methods=['GET'])
def get_status():
    return "API is running.", 200

@app.route('/update_progress', methods=['POST'])
def update_progress():
    global global_progress

    try:       
        data = request.get_json()
                
        if 'progress' not in data:
            return jsonify({'error': 'Missing "progress" in request data'}), 400
        global_progress = data['progress']
        
        #print(f"Received progress update: {global_progress}")
               
        return jsonify({'message': 'Progress updated successfully'}), 200
    except Exception as e:
        
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

@app.route('/all_progress', methods=['GET'])
def get_all_progress():
    return jsonify({'progress': global_progress, 'result': global_result}), 200

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
