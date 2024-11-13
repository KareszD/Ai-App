from flask import Flask, request, jsonify
import os
import time
from threading import Thread

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store prediction progress
progress_status = {}

def process_image(file_path, task_id):
    # Simulate a long prediction process
    total_steps = 100
    for step in range(total_steps):
        time.sleep(1)  # Simulate time taken for each step (1 second per step)
        progress_status[task_id] = (step + 1) / total_steps * 100
        print(f'Task {task_id} progress: {progress_status[task_id]}%')  # Output progress to console
    progress_status[task_id] = 100
    # Optionally, remove the task_id after completion to clean up
    # del progress_status[task_id]

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image to the upload folder
    task_id = str(time.time()).replace('.', '')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(file_path)

    # Start a new thread to process the image
    thread = Thread(target=process_image, args=(file_path, task_id))
    thread.start()

    return jsonify({'message': 'Image uploaded successfully', 'task_id': task_id}), 200

@app.route('/progress', methods=['GET'])
def get_progress():
    if progress_status:
        # Calculate average progress across all tasks
        avg_progress = sum(progress_status.values()) / len(progress_status)
    else:
        avg_progress = 0
    return jsonify({'progress': avg_progress})

@app.route('/status', methods=['GET'])
def get_status():
    return "up"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
