from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import threading
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
import shutil
from werkzeug.utils import secure_filename
from main import process_video
from clsConfig import clsConfig

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['COMPARISONS_FOLDER'] = 'static/comparisons'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Simple in-memory progress tracking
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'metrics': None,
    'original_video': None,
    'processed_video': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_video_metrics(original_path, processed_path):
    try:
        # Extract sample frames from both videos
        cap_orig = cv2.VideoCapture(original_path)
        cap_proc = cv2.VideoCapture(processed_path)
        
        frame_count = 0
        psnr_values = []
        ssim_values = []
        
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_proc, frame_proc = cap_proc.read()
            
            if not ret_orig or not ret_proc:
                break
                
            if frame_count % 10 == 0:  # Sample every 10th frame
                if frame_orig.shape != frame_proc.shape:
                    frame_proc = cv2.resize(frame_proc, (frame_orig.shape[1], frame_orig.shape[0]))
                
                gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                gray_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
                
                psnr_val = psnr(frame_orig, frame_proc)
                ssim_val = ssim(gray_orig, gray_proc, 
                              data_range=gray_proc.max() - gray_proc.min())
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
            
            frame_count += 1
        
        cap_orig.release()
        cap_proc.release()
        
        if not psnr_values:
            return None
            
        return {
            'avg_psnr': np.mean(psnr_values),
            'avg_ssim': np.mean(ssim_values),
            'total_frames': frame_count,
            'sampled_frames': len(psnr_values)
        }
    except Exception as e:
        print(f"Video metrics calculation error: {str(e)}")
        return None

def process_video_task(input_path, output_path):
    global processing_status
    
    def progress_callback(progress, message):
        processing_status['progress'] = progress
        processing_status['message'] = message
    
    try:
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': 'Starting processing...',
            'metrics': None,
            'original_video': os.path.basename(input_path),
            'processed_video': None
        })
        
        success = process_video(input_path, output_path, progress_callback)
        
        if success:
            metrics = calculate_video_metrics(input_path, output_path)
            
            processing_status.update({
                'progress': 100,
                'message': 'Processing complete!',
                'metrics': metrics,
                'processed_video': os.path.basename(output_path),
                'is_processing': False
            })
        else:
            processing_status.update({
                'message': 'Processing failed',
                'is_processing': False
            })

    except Exception as e:
        processing_status.update({
            'message': f'Error: {str(e)}',
            'is_processing': False
        })

@app.route('/')
def index():
    config = clsConfig()
    return render_template('index.html', cfg=config.conf)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'enhanced_{filename}')
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
        file.save(input_path)
        
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': 'Starting processing...',
            'metrics': None,
            'original_video': filename,
            'processed_video': None
        })
        
        thread = threading.Thread(target=process_video_task, args=(input_path, output_path))
        thread.start()
        
        return jsonify({
            'message': 'File uploaded and processing started',
            'filename': filename
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/progress')
def get_progress():
    return jsonify(processing_status)

@app.route('/results/<filename>')
def show_results(filename):
    original_video = filename
    processed_video = f'enhanced_{filename}'
    metrics = processing_status.get('metrics', None)
    
    return render_template('results.html',
                         original_video=original_video,
                         processed_video=processed_video,
                         metrics=metrics)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed_video/<filename>')
def serve_processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['COMPARISONS_FOLDER'], exist_ok=True)
    app.run(debug=True)