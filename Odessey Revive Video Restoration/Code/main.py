import os
import time
import logging
import traceback
import shutil
from typing import Optional, Callable
from clsVideo2Frame import clsVideo2Frame
from clsFrameEnhance import clsFrameEnhance
from clsFrame2Video import clsFrame2Video
from clsConfig import clsConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_enhancement.log'),
            logging.StreamHandler()
        ]
    )

def ensure_directory_structure(working_dir: str):
    """Create all required directories if they don't exist"""
    required_dirs = [
        os.path.join(working_dir, 'Source'),
        os.path.join(working_dir, 'Target'),
        os.path.join(working_dir, 'Temp'),
        os.path.join(working_dir, 'Enhanced_4K'),
        os.path.join(working_dir, 'Debug')
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Verified directory: {dir_path}")

def prepare_source_file(input_path: str, working_dir: str) -> str:
    """
    Prepare the source video file and return its final path
    Handles cases where file might already exist in Source directory
    """
    file_name = os.path.basename(input_path)
    source_path = os.path.join(working_dir, 'Source', file_name)
    
    # If the file is already in Source directory, use it directly
    if os.path.normpath(input_path) == os.path.normpath(source_path):
        return source_path
        
    # If destination exists, remove it first
    if os.path.exists(source_path):
        logging.info(f"Removing existing file at: {source_path}")
        os.remove(source_path)
    
    # Move/copy the file
    try:
        shutil.move(input_path, source_path)
    except Exception as e:
        logging.warning(f"Move failed, trying copy: {str(e)}")
        shutil.copy2(input_path, source_path)
        os.remove(input_path)
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Failed to prepare source at {source_path}")
    
    return source_path

def process_video(input_path: str, output_path: str, progress_callback: Optional[Callable] = None) -> bool:
    """Enhanced video processing pipeline with robust file handling"""
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        working_dir = os.path.dirname(input_path)
        ensure_directory_structure(working_dir)
        
        # Prepare source file
        source_path = prepare_source_file(input_path, working_dir)
        logger.info(f"Using source video at: {source_path}")
        
        # Initialize configuration
        file_name = os.path.splitext(os.path.basename(source_path))[0]
        config = clsConfig(custom_path=working_dir)
        config.conf['FILE_NAME'] = file_name
        
        # Processing pipeline
        stages = [
            ("Extracting frames", 10, 30, lambda: clsVideo2Frame(progress_callback).genFrame()),
            ("Enhancing frames", 40, 70, lambda: clsFrameEnhance(progress_callback).doEnhance()),
            ("Reconstructing video", 80, 95, lambda: clsFrame2Video(progress_callback).convert2Vid())
        ]

        for name, start_progress, end_progress, stage_func in stages:
            if progress_callback:
                progress_callback(start_progress, f"Starting {name.lower()}...")
                
            logger.info(f"\n[{name.upper()}]")
            if not stage_func():
                raise RuntimeError(f"{name} failed")

            if progress_callback:
                progress_callback(end_progress, f"{name} completed")

        # Handle final output
        output_suffix = '_4k_restored' if config.conf['PROCESSING_STEPS']['4k_processing'] else '_restored'
        final_output = os.path.join(working_dir, 'Target', f"{file_name}{output_suffix}.mp4")
        
        if not os.path.exists(final_output):
            raise FileNotFoundError(f"Output video not created at: {final_output}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(final_output, output_path)
        
        # Cleanup
        if not config.conf['DEBUG']['keep_temp_files']:
            logger.info("Cleaning temporary files...")
            for temp_dir in ['Temp', 'Enhanced_4K', 'Debug']:
                shutil.rmtree(os.path.join(working_dir, temp_dir), ignore_errors=True)

        logger.info(f"Processing completed in {time.time()-start_time:.2f}s")
        if progress_callback:
            progress_callback(100, "Processing completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        if progress_callback:
            progress_callback(-1, f"Processing failed: {str(e)}")
        return False