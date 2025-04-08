import os
import glob
import cv2
import ffmpeg
from tqdm import tqdm
from clsConfig import clsConfig as cf
from typing import Optional, Callable

class clsFrame2Video:
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.cfg = cf().conf
        self.progress_callback = progress_callback
        self._4k_enabled = self.cfg['PROCESSING_STEPS'].get('4k_processing', False)
        
    def _update_progress(self, progress: int, message: str):
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _get_source_fps(self) -> float:
        try:
            video_path = os.path.join(
                self.cfg['INIT_PATH'],
                'Source',
                f"{self.cfg['FILE_NAME']}.mp4"
            )
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0
        except Exception as e:
            self._update_progress(-1, f"FPS detection error: {str(e)}")
            return 30.0

    def _check_nvenc(self) -> bool:
        try:
            result = os.popen("ffmpeg -encoders | findstr nvenc").read()
            return "hevc_nvenc" in result.lower() or "h264_nvenc" in result.lower()
        except:
            return False

    def convert2Vid(self) -> bool:
        try:
            enhanced_path = os.path.join(
                self.cfg['INIT_PATH'],
                'Enhanced_4K' if self._4k_enabled else 'Enhanced'
            )
            target_path = os.path.join(self.cfg['INIT_PATH'], 'Target')
            os.makedirs(target_path, exist_ok=True)

            original_fps = self._get_source_fps()
            self._update_progress(92, f"Using source frame rate: {original_fps:.2f} FPS")

            frame_files = sorted(glob.glob(os.path.join(enhanced_path, 'frame-*.jpg')))
            if not frame_files:
                self._update_progress(-1, "No enhanced frames found for video creation")
                return False
            
            output_suffix = '_4k_restored' if self._4k_enabled else '_restored'
            final_output = os.path.join(
                target_path, 
                f"{self.cfg['FILE_NAME']}{output_suffix}.mp4"
            )
            
            # Use hardware acceleration if available
            hw_accel = ''
            if self._check_nvenc():
                hw_accel = 'h264_nvenc' if not self._4k_enabled else 'hevc_nvenc'
            
            # Use faster preset for non-4K
            preset = 'fast' if not self._4k_enabled else 'slow'
            
            self._update_progress(95, "Encoding video with hardware acceleration" if hw_accel else "Encoding video")
            
            # Multi-threaded encoding with optimized settings
            (
                ffmpeg
                .input(os.path.join(enhanced_path, 'frame-%04d.jpg'), 
                  framerate=original_fps,
                  thread_queue_size=512)  # Larger buffer for better performance
                .output(final_output,
                       vcodec=hw_accel if hw_accel else 'libx264',
                       crf=18 if self._4k_enabled else 20,
                       preset=preset,
                       pix_fmt='yuv420p10le' if self._4k_enabled else 'yuv420p',
                       threads=8,  # Use multiple threads
                       movflags='+faststart',  # For web playback
                       **{'b:v': '20M'} if self._4k_enabled else {})
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Merge audio if available
            audio_path = os.path.join(self.cfg['INIT_PATH'], 'Source', f"{self.cfg['FILE_NAME']}.mp3")
            if os.path.exists(audio_path):
                self._update_progress(98, "Merging audio track")
                temp_output = os.path.join(target_path, 'temp_output.mp4')
                os.rename(final_output, temp_output)
                
                (
                    ffmpeg
                    .input(temp_output)
                    .output(final_output,
                           vcodec='copy',
                           acodec='aac',
                           audio_bitrate='192k',
                           **{'metadata:s:v:0': 'title=Enhanced Video',
                              'metadata:s:a:0': 'title=Enhanced Audio'})
                    .overwrite_output()
                    .run(quiet=True)
                )
                os.remove(temp_output)
            
            self._update_progress(100, "Video reconstruction completed successfully")
            return True
            
        except Exception as e:
            self._update_progress(-1, f"Video reconstruction failed: {str(e)}")
            return False