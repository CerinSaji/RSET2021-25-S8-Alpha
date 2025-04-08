import os
import subprocess
from clsConfig import clsConfig as cf
from typing import Optional, Callable, Tuple, List
import logging
from pathlib import Path

class clsVideo2Frame:
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.cfg = cf().conf
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
    def _update_progress(self, progress: int, message: str):
        """Update progress with logging and callback handling."""
        self.logger.info(message)
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _find_video_file(self) -> str:
        """Find the video file with flexible naming and extensions."""
        base_path = Path(self.cfg['INIT_PATH'])
        source_path = base_path / 'Source'
        
        # Get all possible video files in both locations
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        possible_files = []
        
        # Check in Source directory first
        if source_path.exists():
            for ext in video_extensions:
                possible_files.extend(source_path.glob(f'*{ext}'))
        
        # Check in main directory
        for ext in video_extensions:
            possible_files.extend(base_path.glob(f'*{ext}'))
        
        # If we have a specific FILE_NAME in config, prioritize it
        if 'FILE_NAME' in self.cfg:
            specific_files = [f for f in possible_files 
                            if f.stem == self.cfg['FILE_NAME']]
            if specific_files:
                return str(specific_files[0].resolve())
        
        # If no specific file found but we have videos available
        if possible_files:
            # Try to find the newest video file
            newest_file = max(possible_files, key=os.path.getmtime)
            self._update_progress(0, f"Using video file: {newest_file.name}")
            return str(newest_file.resolve())
        
        # No video files found at all
        available_files = []
        for path in [base_path, source_path]:
            if path.exists():
                available_files.extend(f.name for f in path.glob('*') if f.is_file())
        
        error_msg = (
            "No supported video files found (.mp4, .mov, .avi, .mkv, .webm)\n"
            f"Searched in:\n- {base_path}\n- {source_path}\n\n"
            "Available files:\n" + "\n".join(f"  - {f}" for f in available_files[:20])
        )
        raise FileNotFoundError(error_msg)

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        try:
            base = Path(video_path).with_suffix('')
            audio_path = f"{base}.mp3"
            
            self._update_progress(0, f"Extracting audio to {audio_path}")
            
            result = subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-q:a', '0', '-map', 'a', audio_path
            ], capture_output=True, text=True, check=True)
            
            if not Path(audio_path).exists():
                raise RuntimeError(f"Audio file not created at {audio_path}")
                
            return audio_path
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Audio extraction failed: {e.stderr}"
            self._update_progress(-1, error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            self._update_progress(-1, f"Audio extraction error: {str(e)}")
            raise

    def _get_video_info(self, video_path: str) -> Tuple[Optional[int], Optional[int], float, Optional[float]]:
        """Get video metadata using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ], capture_output=True, text=True, check=True)
            
            parts = [x.strip() for x in result.stdout.split('\n') if x.strip()]
            if len(parts) != 4:
                raise ValueError(f"Unexpected ffprobe output: {result.stdout}")
                
            width = int(parts[0]) if parts[0] else None
            height = int(parts[1]) if parts[1] else None
            fps = float(eval(parts[2])) if parts[2] else 30.0
            duration = float(parts[3]) if parts[3] else None
            
            return width, height, fps, duration
            
        except subprocess.CalledProcessError as e:
            self._update_progress(-1, f"FFprobe error: {e.stderr}")
            return None, None, 30.0, None
        except Exception as e:
            self._update_progress(-1, f"Video info error: {str(e)}")
            return None, None, 30.0, None

    def _analyze_ffmpeg_error(self, error_output: str) -> str:
        """Analyze FFmpeg error output and return helpful message."""
        error_msg = "Frame extraction failed"
        
        if "Invalid data found" in error_output:
            error_msg += "\n- The video file might be corrupted or in an unsupported format"
        elif "Permission denied" in error_output:
            error_msg += "\n- Permission denied when writing frames"
        elif "No such file or directory" in error_output:
            error_msg += "\n- Output directory doesn't exist or is inaccessible"
        elif "Invalid argument" in error_output:
            error_msg += "\n- Possible codec or format incompatibility"
        elif "Operation not permitted" in error_output:
            error_msg += "\n- Check file permissions and disk space"
        
        if "h264" in error_output.lower():
            error_msg += "\n- H.264 codec issue detected - try installing proper codecs"
        if "hevc" in error_output.lower():
            error_msg += "\n- HEVC/H.265 codec issue detected - may need additional codecs"
        
        return error_msg + f"\n\nTechnical details:\n{error_output[:500]}"

    def genFrame(self) -> bool:
        """Main method to extract frames and audio from video."""
        try:
            # Find video file with flexible approach
            video_path = self._find_video_file()
            self._update_progress(5, f"Using video file: {video_path}")
            
            # Create temp directory
            temp_path = Path(self.cfg['INIT_PATH']) / 'Temp'
            temp_path.mkdir(exist_ok=True, parents=True)
            
            # Get video info
            width, height, fps, duration = self._get_video_info(video_path)
            self._update_progress(10, 
                f"Video info: {width or '?'}x{height or '?'} "
                f"at {fps:.2f}fps, duration: {duration or '?'}s"
            )
            
            # Extract audio
            try:
                audio_path = self._extract_audio(video_path)
                self._update_progress(20, f"Audio extracted to: {audio_path}")
            except Exception as e:
                self._update_progress(20, f"Audio extraction skipped: {str(e)}")
                audio_path = None
            
            # Extract frames
            frame_pattern = str(temp_path / 'frame-%04d.jpg')
            self._update_progress(30, f"Extracting frames to: {frame_pattern}")
            
            try:
                result = subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-vsync', '0',
                    '-qscale:v', '2',
                    '-sws_flags', 'lanczos+accurate_rnd+full_chroma_int',
                    frame_pattern
                ], capture_output=True, text=True, check=True)
                
                frame_count = len(list(temp_path.glob('frame-*.jpg')))
                if frame_count == 0:
                    raise RuntimeError("No frames extracted - check video format")
                
                self._update_progress(50, f"Extracted {frame_count} frames successfully")
                return True
                
            except subprocess.CalledProcessError as e:
                error_msg = self._analyze_ffmpeg_error(e.stderr)
                self._update_progress(-1, error_msg)
                raise RuntimeError(error_msg) from e
                
        except Exception as e:
            self._update_progress(-1, f"Processing failed: {str(e)}")
            self.logger.error("Video processing error", exc_info=True)
            return False