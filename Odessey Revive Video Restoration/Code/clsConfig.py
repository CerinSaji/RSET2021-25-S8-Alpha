import json
import os
from typing import Dict, Any

class clsConfig:
    def __init__(self, config_path: str = "config.json", custom_path: str = None):
        self.config_path = config_path
        self.custom_path = custom_path 
        self.conf = self._load_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with comprehensive default values"""
        default_config = {
            "INIT_PATH": self.custom_path if self.custom_path else os.getcwd(),
            "FILE_NAME": "video",
            "MODEL_PATH": os.path.join(os.path.dirname(__file__), "models"),
            "TEMP_PATH": os.path.join(os.getcwd(), "Temp"),
            "PROCESSING_STEPS": {
                "super_resolution": True,
                "colorization": True,
                "face_enhancement": True,
                "temporal_filtering": True,
                "4k_processing": True,
                "deblurring": True,
                "color_correction": True,
                "frame_interpolation": False
            },
            "DEBLUR_PARAMS": {
                "method": "hybrid",
                "iterations": {
                    "richardson_lucy": 15,
                    "wiener": 3
                },
                "kernel_size": 7,
                "motion_compensation": True,
                "temporal_window": 3,
                "sharpening": {
                    "enabled": True,
                    "strength": 0.7,
                    "edge_preserving": True
                },
                "contrast": {
                    "clahe_clip_limit": 2.0,
                    "clahe_grid_size": (16, 16)
                }
            },
            "COLOR_CORRECTION": {
                "contrast": {
                    "enabled": True,
                    "clahe_clip_limit": 2.0,
                    "clahe_grid_size": (8,8)
                },
                "white_balance": {
                    "enabled": True,
                    "method": "grayworld",
                    "saturation_threshold": 0.95
                },
                "color_grading": {
                    "enabled": True,
                    "lut_strength": 0.7,
                    "shadows_boost": 0.1,
                    "highlights_rolloff": 0.9,
                    "midtone_shift": {
                        "red": 0.98,
                        "green": 1.02,
                        "blue": 1.05
                    }
                },
                "colorize": {
                    "enabled": True,
                    "strength": 0.8
                }
            },
            "MODEL_PARAMS": {
                "esrgan": {
                    "model_name": "Real-ESRGAN-x4plus.onnx",
                    "input_size": None,
                    "scale_factor": 4,
                    "half_precision": True
                },
                "deoldify": {
                    "model_name": "DeoldifyVideo_dyn.onnx",
                    "input_size": (384, 384),
                    "color_boost": 1.2,
                    "saturation_factor": 1.1
                },
                "gfpgan": {
                    "model_name": "GFPGANv1.4.onnx",
                    "face_size": (512, 512),
                    "eye_preservation": True
                }
            },
            "HARDWARE": {
                "use_gpu": True,
                "gpu_id": 0,
                "gpu_mem_limit": 8192,
                "batch_size": 4,
                "num_threads": 4,
                "memory_optimization": True
            },
            "OUTPUT": {
                "4k_resolution": (3840, 2160),
                "output_codec": "hevc_nvenc" if self._check_nvenc() else "libx265",
                "color_depth": 10,
                "output_format": "mp4",
                "quality_preset": "slow",
                "bitrate": "20M",
                "interpolation_factor": 1,
                "keep_temp_files": False
            },
            "DEBUG": {
                "save_intermediate": False,
                "log_level": "INFO",
                "visualize_motion": False,
                "profile_performance": False
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    user_config = json.load(f)
                    return self._deep_merge(default_config, user_config)
            return default_config
        except Exception as e:
            print(f"Config loading error: {str(e)}, using defaults")
            return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge dictionaries"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def _create_directories(self) -> None:
        base_path = self.conf["INIT_PATH"]
        dirs = [
            self.conf["MODEL_PATH"],
            os.path.join(self.conf["INIT_PATH"], "Source"),
            self.conf["TEMP_PATH"],
            os.path.join(self.conf["INIT_PATH"], "Enhanced_4K"),
            os.path.join(self.conf["INIT_PATH"], "Target"),
            os.path.join(self.conf["INIT_PATH"], "Debug"),
            os.path.join(self.conf["INIT_PATH"], "Logs")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _check_nvenc(self) -> bool:
        try:
            result = os.popen("ffmpeg -encoders | findstr nvenc").read()
            return "hevc_nvenc" in result.lower()
        except:
            return False

    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        try:
            with open(save_path, "w") as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False