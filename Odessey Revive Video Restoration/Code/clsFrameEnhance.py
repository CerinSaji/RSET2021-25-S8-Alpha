import os
import cv2
import numpy as np
import glob
import onnxruntime as ort
from tqdm import tqdm
import traceback
import gc
from skimage import restoration
from clsConfig import clsConfig as cf
from typing import Optional, Callable, Tuple, Dict, Any

class clsFrameEnhance:
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.cfg = cf().conf
        self.progress_callback = progress_callback
        self.debug_dir = os.path.join(self.cfg['INIT_PATH'], 'Debug')
        os.makedirs(self.debug_dir, exist_ok=True)
        self.target_scale = self.cfg['OUTPUT'].get('scale_factor', 4)
        self.sessions = self._load_models()
        self.temporal_filter = self.TemporalFilter()
        self._4k_enabled = self.cfg['PROCESSING_STEPS'].get('4k_processing', False)
        self.prev_y_channel = None
        self._init_face_detection()

    class TemporalFilter:
        def __init__(self, alpha: float = 0.15):
            self.alpha = alpha
            self.prev_frame = None
            self.prev_prev_frame = None

        def stabilize(self, current_frame: np.ndarray) -> np.ndarray:
            if self.prev_frame is None:
                self.prev_frame = current_frame.copy()
                return current_frame

            if self.prev_prev_frame is None:
                blended = cv2.addWeighted(current_frame, self.alpha, 
                                       self.prev_frame, 1 - self.alpha, 0)
            else:
                blended = cv2.addWeighted(
                    cv2.addWeighted(current_frame, 0.6, self.prev_frame, 0.4, 0),
                    0.7,
                    self.prev_prev_frame,
                    0.3,
                    0
                )

            self.prev_prev_frame = self.prev_frame.copy()
            self.prev_frame = blended.copy()
            return blended

    def _init_face_detection(self):
        if not self.cfg['PROCESSING_STEPS']['face_enhancement']:
            return

        try:
            # Try DNN first
            if hasattr(cv2, 'dnn'):
                prototxt_path = os.path.join(self.cfg['MODEL_PATH'], "deploy.prototxt")
                caffemodel_path = os.path.join(self.cfg['MODEL_PATH'], "res10_300x300_ssd_iter_140000.caffemodel")
                if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                    self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                    self._log("Using DNN-based face detector")
                else:
                    self._log("Face detection model files not found. Using Haar cascades.")
            else:
                self._log("OpenCV DNN module not available. Using Haar cascades.")

            # Fallback to Haar cascades
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

        except Exception as e:
            self._log(f"Failed to initialize face detector: {str(e)}", is_error=True)
            self.cfg['PROCESSING_STEPS']['face_enhancement'] = False

    def _log(self, message: str, is_error: bool = False):
        if is_error:
            print(f"Error: {message}")
            traceback.print_exc()
        else:
            print(message)

        if self.progress_callback:
            self.progress_callback(-1 if is_error else 0, message)

    def _load_models(self) -> Dict[str, ort.InferenceSession]:
        sessions = {}
        try:
            available_providers = ort.get_available_providers()
            self._log(f"Available ONNX Runtime providers: {', '.join(available_providers)}")

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            if 'CUDAExecutionProvider' in available_providers and self.cfg['HARDWARE']['use_gpu']:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': self.cfg['HARDWARE']['gpu_id'],
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': self.cfg['HARDWARE']['gpu_mem_limit'] * 1024 * 1024,
                        'cudnn_conv_algo_search': 'DEFAULT',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']
                self._log("Using CPU execution")

            if self.cfg['PROCESSING_STEPS']['super_resolution']:
                try:
                    sessions['sr'] = self._load_model(
                        model_name=self.cfg['MODEL_PARAMS']['esrgan']['model_name'],
                        providers=providers,
                        options=options,
                        model_type="ESRGAN"
                    )
                except Exception as e:
                    self._log(f"Failed to load ESRGAN with GPU, falling back to CPU: {str(e)}")
                    sessions['sr'] = self._load_model(
                        model_name=self.cfg['MODEL_PARAMS']['esrgan']['model_name'],
                        providers=['CPUExecutionProvider'],
                        options=options,
                        model_type="ESRGAN"
                    )

            if self.cfg['PROCESSING_STEPS']['colorization']:
                sessions['colorizer'] = self._load_model(
                    model_name=self.cfg['MODEL_PARAMS']['deoldify']['model_name'],
                    providers=providers,
                    options=options,
                    model_type="DeOldify"
                )

            if self.cfg['PROCESSING_STEPS']['face_enhancement']:
                sessions['face'] = self._load_model(
                    model_name=self.cfg['MODEL_PARAMS']['gfpgan']['model_name'],
                    providers=providers,
                    options=options,
                    model_type="GFPGAN"
                )

        except Exception as e:
            self._log(f"Model loading failed: {str(e)}", is_error=True)
            raise

        return sessions

    def _load_model(self, model_name: str, providers: list, options: ort.SessionOptions, model_type: str) -> ort.InferenceSession:
        model_path = os.path.join(self.cfg['MODEL_PATH'], model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_type} model not found at {model_path}")

        session = ort.InferenceSession(model_path, providers=providers, sess_options=options)
        self._log(f"{model_type} loaded with {session.get_providers()[0]}")
        return session

    def _motion_deblur(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0].astype(np.float32)

            if self.cfg['DEBUG']['save_intermediate']:
                cv2.imwrite(os.path.join(self.debug_dir, f'original_Y_{frame_number:04d}.jpg'), ycrcb[:, :, 0])

            method = self.cfg['DEBLUR_PARAMS']['method']
            if method == 'wiener':
                kernel = np.ones((5, 5), np.float32) / 25
                y_blurred = cv2.filter2D(y_channel, -1, kernel)
                y_deblurred = cv2.ximgproc.wienerFilter(y_blurred, None, 5, 5)
            elif method == 'richardson_lucy':
                kernel = np.ones((5, 5), np.float32) / 25
                y_deblurred = restoration.richardson_lucy(
                    y_channel / 255.0,
                    kernel,
                    iterations=self.cfg['DEBLUR_PARAMS']['iterations']['richardson_lucy']
                ) * 255
            else:  # hybrid
                kernel = np.array([[0, -0.25, 0],
                                 [-0.25, 2, -0.25],
                                 [0, -0.25, 0]])
                y_deblurred = cv2.filter2D(y_channel, -1, kernel)

            y_deblurred = np.clip(y_deblurred, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=self.cfg['DEBLUR_PARAMS']['contrast']['clahe_clip_limit'],
                tileGridSize=self.cfg['DEBLUR_PARAMS']['contrast']['clahe_grid_size']
            )
            y_enhanced = clahe.apply(y_deblurred)

            ycrcb[:, :, 0] = y_enhanced
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            if self.cfg['DEBUG']['save_intermediate']:
                cv2.imwrite(os.path.join(self.debug_dir, f'deblurred_Y_{frame_number:04d}.jpg'), y_enhanced)
                cv2.imwrite(os.path.join(self.debug_dir, f'deblurred_{frame_number:04d}.jpg'), result)

            return result

        except Exception as e:
            self._log(f"Deblurring error: {str(e)}", is_error=True)
            return frame

    def _colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            model_params = self.cfg['MODEL_PARAMS']['deoldify']
            input_size = model_params['input_size']

            original_bgr = frame.copy()
            original_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            original_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_channel = original_lab[:, :, 0].astype(np.float32)

            # Fix: Ensure tileGridSize is a tuple
            grid_size = self.cfg['COLOR_CORRECTION']['contrast']['clahe_grid_size']
            if isinstance(grid_size, int):
                grid_size = (grid_size, grid_size)

            clahe = cv2.createCLAHE(
                clipLimit=self.cfg['COLOR_CORRECTION']['contrast']['clahe_clip_limit'],
                tileGridSize=grid_size
            )

            l_normalized = clahe.apply((l_channel * 255).astype(np.uint8)).astype(np.float32) / 255.0

            resized_l = cv2.resize(l_normalized, input_size, interpolation=cv2.INTER_CUBIC)
            l_3channel = np.repeat(resized_l[:, :, np.newaxis], 3, axis=2)
            input_tensor = np.expand_dims(l_3channel.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

            output = self.sessions['colorizer'].run(
                None,
                {self.sessions['colorizer'].get_inputs()[0].name: input_tensor}
            )[0]

            ab_channels = output.squeeze().transpose(1, 2, 0)[:, :, :2]
            ab_channels = (ab_channels * 255).astype(np.uint8)
            
            # Resize ab_channels to match l_normalized dimensions
            ab_channels = cv2.resize(ab_channels, (frame.shape[1], frame.shape[0]))

            # Apply guided filter with size-matched images
            ab_channels = cv2.ximgproc.guidedFilter(
                guide=(l_normalized * 255).astype(np.uint8),
                src=ab_channels,
                radius=10,
                eps=100
            )

            lab_result = original_lab.copy()
            lab_result[:, :, 1:] = ab_channels
            colorized_bgr = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

            saturation_mask = cv2.normalize(
                original_hsv[:, :, 1].astype(np.float32),
                None, 0, 1, cv2.NORM_MINMAX
            )

            blended = cv2.addWeighted(
                colorized_bgr,
                0.7 - (0.3 * saturation_mask),
                original_bgr,
                0.3 + (0.3 * saturation_mask),
                0
            )

            blended_ycrcb = cv2.cvtColor(blended, cv2.COLOR_BGR2YCrCb)
            original_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
            blended_ycrcb[:, :, 1:] = original_ycrcb[:, :, 1:]

            final = cv2.cvtColor(blended_ycrcb, cv2.COLOR_YCrCb2BGR)

            final_hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
            final_hsv[:, :, 1] = np.clip(
                final_hsv[:, :, 1] * model_params['saturation_factor'],
                0, 255
            )
            final = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            return final

        except Exception as e:
            self._log(f"Colorization error: {str(e)}", is_error=True)
            return frame

    def _super_resolution(self, frame: np.ndarray) -> np.ndarray:
        try:
            model = self.sessions['sr']
            input_shape = model.get_inputs()[0].shape
            h, w = input_shape[2], input_shape[3]

            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
            input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

            output = model.run(None, {model.get_inputs()[0].name: input_tensor})[0]

            output = output.squeeze().transpose(1, 2, 0)
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        except Exception as e:
            self._log(f"Super resolution error: {str(e)}", is_error=True)
            return frame

    def _multi_step_upscale(self, frame: np.ndarray) -> np.ndarray:
        try:
            original_dtype = frame.dtype
            original_size = frame.shape[:2]

            if min(original_size) < 128:
                frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LANCZOS4)

            frame = self._super_resolution(frame)

            if self._4k_enabled:
                frame = cv2.resize(frame,
                                 self.cfg['OUTPUT']['4k_resolution'],
                                 interpolation=cv2.INTER_LANCZOS4)

            return frame.astype(original_dtype) if frame.dtype != original_dtype else frame

        except Exception as e:
            self._log(f"Upscaling error: {str(e)}", is_error=True)
            return frame

    def _enhance_faces(self, frame: np.ndarray) -> np.ndarray:
        if not self.cfg['PROCESSING_STEPS']['face_enhancement']:
            return frame

        try:
            (h, w) = frame.shape[:2]
            faces = self._detect_faces(frame, h, w)

            for (x, y, x2, y2) in faces:
                w_face, h_face = x2 - x, y2 - y

                y_exp = max(0, y - int(h_face * 0.25))
                x_exp = max(0, x - int(w_face * 0.25))
                h_exp = min(h - y_exp, int(h_face * 1.5))
                w_exp = min(w - x_exp, int(w_face * 1.5))

                if w_exp <= 0 or h_exp <= 0:
                    continue

                face_roi = frame[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp]
                enhanced_face = self._process_face_region(face_roi, w_exp, h_exp)
                frame[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp] = enhanced_face

        except Exception as e:
            self._log(f"Face enhancement error: {str(e)}", is_error=True)

        return frame

    def _detect_faces(self, frame: np.ndarray, h: int, w: int) -> list:
        faces = []

        if hasattr(self, 'face_detector'):
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
                swapRB=False,
                crop=False
            )
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.7:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))
        elif hasattr(self, 'face_cascade'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            faces = [(x, y, x + w, y + h) for (x, y, w, h) in face_rects]

        return faces

    def _process_face_region(self, face_roi: np.ndarray, w_exp: int, h_exp: int) -> np.ndarray:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        eye_mask = np.zeros_like(gray_face)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(eye_mask, (ex, ey), (ex + ew, ey + eh), 255, -1)

        input_face = cv2.resize(face_roi, (512, 512))
        input_tensor = np.expand_dims(input_face.transpose(2, 0, 1), 0)
        input_tensor = input_tensor.astype(np.float32) / 255.0

        enhanced_face = self.sessions['face'].run(None, {'input': input_tensor})[0]
        enhanced_face = enhanced_face.squeeze().transpose(1, 2, 0)
        enhanced_face = np.clip(enhanced_face * 255, 0, 255).astype(np.uint8)
        enhanced_face = cv2.resize(enhanced_face, (w_exp, h_exp))

        eye_mask = cv2.resize(eye_mask, (w_exp, h_exp))
        enhanced_face_with_eyes = cv2.bitwise_and(enhanced_face, enhanced_face, mask=255 - eye_mask)
        original_face_with_eyes = cv2.bitwise_and(face_roi, face_roi, mask=eye_mask)

        return cv2.add(enhanced_face_with_eyes, original_face_with_eyes)

    def _apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        try:
            # White Balance
            if self.cfg['COLOR_CORRECTION']['white_balance']['enabled']:
                try:
                    if hasattr(cv2, 'xphoto'):
                        wb = cv2.xphoto.createGrayworldWB()
                        wb.setSaturationThreshold(
                            self.cfg['COLOR_CORRECTION']['white_balance']['saturation_threshold']
                        )
                        corrected = wb.balanceWhite(frame)
                    else:
                        # Fallback simple white balance
                        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        avg_a = np.average(result[:, :, 1])
                        avg_b = np.average(result[:, :, 2])
                        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
                        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
                        corrected = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
                except Exception as e:
                    self._log(f"White balance error, using original: {str(e)}")
                    corrected = frame.copy()
            else:
                corrected = frame.copy()

            # Rest of the color correction code remains the same...
            if self.cfg['COLOR_CORRECTION']['color_grading']['enabled']:
                corrected = self._apply_film_lut(corrected)

            # Contrast enhancement
            lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            l_contrast = cv2.normalize(
                l_channel * 1.2 - 30, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            lab[:, :, 0] = cv2.addWeighted(l_contrast, 0.7, lab[:, :, 0], 0.3, 0)
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            return corrected

        except Exception as e:
            self._log(f"Color correction error: {str(e)}", is_error=True)
            return frame

    def _apply_film_lut(self, frame: np.ndarray) -> np.ndarray:
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip(i * 1.1 - 20, 0, 255))
            g = int(np.clip(i * 1.05 - 15, 0, 255))
            b = int(np.clip(i * 0.95, 0, 255))

            if i > 192:
                r = int(r * self.cfg['COLOR_CORRECTION']['color_grading']['highlights_rolloff'])
                g = int(g * self.cfg['COLOR_CORRECTION']['color_grading']['highlights_rolloff'])

            if 64 < i < 192:
                r = int(r * self.cfg['COLOR_CORRECTION']['color_grading']['midtone_shift']['red'])
                g = int(g * self.cfg['COLOR_CORRECTION']['color_grading']['midtone_shift']['green'])
                b = int(b * self.cfg['COLOR_CORRECTION']['color_grading']['midtone_shift']['blue'])

            lut[i, 0, :] = [b, g, r]

        graded = cv2.LUT(frame, lut)
        return cv2.addWeighted(
            graded,
            self.cfg['COLOR_CORRECTION']['color_grading']['lut_strength'],
            frame,
            1 - self.cfg['COLOR_CORRECTION']['color_grading']['lut_strength'],
            0
        )

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        if frame is None or frame.size == 0:
            self._log(f"Invalid frame {frame_number}", is_error=True)
            return np.zeros((100, 100, 3), dtype=np.uint8)  # Return blank frame

        try:
            if self.cfg['DEBUG']['save_intermediate']:
                cv2.imwrite(os.path.join(self.debug_dir, f'input_{frame_number:04d}.jpg'), frame)

            processed = frame.copy().astype(np.float32) / 255.0

            # 1. Enhanced motion deblur
            if self.cfg['PROCESSING_STEPS']['deblurring']:
                processed = self._motion_deblur((processed * 255).astype(np.uint8), frame_number).astype(np.float32) / 255.0
                if self.cfg['DEBUG']['save_intermediate']:
                    cv2.imwrite(os.path.join(self.debug_dir, f'post_deblur_{frame_number:04d}.jpg'), (processed * 255).astype(np.uint8))

            # 2. Super resolution
            if self.cfg['PROCESSING_STEPS']['super_resolution'] and self._4k_enabled:
                processed = self._multi_step_upscale((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0
                if self.cfg['DEBUG']['save_intermediate']:
                    cv2.imwrite(os.path.join(self.debug_dir, f'post_upscale_{frame_number:04d}.jpg'), (processed * 255).astype(np.uint8))

            # 3. Color preservation processing
            original_colors = processed.copy()

            # 4. Colorization with adaptive blending
            if self.cfg['PROCESSING_STEPS']['colorization']:
                colorized = self._colorize_frame((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0
                processed = cv2.addWeighted(colorized, 0.7, original_colors, 0.3, 0)
                if self.cfg['DEBUG']['save_intermediate']:
                    cv2.imwrite(os.path.join(self.debug_dir, f'post_color_{frame_number:04d}.jpg'), (processed * 255).astype(np.uint8))

            # 5. Face enhancement
            if self.cfg['PROCESSING_STEPS']['face_enhancement']:
                processed = self._enhance_faces((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0
                if self.cfg['DEBUG']['save_intermediate']:
                    cv2.imwrite(os.path.join(self.debug_dir, f'post_face_{frame_number:04d}.jpg'), (processed * 255).astype(np.uint8))

            # 6. Advanced color correction
            if self.cfg['PROCESSING_STEPS']['color_correction']:
                processed = self._apply_color_correction((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0

            # 7. Conservative sharpening
            if self._4k_enabled:
                processed = cv2.detailEnhance((processed * 255).astype(np.uint8), sigma_s=1.5, sigma_r=0.05).astype(np.float32) / 255.0
            else:
                kernel = np.array([[0, -0.25, 0],
                                 [-0.25, 2, -0.25],
                                 [0, -0.25, 0]])
                processed = cv2.filter2D((processed * 255).astype(np.uint8), -1, kernel).astype(np.float32) / 255.0

            # 8. Temporal filtering
            if self.cfg['PROCESSING_STEPS']['temporal_filtering']:
                processed = self.temporal_filter.stabilize((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0

            processed = np.clip(processed * 255, 0, 255).astype(np.uint8)

            if self.cfg['DEBUG']['save_intermediate']:
                cv2.imwrite(os.path.join(self.debug_dir, f'output_{frame_number:04d}.jpg'), processed)

            return processed

        except Exception as e:
            self._log(f"Frame processing error: {str(e)}", is_error=True)
            return (frame * 255).astype(np.uint8)

    def doEnhance(self) -> bool:
        try:
            temp_path = os.path.join(self.cfg['INIT_PATH'], 'Temp')
            enhanced_path = os.path.join(self.cfg['INIT_PATH'],
                                       'Enhanced_4K' if self._4k_enabled else 'Enhanced')
            os.makedirs(enhanced_path, exist_ok=True)

            frame_files = sorted(glob.glob(os.path.join(temp_path, '*.jpg')))
            total_frames = len(frame_files)

            if total_frames == 0:
                self._log("No frames found for enhancement", is_error=True)
                return False

            self._log(f"Starting enhancement of {total_frames} frames")

            for frame_number, frame_file in enumerate(frame_files, 1):
                try:
                    frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"Corrupt or unsupported frame: {frame_file}")

                    enhanced = self._process_frame(frame, frame_number)
                    output_path = os.path.join(enhanced_path, os.path.basename(frame_file))
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save with compression (adjust quality if needed)
                    cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    if not self.cfg['DEBUG']['keep_temp_files']:
                        os.remove(frame_file)

                    # Release memory explicitly
                    del frame, enhanced
                    if frame_number % 50 == 0:
                        gc.collect()  # Garbage collect periodically

                except Exception as e:
                    self._log(f"Error processing frame {frame_file}: {str(e)}", is_error=True)
                    continue

                if frame_number % 10 == 0 and self.progress_callback:
                    progress = int((frame_number / total_frames) * 100)
                    self.progress_callback(progress, f"Processed {frame_number}/{total_frames} frames")

            self._log("Frame enhancement completed successfully")
            return True

        except Exception as e:
            self._log(f"Enhancement pipeline failed: {str(e)}", is_error=True)
            return False