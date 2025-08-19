# facial_feature_extractor/analysis.py
import os
import dlib
import logging
from .utils import load_image
from .detection import detect_faces, get_best_face
from .landmarks import get_landmarks, align_face_chip, extract_landmark_points
from .enhancement import apply_optional_enhancements
from .features import calculate_all_features, ALL_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FaceAnalyzer:
    """
    A class to orchestrate the full pipeline of face analysis:
    detection, landmarking, alignment, enhancement, and feature extraction.
    """

    DEFAULT_CONFIG = {
        'detect_upsample': 1,
        'align_target_size': 256,
        'align_padding': 0.25,
        'apply_illumination_norm': True,
        'apply_grayscale': True,
        'apply_bilateral_filter': True,
        'bilateral_d': 7,
        'bilateral_sigma_color': 50,
        'bilateral_sigma_space': 50,
    }

    def __init__(self, shape_predictor_path: str, config: dict = None):
        """
        Initializes the FaceAnalyzer.

        Args:
            shape_predictor_path (str): Path to the dlib shape predictor model file.
            config (dict, optional): A dictionary to override default processing settings.
        """
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor model not found at: {shape_predictor_path}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

    def process_image(self, image_path: str):
        """
        Runs the full analysis pipeline on a single image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            dict: A dictionary containing the results of the analysis, including status,
                  images, landmarks, and calculated features.
        """
        result = {
            'image_path': image_path,
            'status': 'failed',
            'error_message': None,
            'face_area': None,
            'landmarks': None,
            'aligned_image': None,
            'final_image': None,
            'features': {name: None for name in ALL_FEATURE_NAMES}
        }

        original_img = load_image(image_path)
        if original_img is None:
            result['error_message'] = 'Could not load image'
            return result

        # 1. Face Detection
        detected_faces = detect_faces(original_img, self.detector, self.config['detect_upsample'])
        if not detected_faces:
            result['status'] = 'no_face_detected'
            result['error_message'] = 'No face detected in the image'
            return result

        best_face_rect = get_best_face(detected_faces)
        result['face_area'] = int(best_face_rect.width() * best_face_rect.height())

        # 2. Landmark Prediction
        landmarks_obj = get_landmarks(original_img, best_face_rect, self.predictor)
        if landmarks_obj is None:
            result['status'] = 'landmark_error'
            result['error_message'] = 'Failed to detect facial landmarks'
            return result

        landmark_points = extract_landmark_points(landmarks_obj)
        result['landmarks'] = landmark_points

        # 3. Face Alignment
        aligned_chip = align_face_chip(
            original_img, landmarks_obj,
            target_size=self.config['align_target_size'],
            padding=self.config['align_padding']
        )
        if aligned_chip is None:
            result['status'] = 'alignment_error'
            result['error_message'] = 'Failed to align face chip'
            return result
        result['aligned_image'] = aligned_chip

        # 4. Optional Enhancements
        final_image = apply_optional_enhancements(aligned_chip, self.config)
        if final_image is None:
            result['status'] = 'enhancement_error'
            result['error_message'] = 'Failed during image enhancement steps'
            return result
        result['final_image'] = final_image

        # 5. Feature Calculation
        features = calculate_all_features(landmark_points)
        result['features'] = features

        result['status'] = 'success'
        return result