# facial_feature_extractor/landmarks.py
import dlib
import numpy as np
import cv2

def get_landmarks(image: np.ndarray, face_rect: dlib.rectangle, predictor: dlib.shape_predictor):
    """
    Finds facial landmarks for a given face rectangle.

    Args:
        image (np.ndarray): Input BGR or grayscale image.
        face_rect (dlib.rectangle): The rectangle defining the face region.
        predictor (dlib.shape_predictor): An initialized dlib shape predictor model.

    Returns:
        dlib.full_object_detection | None: The detected landmarks object, or None on failure.
    """
    if image is None or face_rect is None or predictor is None:
        return None

    try:
        if len(image.shape) > 2:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        landmarks = predictor(img_gray, face_rect)
        return landmarks if landmarks.num_parts > 0 else None
    except Exception as e:
        print(f"Error during dlib landmark prediction: {e}")
        return None

def align_face_chip(image: np.ndarray, landmarks: dlib.full_object_detection,
                     target_size: int = 256, padding: float = 0.2):
    """
    Extracts an aligned face chip using dlib's get_face_chip.

    Args:
        image (np.ndarray): The original BGR input image.
        landmarks (dlib.full_object_detection): The detected landmarks object.
        target_size (int): The desired output size (width and height) of the chip.
        padding (float): Padding around the face, relative to face size.

    Returns:
        np.ndarray | None: The aligned face chip in BGR format, or None on failure.
    """
    if image is None or landmarks is None:
        return None

    try:
        return dlib.get_face_chip(image, landmarks, size=target_size, padding=padding)
    except Exception as e:
        print(f"Error during dlib.get_face_chip: {e}")
        return None

def extract_landmark_points(landmarks: dlib.full_object_detection) -> list[tuple[int, int]] | None:
    """Extracts landmark coordinates as a list of (x, y) tuples."""
    if landmarks is None or landmarks.num_parts == 0:
        return None
    return [(p.x, p.y) for p in landmarks.parts()]