# facial_feature_extractor/utils.py
import cv2
import os
import numpy as np
import json
import math

# Dlib 68-point landmark indices
JAWLINE = list(range(0, 17))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
NOSE_BRIDGE = list(range(27, 31))
NOSE_TIP_LOWER = list(range(31, 36))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
OUTER_LIP = list(range(48, 60))
INNER_LIP = list(range(60, 68))


def create_dir_if_not_exists(directory: str):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create directory {directory}: {e}")
            raise


def load_image(image_path: str) -> np.ndarray | None:
    """Loads an image from a file path using OpenCV."""
    if not image_path or not isinstance(image_path, str) or not os.path.exists(image_path):
        return None
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, path: str) -> bool:
    """Saves an image to a specified path using OpenCV."""
    if image is None or not isinstance(image, np.ndarray):
        return False
    try:
        directory = os.path.dirname(path)
        if directory:
            create_dir_if_not_exists(directory)
        ext = os.path.splitext(path)[1] or ".png"
        is_success, buffer = cv2.imencode(ext, image)
        if is_success:
            with open(path, 'wb') as f:
                f.write(buffer)
            return True
        return False
    except Exception as e:
        print(f"Error saving image to {path}: {e}")
        return False


def calculate_distance(p1: tuple, p2: tuple) -> float:
    """Calculates the Euclidean distance between two points."""
    try:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    except (TypeError, IndexError):
        return np.nan


def parse_landmarks(landmark_data: str | list | None) -> list[tuple[int, int]] | None:
    """Parses landmarks from a JSON string or a list into a list of (x, y) tuples."""
    if landmark_data is None:
        return None
    try:
        if isinstance(landmark_data, str):
            landmarks_list = json.loads(landmark_data)
        elif isinstance(landmark_data, list):
            landmarks_list = landmark_data
        else:
            return None

        if not isinstance(landmarks_list, list): return None

        parsed_landmarks = []
        for p in landmarks_list:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                parsed_landmarks.append((int(round(float(p[0]))), int(round(float(p[1])))))
            else:
                return None  # Invalid point format

        return parsed_landmarks if len(parsed_landmarks) == 68 else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def draw_landmarks_on_image(image: np.ndarray, landmarks: list, color=(0, 255, 0), radius=2) -> np.ndarray:
    """Draws landmark points on an image."""
    img_copy = image.copy()
    if len(img_copy.shape) == 2:  # If grayscale, convert to BGR to draw in color
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    parsed_landmarks = parse_landmarks(landmarks)
    if parsed_landmarks:
        for (x, y) in parsed_landmarks:
            cv2.circle(img_copy, (x, y), radius, color, -1)
    return img_copy


def crop_and_resize(image, target_size=None, crop_box=None):
    """Crops and/or resizes an image using OpenCV."""
    processed_image = image.copy()
    if crop_box:
        try:
            x, y, w, h = map(int, crop_box)
            ih, iw = processed_image.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            if x2 > x1 and y2 > y1:
                processed_image = processed_image[y1:y2, x1:x2]
        except Exception as e:
            print(f"Error during cropping: {e}")
            return image

    if target_size:
        try:
            target_width, target_height = map(int, target_size)
            h, w = processed_image.shape[:2]
            if w != target_width or h != target_height:
                interpolation = cv2.INTER_AREA if target_width < w else cv2.INTER_LINEAR
                processed_image = cv2.resize(processed_image, (target_width, target_height),
                                             interpolation=interpolation)
        except Exception as e:
            print(f"Error during resizing: {e}")
            return processed_image  # Return the unresized (but possibly cropped) image

    return processed_image