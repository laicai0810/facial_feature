# facial_feature_extractor/detection.py
import dlib
import numpy as np
import cv2


def detect_faces(image: np.ndarray, detector: dlib.fhog_object_detector, upsample=1):
    """
    Detects faces in an image using a dlib HOG face detector.

    Args:
        image (np.ndarray): Input BGR image.
        detector: An initialized dlib face detector.
        upsample (int): Number of times to upsample the image to detect smaller faces.

    Returns:
        list[dlib.rectangle]: A list of detected face rectangles.
    """
    if image is None or image.size == 0 or detector is None:
        return []

    try:
        if len(image.shape) > 2 and image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # Assume it's already grayscale
            img_gray = image

        return list(detector(img_gray, upsample))
    except Exception as e:
        print(f"Error during dlib face detection: {e}")
        return []


def get_best_face(detected_faces: list):
    """
    Selects the best face (the one with the largest area) from a list of dlib rectangles.

    Args:
        detected_faces (list[dlib.rectangle]): A list of detected face rectangles.

    Returns:
        dlib.rectangle | None: The largest face rectangle, or None if the list is empty.
    """
    if not detected_faces:
        return None

    try:
        return max(detected_faces, key=lambda rect: rect.width() * rect.height())
    except Exception as e:
        print(f"Error selecting best face: {e}")
        return None