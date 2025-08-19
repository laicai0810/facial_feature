# facial_feature_extractor/enhancement.py
import cv2
import numpy as np


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """Applies illumination normalization using CLAHE."""
    try:
        if len(image.shape) == 3:  # Color image
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        elif len(image.shape) == 2:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        return image
    except Exception as e:
        print(f"Error in normalize_illumination: {e}")
        return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an image to grayscale if it is not already."""
    try:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    except Exception as e:
        print(f"Error in convert_to_grayscale: {e}")
        return image


def apply_bilateral_filter(image: np.ndarray, d: int, sigma_color: int, sigma_space: int) -> np.ndarray:
    """Applies a bilateral filter."""
    try:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    except Exception as e:
        print(f"Error in apply_bilateral_filter: {e}")
        return image


def apply_optional_enhancements(
        image: np.ndarray,
        config: dict
) -> np.ndarray:
    """
    Applies a sequence of optional image enhancement steps based on a config dictionary.

    Args:
        image (np.ndarray): The input image.
        config (dict): A dictionary specifying which enhancements to apply.

    Returns:
        np.ndarray: The processed image.
    """
    if image is None:
        return None

    processed_image = image.copy()

    if config.get('apply_bilateral_filter'):
        processed_image = apply_bilateral_filter(
            processed_image,
            d=config.get('bilateral_d', 7),
            sigma_color=config.get('bilateral_sigma_color', 50),
            sigma_space=config.get('bilateral_sigma_space', 50)
        )

    if config.get('apply_illumination_norm'):
        processed_image = normalize_illumination(processed_image)

    # Final grayscale conversion if requested
    if config.get('apply_grayscale'):
        processed_image = convert_to_grayscale(processed_image)

    return processed_image