# Facial Feature Extractor

A comprehensive Python library for detecting faces, extracting facial landmarks, performing image enhancements, and calculating a rich set of over 150 geometric facial features from images. This tool is built using Dlib and OpenCV.

## Key Features

* **Face Detection**: Utilizes Dlib's HOG-based frontal face detector.
* **Landmark Prediction**: Extracts 68-point facial landmarks.
* **Face Alignment**: Generates aligned facial "chips" for normalized views.
* **Image Enhancement**: A pipeline of optional image pre-processing steps including:
    * Illumination Normalization (CLAHE)
    * Grayscale Conversion
    * Denoising (Gaussian & Bilateral Filters)
    * Image Sharpening
    * Self-Quotient Image (SQI) for lighting invariance
* **Rich Feature Calculation**: Computes over 150 geometric features, including:
    * Distances, angles, and ratios (e.g., Eye Aspect Ratio, Nose-to-Face Ratio).
    * Symmetry and proportion metrics (e.g., "Three Courts and Five Eyes").
    * Facial contour and area measurements.
    * Emotion-related geometric cues (for tension, anger, smiling).

## Project Structure

```
.
├── facial_feature_extractor/   # Main library package
├── scripts/                    # Example scripts for usage
├── models/                     # Directory for Dlib model file
├── data/                       # Directory for sample data
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set up a Python environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    This project requires `dlib`, which can be tricky to install. It's often easier to install it before the other packages.

    ```bash
    pip install dlib
    pip install -r requirements.txt
    ```
    *Note: If you have issues installing dlib, you may need to install `cmake` and a C++ compiler first.*

4.  **Download the Dlib Model:**
    Download the shape predictor model file from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    * Unzip the file to get `shape_predictor_68_face_landmarks.dat`.
    * Place this file inside the `models/` directory.

## Usage

The core functionality is wrapped in the `FaceAnalyzer` class.

### Basic Example: Processing a Single Image

```python
from facial_feature_extractor.analysis import FaceAnalyzer
from facial_feature_extractor.utils import save_image, draw_landmarks_on_image
import pprint

# 1. Initialize the analyzer with the path to the dlib model
analyzer = FaceAnalyzer(shape_predictor_path='models/shape_predictor_68_face_landmarks.dat')

# 2. Process an image
image_path = 'path/to/your/image.jpg'
result = analyzer.process_image(image_path)

# 3. Print the results
print(f"Processing Status: {result['status']}")
if result['status'] == 'success':
    print(f"Detected Face Area: {result['face_area']}")
    
    # Print a few of the calculated features
    print("\nSample of Calculated Features:")
    pprint.pprint({k: v for k, v in result['features'].items() if 'eye_aspect_ratio' in k or 'mouth' in k})

    # Save the final processed image with landmarks drawn
    if result['final_image'] is not None:
        final_img_with_landmarks = draw_landmarks_on_image(result['final_image'], result['landmarks'])
        save_image(final_img_with_landmarks, 'output_image_with_landmarks.png')
        print("\nSaved final image with landmarks to 'output_image_with_landmarks.png'")

```

### Batch Processing

The `scripts/2_run_batch_processing.py` script provides a powerful example of how to process a directory of images and save all results to a single CSV file.

**To run the batch script:**

1.  Place your images in a directory (e.g., `data/input_images/`).
2.  Modify the configuration variables at the top of the script:
    ```python
    # scripts/2_run_batch_processing.py
    IMAGE_DIRECTORY = "data/input_images"
    OUTPUT_CSV_PATH = "data/facial_features_output.csv"
    SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
    ```
3.  Execute the script:
    ```bash
    python scripts/2_run_batch_processing.py
    ```
    This will generate a CSV file containing all extracted features and processing metadata for each image.

### Downloading Images

The `scripts/1_download_images.py` script shows how to download images from a list of URLs in a source CSV file. This can be used as a preliminary step before batch processing.

### Customization

You can customize the processing pipeline by passing a configuration dictionary to the `FaceAnalyzer`. See the `analysis.py` file and the batch processing script for a full list of configurable parameters.

```python
from facial_feature_extractor.analysis import FaceAnalyzer

# Example of custom configuration
custom_config = {
    'align_target_size': 128,
    'align_padding': 0.1,
    'apply_grayscale': False,
    'apply_illumination_norm': True,
    'apply_bilateral_filter': False,
    # etc.
}

analyzer = FaceAnalyzer(
    shape_predictor_path='models/shape_predictor_68_face_landmarks.dat',
    config=custom_config
)
```

## Feature List

This library calculates over 150 geometric features. For a complete list and their descriptions, please see the `ALL_FEATURE_NAMES` list within the `facial_feature_extractor/features.py` file.

## Downstream Example

The extracted features can be used for a variety of machine learning tasks, such as emotion recognition, biometric analysis, or credit risk modeling. The `functionF.py` script (not included in the core library) from the original project demonstrates how these features can be fed into an XGBoost model with hyperparameter tuning for a classification task.

## License

This project is released under the MIT License.