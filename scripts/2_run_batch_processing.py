# scripts/2_run_batch_processing.py
import os
import pandas as pd
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from facial_feature_extractor.analysis import FaceAnalyzer
from facial_feature_extractor.features import ALL_FEATURE_NAMES

# --- Configuration ---
# Directory containing the images to process
IMAGE_DIRECTORY = "data/input_images"
# Path to save the final CSV with all features
OUTPUT_CSV_PATH = "data/facial_features_output.csv"
# Path to the dlib shape predictor model
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
# Number of parallel processes to use
MAX_WORKERS = max(1, os.cpu_count() - 1)

# --- Global Initializer for Multiprocessing ---
# This avoids re-loading the model in every single process call
analyzer = None


def initialize_worker(predictor_path):
    """Initializes the FaceAnalyzer in each worker process."""
    global analyzer
    logging.info(f"Initializing analyzer in process {os.getpid()}...")
    analyzer = FaceAnalyzer(shape_predictor_path=predictor_path)


def process_image_task(image_path):
    """A wrapper function to be called by each process."""
    if analyzer is None:
        raise RuntimeError("Analyzer is not initialized in this process.")
    return analyzer.process_image(image_path)


def main():
    """Main function to run batch processing on a directory of images."""
    if not os.path.exists(IMAGE_DIRECTORY):
        logging.error(f"Image directory not found: {IMAGE_DIRECTORY}")
        return

    image_paths = [os.path.join(IMAGE_DIRECTORY, f) for f in os.listdir(IMAGE_DIRECTORY) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths:
        logging.warning(f"No images found in '{IMAGE_DIRECTORY}'.")
        return

    logging.info(f"Found {len(image_paths)} images. Starting processing with {MAX_WORKERS} workers...")

    all_results = []

    # Using ProcessPoolExecutor to leverage multiple CPU cores
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=initialize_worker,
                             initargs=(SHAPE_PREDICTOR_PATH,)) as executor:

        futures = {executor.submit(process_image_task, path): path for path in image_paths}

        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Processing Images"):
            try:
                result = future.result()
                # Flatten the result for the DataFrame
                flat_result = {
                    'image_path': result['image_path'],
                    'status': result['status'],
                    'error_message': result['error_message'],
                    'face_area': result['face_area']
                }
                # Add all feature values
                flat_result.update(result['features'])
                all_results.append(flat_result)

            except Exception as e:
                image_path = futures[future]
                logging.error(f"Error processing {image_path}: {e}")
                all_results.append({'image_path': image_path, 'status': 'critical_error', 'error_message': str(e)})

    # Create a DataFrame from the results
    # Define column order for better readability
    columns = ['image_path', 'status', 'error_message', 'face_area'] + ALL_FEATURE_NAMES
    results_df = pd.DataFrame(all_results, columns=columns)

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    logging.info(f"\nProcessing complete. Results saved to {OUTPUT_CSV_PATH}")
    logging.info(f"Status summary:\n{results_df['status'].value_counts().to_string()}")


if __name__ == "__main__":
    # This is important for multiprocessing on some platforms (like Windows)
    main()