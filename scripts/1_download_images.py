# scripts/1_download_images.py
import pandas as pd
import requests
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import mimetypes

# --- Configuration ---
# Input CSV containing URLs
INPUT_CSV_PATH = 'data/source_urls.csv'
# Column in the CSV that contains the image URLs
URL_COLUMN = 'validate_photo_url'
# Column to use for creating filenames (e.g., a unique ID)
ID_COLUMN = 'order_no'
# Output directory to save images
IMAGE_SAVE_DIR = 'downloaded_images'
# Output CSV with local file paths
OUTPUT_CSV_PATH = 'data/downloaded_image_paths.csv'

MAX_WORKERS = 10
REQUEST_TIMEOUT = 15
DEFAULT_EXTENSION = '.jpg'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_file_extension(content_type: str | None) -> str:
    """Guesses file extension from MIME type."""
    if not content_type:
        return DEFAULT_EXTENSION
    return mimetypes.guess_extension(content_type.split(';')[0].strip()) or DEFAULT_EXTENSION


def download_and_save(url: str, id_val: str, index: int) -> dict:
    """Downloads a single image and saves it to disk."""
    result = {'index': index, 'image_filepath': None, 'error': None}
    if not isinstance(url, str) or not url.startswith('http'):
        result['error'] = f'Invalid URL format: {url}'
        return result

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        ext = get_file_extension(response.headers.get('Content-Type'))
        filename = f"{str(id_val).strip()}{ext}"
        filepath = os.path.join(IMAGE_SAVE_DIR, filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)
        result['image_filepath'] = filepath

    except requests.RequestException as e:
        result['error'] = str(e)
    except IOError as e:
        result['error'] = f"File save error: {e}"

    return result


def main():
    """Main function to download images based on a CSV file."""
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        logging.info(f"Loaded {len(df)} rows from {INPUT_CSV_PATH}")
    except FileNotFoundError:
        logging.error(f"Input CSV not found: {INPUT_CSV_PATH}")
        return

    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    logging.info(f"Images will be saved to '{IMAGE_SAVE_DIR}'")

    tasks = []
    for index, row in df.iterrows():
        url = row.get(URL_COLUMN)
        id_val = row.get(ID_COLUMN)
        if pd.notna(url) and pd.notna(id_val):
            tasks.append((url, id_val, index))

    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_and_save, url, id_val, index): index for url, id_val, index in tasks}

        for future in tqdm(futures, total=len(tasks), desc="Downloading Images"):
            res = future.result()
            results[res['index']] = res

    # Add results back to the DataFrame
    df['image_filepath'] = [r['image_filepath'] if r else None for r in results]
    df['download_error'] = [r['error'] if r else 'No URL/ID' for r in results]

    # Save the new CSV with local paths
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    logging.info(f"Saved new CSV with local paths to {OUTPUT_CSV_PATH}")

    success_count = df['image_filepath'].notna().sum()
    logging.info(f"Successfully downloaded {success_count} / {len(tasks)} images.")


if __name__ == "__main__":
    main()