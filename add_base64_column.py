#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read a CSV with an 'imageUri' column, download each image,
resize to 640x640, convert to base64 (JPEG quality=100), and
write a new CSV with an extra 'imageBase64' column.
Also logs failures (URI + status code) in a separate failures CSV.

Usage:
    python add_base64_column.py --input input.csv --output facecrime_sex_offenders_with_640_base64.csv
"""
import argparse
import csv
import logging
import base64
from io import BytesIO

from PIL import Image
import requests
from requests.adapters import HTTPAdapter, Retry
import urllib3

# Disable SSL warnings and turn off cert verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create a session with retry logic
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    })
    # Disable SSL verification
    session.verify = False
    return session

session = create_session()

def process_csv(input_path: str, output_path: str, uri_col: str = "imageUri"):
    """
    Reads input CSV, downloads each image from the given URI column,
    resizes to 640x640, encodes to base64 (JPEG quality=100), and writes
    out a new CSV with an added 'imageBase64' column.
    Records failures to a separate CSV with URI and status code.
    """
    failures = []
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames) + ['imageBase64']
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(reader, start=1):
                uri = row.get(uri_col, "")
                image_b64 = ""
                if uri:
                    try:
                        resp = session.get(uri, timeout=30)
                        resp.raise_for_status()
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        img = img.resize((640, 640), Image.LANCZOS)
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG", quality=100)
                        image_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
                    except requests.exceptions.RequestException as e:
                        status_code = None
                        if hasattr(e, 'response') and e.response is not None:
                            status_code = e.response.status_code
                        else:
                            status_code = str(e)
                        logger.warning(f"Row {idx}: Failed to fetch or process '{uri}': {status_code}")
                        failures.append({'row': idx, uri_col: uri, 'status_code': status_code})
                    except Exception as e:
                        logger.warning(f"Row {idx}: Error processing image '{uri}': {e}")
                        failures.append({'row': idx, uri_col: uri, 'status_code': str(e)})
                else:
                    logger.warning(f"Row {idx}: No URI in column '{uri_col}'")
                    failures.append({'row': idx, uri_col: uri, 'status_code': 'no_uri'})

                row['imageBase64'] = image_b64
                writer.writerow(row)

                if idx % 100 == 0:
                    logger.info(f"Processed {idx} rows...")

    logger.info(f"Finished processing. Output saved to '{output_path}'")
    # Write failures CSV if any
    if failures:
        failure_file = output_path.replace('.csv', '_failures.csv')
        with open(failure_file, 'w', newline='', encoding='utf-8') as ffail:
            fail_writer = csv.DictWriter(ffail, fieldnames=['row', uri_col, 'status_code'])
            fail_writer.writeheader()
            fail_writer.writerows(failures)
        logger.info(f"Failures recorded to '{failure_file}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add base64-encoded 640x640 images to CSV rows, with retries and UA, skipping SSL verification.'
    )
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Path for output CSV file')
    parser.add_argument('--uri-col', default='imageUri', help='Name of the CSV column with image URIs')
    args = parser.parse_args()

    process_csv(args.input, args.output, args.uri_col)

