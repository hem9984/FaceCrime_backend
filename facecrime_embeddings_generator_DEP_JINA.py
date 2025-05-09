#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kagglehub
from pathlib import Path
from PIL import Image
import base64
import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModel
import csv
import json
import pandas as pd

def main():
    # ---------------------------------------------------------------------
    # Step 1: Download dataset
    # ---------------------------------------------------------------------
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("elliotp/idoc-mugshots")
    print("Path to dataset files:", path)
    folder_path = Path(path)

    # ---------------------------------------------------------------------
    # Step 2: Load embedding model and processor
    # ---------------------------------------------------------------------
    print("Loading processor and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True).to(device)
    model.eval()

    # ---------------------------------------------------------------------
    # Step 3: Define helper functions
    # ---------------------------------------------------------------------
    # 3a. Image to base64 encoder
    def load_image_base64(file_path):
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"[Base64 Error] {file_path.name}: {e}")
            return None

    # 3b. Get CLIP embeddings (image only) from the model
    def get_embeddings(file_path):
        try:
            # Load image via Pillow
            image = Image.open(file_path).convert("RGB")

            # Let the CLIP processor handle any required resizing, normalization, etc.
            inputs = processor(images=image, return_tensors="pt")

            # Move tensor to the GPU/CPU device
            pixel_values = inputs['pixel_values'].to(device)

            with torch.no_grad():
                # For Jina CLIP-based models, we directly call get_image_features to get image embeddings
                embeddings = model.get_image_features(pixel_values=pixel_values)

            # Flatten & convert to CPU numpy
            embeddings = embeddings.squeeze().cpu().numpy()
            return embeddings.tolist()

        except Exception as e:
            print(f"[Embedding Error] {file_path.name}: {e}")
            return None

    # ---------------------------------------------------------------------
    # Step 4: Generate embeddings CSV
    # ---------------------------------------------------------------------
    embeddings_csv_file = "embeddings.csv"
    fieldnames = ["filename", "image_base64", "embedding"]

    print("Generating embeddings and writing to embeddings.csv...")
    with open(embeddings_csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through all files in the dataset directory
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                # Attempt to open as an image; if it fails, skip
                try:
                    Image.open(file_path).convert("RGB")
                except:
                    continue

                image_base64 = load_image_base64(file_path)
                embeddings = get_embeddings(file_path)

                if image_base64 and embeddings:
                    writer.writerow({
                        "filename": file_path.name,
                        "image_base64": image_base64,
                        "embedding": json.dumps(embeddings)
                    })

    print(f"✅ CSV file created: {embeddings_csv_file}")

    # ---------------------------------------------------------------------
    # Step 5: Merge embeddings with labels CSV
    # ---------------------------------------------------------------------
    labels_csv_file = "labels_utf8.csv"     # Adjust path if needed
    merged_csv_file = "merged_embeddings.csv"

    print("Merging embeddings.csv with labels_utf8.csv on matching ID/filename...")

    # Load the embeddings CSV into a pandas DataFrame
    embeddings_df = pd.read_csv(embeddings_csv_file)

    # Load the labels CSV into a pandas DataFrame
    labels_df = pd.read_csv(labels_csv_file)

    # Merge on 'filename' in embeddings and 'ID' in labels
    merged_df = pd.merge(
        embeddings_df,
        labels_df,
        how="left",
        left_on="filename",
        right_on="ID"
    )

    # Write the merged data to a new CSV file
    merged_df.to_csv(merged_csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"✅ Merged CSV file created: {merged_csv_file}")


if __name__ == "__main__":
    main()

