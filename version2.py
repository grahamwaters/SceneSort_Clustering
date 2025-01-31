# travel_scene_sorter_dynamic.py

import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip  # OpenAI's CLIP model

from sklearn.cluster import DBSCAN
print('loaded libraries')  # ! debugging to show progress.
# -------------
# Configuration
# -------------
INPUT_FOLDER = "/Volumes/BigBoy/portugal trip 2025"  # Folder containing unclassified photos and videos
OUTPUT_FOLDER = "/Volumes/BigBoy/sorted_media"  # Output directory where files are sorted by scene

# DBSCAN parameters (adjust these for your data)
EPS = 0.14  # Maximum distance between samples for them to be considered in the same neighborhood
MIN_SAMPLES = 2  # Minimum number of samples required in a neighborhood to form a cluster

VIDEO_FRAME_INTERVAL = 60  # seconds between key frames for video processing

# Allowed file extensions for images and videos
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dng"}  # note - dng not included in initial code, may cause issues.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# -------------
# Setup device and load the CLIP model
# -------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -------------
# Functions to process media files
# -------------
# ... (get_image_embedding, get_video_embedding, extract_embedding remain the same)
from main import get_image_embedding, get_video_embedding, extract_embedding #note: these are imported from the other file main.py
# -------------
# Main processing
# -------------
def main():
    # Gather all media files from the input folder and subfolders
    file_paths = []
    for root, _, files in tqdm(os.walk(INPUT_FOLDER)):
            for file in files:
                if file.startswith("._"):  #? Skip hidden files
                    continue  # Go to the next file in the loop
                # print(f'ext: {os.path.splitext(file)}')
                ext = os.path.splitext(file)[1]
                ext = ext.lower() #? does this fix the error?
                if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                    file_paths.append(os.path.join(root, file))

    if not file_paths:
        print("No media files found in the input folder or subfolders.")
        return

    print(f"Found {len(file_paths)} media files. Extracting embeddings...")

    embeddings = []
    valid_file_paths = []  # only files with a valid embedding

    # Process each file and compute its embedding
    for file_path in tqdm(file_paths):
        emb = extract_embedding(file_path)
        if emb is not None:
            embeddings.append(emb)
            valid_file_paths.append(file_path)

    if not embeddings:
        print("No embeddings could be computed. Exiting.")
        return

    embeddings = np.array(embeddings)

    # -------------
    # Clustering with DBSCAN (number of clusters is determined automatically)
    # -------------
    print("Clustering media files by scene using DBSCAN...")
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine')
    clusters = dbscan.fit_predict(embeddings)
    unique_labels = set(clusters)

    print(f"DBSCAN found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters "
          f"and labeled {list(clusters).count(-1)} files as noise.")

    # -------------
    # Create output directories and move/copy files accordingly
    # -------------
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Create directories for each cluster. Outliers (label == -1) go in a 'noise' folder.
    for label in unique_labels:
        if label == -1:
            cluster_dir = os.path.join(OUTPUT_FOLDER, "noise")
        else:
            cluster_dir = os.path.join(OUTPUT_FOLDER, f"scene_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    for file_path, label in zip(valid_file_paths, clusters):
        if label == -1:
            dest_dir = os.path.join(OUTPUT_FOLDER, "noise")
        else:
            dest_dir = os.path.join(OUTPUT_FOLDER, f"scene_{label}")

        # Maintain the original subfolder structure within the output
        relative_path = os.path.relpath(file_path, INPUT_FOLDER) # get the relative path to the file
        dest_path = os.path.join(dest_dir, relative_path) # create the full destination path, including the relative path


        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Create subfolders if necessary

        # Copy the file (change to shutil.move to move the files instead of copying)
        shutil.copy2(file_path, dest_path)

    print("Media files have been sorted by detected scenes:")
    for label in unique_labels:
        if label == -1:
            cluster_dir = os.path.join(OUTPUT_FOLDER, "noise")
            label_name = "noise"
        else:
            cluster_dir = os.path.join(OUTPUT_FOLDER, f"scene_{label}")
            label_name = f"scene_{label}"
        count = len(os.listdir(cluster_dir))
        print(f" - {label_name}: {count} files")


if __name__ == "__main__":
    main()