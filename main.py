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
print('loaded libraries') #! debugging to show progress.
# -------------
# Configuration
# -------------
INPUT_FOLDER = "/Users/grahamwaters/Downloads/TripFootage"   # Folder containing unclassified photos and videos
OUTPUT_FOLDER = "/Users/grahamwaters/Downloads/TripFootage/sorted_scenes"  # Output directory where files are sorted by scene

# DBSCAN parameters (adjust these for your data)
EPS = 0.14           # Maximum distance between samples for them to be considered in the same neighborhood
MIN_SAMPLES = 2     # Minimum number of samples required in a neighborhood to form a cluster

VIDEO_FRAME_INTERVAL = 60  # seconds between key frames for video processing

# Allowed file extensions for images and videos
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",".dng"} #note - dng not included in initial code, may cause issues.
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
def get_image_embedding(image_path):
    """Load an image file, preprocess it, and compute its CLIP embedding."""
    print(f'Processing Image: {image_path}')
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        # Normalize the embedding vector
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_video_embedding(video_path):
    """
    Process a video file: sample key frames every VIDEO_FRAME_INTERVAL seconds,
    compute embeddings for each frame, and return the average embedding.
    """
    print(f'Processing Video: {video_path}')
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # assume a default FPS if unavailable

        frames_embeddings = []
        frame_interval = int(VIDEO_FRAME_INTERVAL * fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while current_frame < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR to RGB and create a PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            frames_embeddings.append(embedding.cpu().numpy()[0])
            current_frame += frame_interval

        cap.release()
        if frames_embeddings:
            avg_embedding = np.mean(frames_embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            return avg_embedding
        else:
            return None
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def extract_embedding(file_path):
    """Determine file type and return its corresponding embedding."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return get_image_embedding(file_path)
    elif ext in VIDEO_EXTENSIONS:
        return get_video_embedding(file_path)
    else:
        return None

# -------------
# Main processing
# -------------
def main():
    # Gather all media files from the input folder
    file_paths = []
    for root, _, files in tqdm(os.walk(INPUT_FOLDER)):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        print("No media files found in the input folder.")
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
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
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
