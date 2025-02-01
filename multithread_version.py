#!/usr/bin/env python3
"""
Enhanced Media Scene Sorter with Hybrid Clustering
"""

import os
import sys
import argparse
import shutil
import logging
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import clip
import rawpy  # For RAW image support

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN  # Optional alternative clustering

from concurrent.futures import ThreadPoolExecutor, as_completed

# Define supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
RAW_EXTENSIONS   = {".raw", ".cr2", ".nef", ".dng", ".arw", ".rw2"}

# Global flag to control processing of large files.
# Default is False: large files are skipped.
process_large_files = False  # This will be updated via a CLI parameter.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("media_sorter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'clustering': {
        'algorithm': 'hdbscan',  # 'dbscan' or 'hdbscan'
        'eps': 0.15,
        'min_samples': 3,
        'visual_weight': 0.7,
        'temporal_weight': 0.3
    },
    'processing': {
        'batch_size': 32,
        'max_workers': 4,
        'video_interval': 30,  # in seconds
        'max_file_size_mb': 500
    },
    'safety': {
        'dry_run': False,
        'min_disk_space_gb': 5,
        'checksum_verify': True
    }
}


class MediaProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = self._load_model()
        self.file_hashes = set()
        # Lock to ensure thread safety when calling the CLIP model
        self.model_lock = threading.Lock()

    def _load_model(self):
        """Load CLIP model with error handling."""
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            sys.exit(1)

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Main processing pipeline."""
        try:
            if not input_dir.exists() or not input_dir.is_dir():
                logger.error(f"Input directory {input_dir} does not exist or is not a directory.")
                sys.exit(1)

            files = self._gather_files(input_dir)
            if not files:
                logger.warning("No valid media files found")
                return

            logger.info(f"Processing {len(files)} files...")

            # Extract features along with the corresponding file paths.
            processed_files, features = self._extract_features(files)
            if not features:
                logger.error("No features were extracted. Exiting.")
                sys.exit(1)

            clusters = self._cluster_files(features)
            self._organize_files(processed_files, clusters, output_dir)

        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            sys.exit(1)

    def _gather_files(self, input_dir: Path) -> List[Path]:
        """Collect media files with validation."""
        media_files = []
        allowed_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | RAW_EXTENSIONS

        for path in tqdm(list(input_dir.rglob('*')), desc="Scanning files"):
            if path.name.startswith('._') or not path.is_file():
                continue
            if path.suffix.lower() in allowed_exts:
                if self._validate_file(path):
                    media_files.append(path)
        return media_files

    def _validate_file(self, path: Path) -> bool:
        """Perform safety checks on files."""
        global verbose #deb - #note debug line
        verbose = True #deb - #note debug line
        global process_large_files
        # Size check
        max_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        if path.stat().st_size > max_size:
            if not process_large_files:
                logger.warning("Skipping large file: %s", path)
                return False
            else:
                logger.warning("Large file found, processing: %s", path)
        else:
            if verbose:
                logger.warning("processing: %s", path)
        # Checksum verification for deduplication
        file_hash = self._calculate_hash(path)
        if self.config['safety']['checksum_verify']:
            if file_hash in self.file_hashes:
                logger.info("Skipping duplicate: %s", path)
                return False
            self.file_hashes.add(file_hash)

        return True

    def _calculate_hash(self, path: Path) -> str:
        """Calculate file hash for deduplication."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_features(self, files: List[Path]) -> Tuple[List[Path], List[Tuple[np.ndarray, float]]]:
        """
        Batch process files for visual and temporal features.
        Returns a tuple of:
          - List of files that were successfully processed.
          - List of tuples containing (visual_embedding, timestamp).
        """
        processed_files = []
        features = []

        with ThreadPoolExecutor(max_workers=self.config['processing']['max_workers']) as executor:
            future_to_path = {executor.submit(self._process_single_file, path): path for path in files}

            for future in tqdm(as_completed(future_to_path), total=len(files), desc="Processing files"):
                try:
                    result = future.result()
                    if result is not None:
                        file_path, feat = result
                        processed_files.append(file_path)
                        features.append(feat)
                except Exception as e:
                    logger.error(f"Failed to process file: {e}")

        return processed_files, features

    def _process_single_file(self, path: Path) -> Optional[Tuple[Path, Tuple[np.ndarray, float]]]:
        """
        Process an individual file for features.
        Returns a tuple of (file_path, (visual_feature, timestamp)).
        """
        try:
            suffix = path.suffix.lower()
            if suffix in (IMAGE_EXTENSIONS | RAW_EXTENSIONS):
                visual_feat = self._get_single_image_embedding(path)
            elif suffix in VIDEO_EXTENSIONS:
                visual_feat = self._get_video_embedding(path)
            else:
                return None

            if visual_feat is None:
                return None

            timestamp = self._get_file_datetime(path).timestamp()

            return (path, (visual_feat, timestamp))
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return None

    def _cluster_files(self, features: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Hybrid visual-temporal clustering."""
        # Separate visual and temporal features
        visual_features = np.array([feat[0] for feat in features])
        temporal_features = np.array([feat[1] for feat in features]).reshape(-1, 1)

        # Normalize features
        visual_scaled = StandardScaler().fit_transform(visual_features)
        temporal_scaled = StandardScaler().fit_transform(temporal_features)

        # Use both weights explicitly (ensuring they sum to 1)
        vw = self.config['clustering'].get('visual_weight', 0.7)
        tw = self.config['clustering'].get('temporal_weight', 0.3)
        total = vw + tw
        vw /= total
        tw /= total

        combined_features = np.hstack([
            vw * visual_scaled,
            tw * temporal_scaled
        ])

        # Choose clustering algorithm
        if self.config['clustering']['algorithm'] == 'hdbscan':
            clusterer = HDBSCAN(
                min_samples=self.config['clustering']['min_samples'],
                metric='euclidean'
            )
        else:
            clusterer = DBSCAN(
                eps=self.config['clustering']['eps'],
                min_samples=self.config['clustering']['min_samples'],
                metric='euclidean'
            )

        return clusterer.fit_predict(combined_features)

    def _organize_files(self, files: List[Path], clusters: np.ndarray, output_dir: Path):
        """
        Organize files into scene directories.
        Files are moved directly into their scene folder (a flat, one-level folder structure).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check disk space
        total_size = sum(f.stat().st_size for f in files)
        self._check_disk_space(output_dir, total_size)

        # Create cluster directories (scene folders)
        cluster_ids = set(clusters)
        for cid in cluster_ids:
            dir_path = output_dir / (f"scene_{cid}" if cid != -1 else "noise")
            dir_path.mkdir(parents=True, exist_ok=True)

        # Move files into the corresponding scene folder using just the file's basename.
        for path,