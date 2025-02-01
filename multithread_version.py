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
        for path, cluster_id in zip(files, clusters):
            dest_dir = output_dir / (f"scene_{cluster_id}" if cluster_id != -1 else "noise")
            dest_path = self._get_unique_dest_path(dest_dir, path.name)

            if self.config['safety']['dry_run']:
                logger.info("[Dry Run] Would move %s → %s", path, dest_path)
                continue

            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest_path))
                logger.info("Moved %s → %s", path, dest_path)
            except Exception as e:
                logger.error("Failed to move %s: %s", path, e)

    def _get_unique_dest_path(self, dest_dir: Path, filename: str) -> Path:
        """
        Ensure that the file name in the destination directory is unique.
        If a file with the same name exists, a counter is appended.
        """
        original_stem = Path(filename).stem
        original_suffix = Path(filename).suffix
        dest_path = dest_dir / filename
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{original_stem}_{counter}{original_suffix}"
            counter += 1
        return dest_path

    def _check_disk_space(self, output_dir: Path, required_bytes: int):
        """Verify sufficient disk space exists on the destination."""
        stat = shutil.disk_usage(output_dir)
        min_space = self.config['safety']['min_disk_space_gb'] * 1024**3

        if stat.free < required_bytes + min_space:
            raise RuntimeError(
                f"Insufficient disk space. Required: {required_bytes/1024**2:.2f}MB, "
                f"Available: {stat.free/1024**3:.2f}GB"
            )

    def _get_file_datetime(self, path: Path) -> datetime:
        """
        Return the file's date and time.
        Here we use the file's modification time.
        """
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception as e:
            logger.error(f"Failed to get datetime for {path}: {e}")
            return datetime.now()

    # ---------------------------
    # Media Processing Core Methods
    # ---------------------------
    def _get_image_embedding(self, image_paths: List[Path]) -> List[np.ndarray]:
        """
        Process a batch of images (including RAW) and return a list of embeddings.
        Expects a list of image Paths.
        """
        try:
            preprocessed = []
            valid_paths = []

            for path in image_paths:
                try:
                    if path.suffix.lower() in RAW_EXTENSIONS:
                        with rawpy.imread(str(path)) as raw:
                            rgb = raw.postprocess()
                            pil_image = Image.fromarray(rgb)
                    else:
                        pil_image = Image.open(path)
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')

                    preprocessed.append(self.preprocess(pil_image).to(self.device))
                    valid_paths.append(path)
                except (rawpy.LibRawFileUnsupportedError, UnidentifiedImageError) as e:
                    logger.error("Unsupported image format: %s (%s)", path, e)
                except Exception as e:
                    logger.error("Error processing %s: %s", path, e)

            if not preprocessed:
                return []

            batch = torch.stack(preprocessed)
            with self.model_lock:
                with torch.no_grad():
                    embeddings = self.model.encode_image(batch).cpu().numpy()

            # Map embeddings back to the original order of image_paths
            embedding_map = {p: emb for p, emb in zip(valid_paths, embeddings)}
            return [embedding_map[p] for p in image_paths if p in embedding_map]

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.warning("GPU OOM error, falling back to CPU processing")
                return self._get_image_embedding_fallback(image_paths)
            raise
        except Exception as e:
            logger.error("Batch image processing failed: %s", e)
            return []

    def _get_single_image_embedding(self, path: Path) -> Optional[np.ndarray]:
        """
        Process a single image file (or RAW) and return its embedding.
        """
        embeddings = self._get_image_embedding([path])
        if embeddings:
            return embeddings[0]
        return None

    def _get_video_embedding(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Process a video by sampling frames and return the averaged embedding.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Could not open video: %s", video_path)
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Fallback if FPS is not available
            frame_interval = int(fps * self.config['processing']['video_interval'])

            frame_batch = []
            embeddings = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_number % frame_interval == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        preprocessed = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                        frame_batch.append(preprocessed)

                        if len(frame_batch) >= self.config['processing']['batch_size']:
                            batch_tensor = torch.cat(frame_batch)
                            with self.model_lock:
                                with torch.no_grad():
                                    batch_emb = self.model.encode_image(batch_tensor).cpu().numpy()
                            embeddings.extend(batch_emb)
                            frame_batch = []

                    pbar.update(1)

            # Process any remaining frames
            if frame_batch:
                batch_tensor = torch.cat(frame_batch)
                with self.model_lock:
                    with torch.no_grad():
                        batch_emb = self.model.encode_image(batch_tensor).cpu().numpy()
                embeddings.extend(batch_emb)

            cap.release()

            if not embeddings:
                logger.warning("No valid frames extracted from %s", video_path)
                return None

            return np.mean(embeddings, axis=0)

        except Exception as e:
            logger.error("Video processing failed for %s: %s", video_path, e)
            return None

    def _get_image_embedding_fallback(self, image_paths: List[Path]) -> List[np.ndarray]:
        """
        CPU fallback for image processing when GPU memory is exhausted.
        """
        original_device = self.device
        try:
            self.device = "cpu"
            self.model = self.model.to(self.device)
            logger.info("Falling back to CPU processing for images")
            return self._get_image_embedding(image_paths)
        finally:
            self.device = original_device
            self.model = self.model.to(self.device)
    # ---------------------------
    # End of Media Processing Core Methods
    # ---------------------------


def load_config(config_path: Path) -> dict:
    """Load configuration from a YAML file, or return the default config."""
    if config_path.exists():
        try:
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
                return loaded_config
        except Exception as e:
            logger.error("Failed to load configuration file: %s", e)
    return DEFAULT_CONFIG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize media files by visual scenes")
    parser.add_argument('--input', type=Path, required=True, help="Input directory")
    parser.add_argument('--output', type=Path, required=True, help="Output directory")
    parser.add_argument('--config', type=Path, default=Path('config.yaml'), help="Config file path")
    parser.add_argument('--dry-run', action='store_true', help="Simulate without moving files")
    parser.add_argument('--process-large-files', action='store_true',
                        help="Process large files even if they exceed the size limit (by default large files are skipped)")
    args = parser.parse_args()

    # Load configuration and override dry-run setting if provided
    config = load_config(args.config)
    config.setdefault('safety', {})['dry_run'] = args.dry_run

    # Update the global variable based on CLI parameter.
    process_large_files = args.process_large_files

    # Initialize and run the processor
    processor = MediaProcessor(config)
    processor.process_directory(args.input, args.output)


"""
Notes:
The multithreading could have also been applied in the scanning loop (unless it is not that way on purpose for some reason).
Between the DBScan loops and the previous file scan we should save progress as a pickle file in case something happens to the runtime.
Allow the user to resume the sort if it crashed from the pickle file.
Maybe have the code process images first then process videos and add them to the same clusterings. Unless this reduces quality.
Also add the epsilon value as a parameter in the CLI.
- if the clustering loop reaches a certain certainty for a video clip regarding what group it should be in, could we terminate that file's processing (keyframes) early? Add this as a global parameter at the top of the file if it makes sense.
We don't have enought data on the results of the analysis. Output model scorings, certainty scores, error ranges, and all other relevant data in a well-formatted comment block within the code's output.
_extended_summary_
"""