#!/usr/bin/env python3
"""
Enhanced Media Scene Sorter with Hybrid Clustering

Extended Summary:
- Processes media files (images, RAW files, videos) and clusters them based on visual and temporal features.
- Can resume from a previously saved progress file if a run was interrupted.
- Supports processing images first and then videos.
- Allows early termination of video keyframe extraction if the embedding variance falls below a specified threshold.
- Outputs a detailed analysis summary including cluster distribution.
"""

import os
import sys
import argparse
import shutil
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

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
from hdbscan import HDBSCAN

from concurrent.futures import ThreadPoolExecutor, as_completed

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
RAW_EXTENSIONS   = {".raw", ".cr2", ".nef", ".dng", ".arw", ".rw2"}

# Global flag to control processing of large files (set via CLI)
process_large_files = False

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("media_sorter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
error_handler = logging.FileHandler("error.log", mode="a")
error_handler.setLevel(logging.WARNING)
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
error_handler.setFormatter(error_formatter)
logger.addHandler(error_handler)

# Default configuration
DEFAULT_CONFIG = {
    'clustering': {
        'algorithm': 'hdbscan',  # Options: 'dbscan' or 'hdbscan'
        'eps': 0.19, # was 0.15
        'min_samples': 3,
        'visual_weight': 0.8, # .7
        'temporal_weight': 0.2 # .3
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
    },
    # Optional parameter for early termination in video processing
    'video_certainty_threshold': None  # e.g., 0.05 (set via CLI)
}


class MediaProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = self._load_model()
        self.file_hashes = set()
        # Lock for thread safety when calling the CLIP model
        self.model_lock = None

    def _load_model(self):
        """Load the CLIP model."""
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            sys.exit(1)

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Main processing pipeline."""
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory {input_dir} does not exist or is not a directory.")
            sys.exit(1)

        input_files = self._gather_files(input_dir)
        processed_basenames = {f.stem for f in output_dir.rglob("*") if f.is_file()} if output_dir.exists() else set()
        files = [f for f in input_files if f.stem not in processed_basenames]

        if not files:
            logger.info("No new media files to process (all already processed).")
            return

        logger.info("Processing %d files...", len(files))
        # Either resume from a pickle or process from scratch
        if args.resume and os.path.exists("progress.pkl"):
            with open("progress.pkl", "rb") as pf:
                progress = pickle.load(pf)
            processed_files, features = progress["processed_files"], progress["features"]
            logger.info("Resumed progress from pickle file with %d files.", len(processed_files))
        else:
            if args.images_first:
                files = sorted(files, key=lambda f: 0 if f.suffix.lower() in (IMAGE_EXTENSIONS | RAW_EXTENSIONS) else 1)
            processed_files, features = self._extract_features(files)
            with open("progress.pkl", "wb") as pf:
                pickle.dump({"processed_files": processed_files, "features": features}, pf)
            logger.info("Saved progress to pickle file with %d files.", len(processed_files))

        clusters = self._cluster_files(features)
        self._organize_files(processed_files, clusters, output_dir)

    def _gather_files(self, input_dir: Path) -> List[Path]:
        """Collect media files from the input directory."""
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
        """Perform safety checks on the file."""
        global process_large_files
        max_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        if path.stat().st_size > max_size:
            if not process_large_files:
                logger.warning("Skipping large file: %s", path)
                return False
            logger.warning("Large file found, processing: %s", path)
        else:
            logger.warning("\tProcessing: %s", path)
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
        Batch process files to extract visual and temporal features.
        Returns a tuple:
          - List of processed file paths.
          - List of tuples: (visual_embedding, timestamp).
        """
        processed_files = []
        features = []
        with ThreadPoolExecutor(max_workers=self.config['processing']['max_workers']) as executor:
            future_to_path = {executor.submit(self._process_single_file, path): path for path in files}
            for future in tqdm(as_completed(future_to_path), total=len(files), desc="Extracting features"):
                try:
                    result = future.result()
                    if result is not None:
                        file_path, feat = result
                        processed_files.append(file_path)
                        features.append(feat)
                except Exception as e:
                    logger.error("Failed to process file: %s", e)
        return processed_files, features

    def _process_single_file(self, path: Path) -> Optional[Tuple[Path, Tuple[np.ndarray, float]]]:
        """
        Process a single file and return (file_path, (visual_feature, timestamp)).
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
            logger.error("Error processing %s: %s", path, e)
            return None

    def _cluster_files(self, features: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Perform hybrid visual-temporal clustering."""
        visual_features = np.array([feat[0] for feat in features])
        temporal_features = np.array([feat[1] for feat in features]).reshape(-1, 1)

        if visual_features.size == 0:
            logger.warning("No valid visual features extracted, skipping clustering.")
            return np.array([])

        if visual_features.ndim == 1:
            visual_features = visual_features.reshape(-1, 1)

        visual_scaled = StandardScaler().fit_transform(visual_features)
        temporal_scaled = StandardScaler().fit_transform(temporal_features)

        vw = self.config['clustering'].get('visual_weight', 0.7)
        tw = self.config['clustering'].get('temporal_weight', 0.3)
        total = vw + tw
        vw /= total
        tw /= total

        combined_features = np.hstack([vw * visual_scaled, tw * temporal_scaled])

        if self.config['clustering']['algorithm'] == 'hdbscan':
            clusterer = HDBSCAN(min_samples=self.config['clustering']['min_samples'],
                                metric='euclidean')
        else:
            clusterer = DBSCAN(eps=self.config['clustering']['eps'],
                               min_samples=self.config['clustering']['min_samples'],
                               metric='euclidean')

        clusters = clusterer.fit_predict(combined_features)
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_distribution = dict(zip(unique, counts))
        logger.info(
            "Clustering Summary:\n"
            "  - Total files processed: %d\n"
            "  - Number of clusters (excluding noise): %d\n"
            "  - Cluster distribution: %s",
            len(combined_features),
            len(unique) - (1 if -1 in unique else 0),
            cluster_distribution
        )
        return clusters

    def _organize_files(self, files: List[Path], clusters: np.ndarray, output_dir: Path):
        """
        Move files into scene directories based on cluster labels.
        Files are organized into a flat, one-level structure.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        total_size = sum(f.stat().st_size for f in files)
        self._check_disk_space(output_dir, total_size)

        # Create subdirectories for each cluster
        for cid in set(clusters):
            scene_dir = output_dir / (f"scene_{cid}" if cid != -1 else "noise")
            scene_dir.mkdir(parents=True, exist_ok=True)

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
        """Ensure the destination filename is unique."""
        original_stem = Path(filename).stem
        original_suffix = Path(filename).suffix
        dest_path = dest_dir / filename
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{original_stem}_{counter}{original_suffix}"
            counter += 1
        return dest_path

    def _check_disk_space(self, output_dir: Path, required_bytes: int):
        """Verify sufficient disk space on the destination drive."""
        stat = shutil.disk_usage(output_dir)
        min_space = self.config['safety']['min_disk_space_gb'] * 1024 ** 3
        if stat.free < required_bytes + min_space:
            raise RuntimeError(
                f"Insufficient disk space. Required: {required_bytes/1024**2:.2f} MB, "
                f"Available: {stat.free/1024**3:.2f} GB"
            )

    def _get_file_datetime(self, path: Path) -> datetime:
        """Return the file's modification time as a datetime object."""
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception as e:
            logger.error("Failed to get datetime for %s: %s", path, e)
            return datetime.now()

    def _get_image_embedding(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Process a batch of images (or RAW files) and return embeddings."""
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
        with torch.no_grad():
            embeddings = self.model.encode_image(batch).cpu().numpy()
        embedding_map = {p: emb for p, emb in zip(valid_paths, embeddings)}
        return [embedding_map[p] for p in image_paths if p in embedding_map]

    def _get_single_image_embedding(self, path: Path) -> Optional[np.ndarray]:
        """Process a single image (or RAW file) and return its embedding."""
        embeddings = self._get_image_embedding([path])
        return embeddings[0] if embeddings else None

    def _get_video_embedding(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Process a video by sampling frames and return the averaged embedding.
        Early termination is applied if the batch's standard deviation is below the threshold.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Could not open video: %s", video_path)
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30
            frame_interval = int(fps * self.config['processing']['video_interval'])
            frame_batch = []
            embeddings = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_threshold = self.config.get("video_certainty_threshold", None)
            batch_counter = 0

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
                        batch_counter += 1
                        batch_tensor = torch.cat(frame_batch)
                        with torch.no_grad():
                            batch_emb = self.model.encode_image(batch_tensor).cpu().numpy()
                        std_val = float(np.mean(np.std(batch_emb, axis=0))) if len(batch_emb) > 1 else 0.0
                        if video_threshold is not None and std_val < video_threshold:
                            logger.info("Early termination for video %s due to low variance (std=%.4f)", video_path, std_val)
                            embeddings.extend(batch_emb)
                            break
                        embeddings.extend(batch_emb)
                        frame_batch = []

            if frame_batch:
                batch_tensor = torch.cat(frame_batch)
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


def load_config(config_path: Path) -> dict:
    """Load configuration from a YAML file if it exists; otherwise, return the default config."""
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
                        help="Process large files even if they exceed the size limit")
    parser.add_argument('--eps', type=float, default=None, help="Epsilon value for DBSCAN clustering (overrides config)")
    parser.add_argument('--resume', action='store_true', help="Resume from progress pickle file if available")
    parser.add_argument('--video-certainty-threshold', type=float, default=None,
                        help="Early termination threshold for video processing keyframes (e.g., 0.05)")
    parser.add_argument('--images-first', action='store_true', help="Process images first, then videos")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault('safety', {})['dry_run'] = args.dry_run
    if args.eps is not None:
        config['clustering']['eps'] = args.eps
    if args.video_certainty_threshold is not None:
        config['video_certainty_threshold'] = args.video_certainty_threshold

    process_large_files = args.process_large_files
    processor = MediaProcessor(config)

    if args.resume and os.path.exists("progress.pkl"):
        with open("progress.pkl", "rb") as pf:
            progress = pickle.load(pf)
        processed_files, features = progress["processed_files"], progress["features"]
        logger.info("Resumed progress from pickle file with %d files.", len(processed_files))
    else:
        files = processor._gather_files(args.input)
        if args.images_first:
            files = sorted(files, key=lambda f: 0 if f.suffix.lower() in (IMAGE_EXTENSIONS | RAW_EXTENSIONS) else 1)
        processed_files, features = processor._extract_features(files)
        with open("progress.pkl", "wb") as pf:
            pickle.dump({"processed_files": processed_files, "features": features}, pf)
        logger.info("Saved progress to pickle file with %d files.", len(processed_files))

    clusters = processor._cluster_files(features)
    processor._organize_files(processed_files, clusters, args.output)

    unique, counts = np.unique(clusters, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    summary = (
        "\n# Analysis Summary:\n"
        f"# Total files processed: {len(processed_files)}\n"
        f"# Number of clusters (excluding noise): {len(unique) - (1 if -1 in unique else 0)}\n"
        f"# Cluster distribution: {cluster_distribution}\n"
        "# Additional metrics can be computed and added here.\n"
    )
    logger.info(summary)