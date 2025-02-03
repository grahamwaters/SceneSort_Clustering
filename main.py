#!/usr/bin/env python3
"""
Enhanced Media Scene Sorter with Robust Adaptive Iterative Processing

Extended Summary:
1. At any termination point, progress is saved (via periodic checkpoints and signal handlers).
2. Before processing begins, the destination is inspected:
   - If scene folders already exist, then:
     a. If the 'noise' folder contains files, those files are processed first.
     b. Else, new files from the input directory (that aren’t already in a scene) are sorted into the existing scenes,
        with new scenes added as needed.
   - Otherwise, process all files normally.
3. Files are traversed using an optimized method (os.walk) and moved as fast as possible.
   - If --fix-dates is supplied and a file’s created date is AFTER its modified date,
     then the created date is updated to match the modified date.
4. Adaptive iterative processing:
   - In each iteration, unsorted files are re-clustered with epsilon increased by 0.05.
   - A counter tracks consecutive iterations with no file movements.
   - If four consecutive iterations yield no changes, the loop exits, leaving those ambiguous files unsorted.
5. Duplication within the noise folder is prevented.
"""

import os
import sys
import argparse
import shutil
import logging
import hashlib
import pickle
import signal
import time
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
from hdbscan import HDBSCAN  # Optional alternative clustering

from concurrent.futures import ThreadPoolExecutor, as_completed

# For CLI coloring in iterative rounds.
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR_CYCLE = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
except ImportError:
    COLOR_CYCLE = [""] * 6
    class DummyStyle:
        RESET_ALL = ""
    Style = DummyStyle()

# Supported file extensions.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
RAW_EXTENSIONS   = {".raw", ".cr2", ".nef", ".dng", ".arw", ".rw2"}

# Global flags (updated via CLI)
process_large_files = False
fix_dates = False

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("media_sorter.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
error_handler = logging.FileHandler("error.log", mode="a")
error_handler.setLevel(logging.WARNING)
error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
error_handler.setFormatter(error_formatter)
logger.addHandler(error_handler)

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
    },
    'video_certainty_threshold': None  # e.g. 0.05 (set via CLI)
}

CHECKPOINT_FILE = "progress.pkl"

def save_progress(processed_files: List[Path], features: List[Tuple[np.ndarray, float]]):
    """Save checkpoint progress to a pickle file."""
    try:
        with open(CHECKPOINT_FILE, "wb") as pf:
            pickle.dump({"processed_files": processed_files, "features": features}, pf)
        logger.info("Checkpoint saved with %d files.", len(processed_files))
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)

def signal_handler(signum, frame):
    """On termination, save progress before exiting."""
    logger.info("Termination signal received; saving checkpoint before exit.")
    if MediaProcessor.instance is not None:
        save_progress(MediaProcessor.instance.processed_files, MediaProcessor.instance.features)
    sys.exit(0)

# Register termination signal handlers.
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class MediaProcessor:
    instance = None  # For global access in signal handler

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = self._load_model()
        self.file_hashes = set()
        self.processed_files: List[Path] = []
        self.features: List[Tuple[np.ndarray, float]] = []
        self.last_checkpoint_time = time.time()
        self.model_lock = torch.multiprocessing.Lock()  # Lock for thread-safe model access
        MediaProcessor.instance = self

    def _load_model(self):
        """Load the CLIP model and preprocessing function."""
        try:
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            model.eval()
            return model, preprocess
        except Exception as e:
            logger.error("Failed to load CLIP model: %s", e)
            sys.exit(1)

    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Main pipeline:
          1. Check the destination directory.
             - If scene folders exist, then:
               a. Process noise folder first if files exist.
               b. Otherwise, process new files from input not already in a scene.
          2. Otherwise, process all input files normally.
        """
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error("Input directory %s does not exist or is not a directory.", input_dir)
            sys.exit(1)

        # Check destination.
        destination_ready = False
        if output_dir.exists():
            scene_folders = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")]
            if scene_folders:
                destination_ready = True
                logger.info("Destination directory already contains scene folders.")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Determine source files.
        if destination_ready:
            noise_folder = output_dir / "noise"
            if noise_folder.exists() and any(p.is_file() and not p.name.startswith("._") for p in noise_folder.iterdir()):
                logger.info("Found files in the 'noise' folder; processing these first.")
                source_files = [p for p in noise_folder.iterdir() if p.is_file() and not p.name.startswith("._")]
            else:
                all_input_files = self._gather_files(input_dir)
                sorted_files = set()
                for scene in output_dir.glob("scene_*"):
                    sorted_files.update({p.stem for p in scene.glob("*")})
                source_files = [f for f in all_input_files if f.stem not in sorted_files]
                logger.info("No noise files found; processing new files from input into existing scenes.")
        else:
            source_files = self._gather_files(input_dir)

        # Resume from checkpoint if requested.
        if args.resume and os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "rb") as pf:
                progress = pickle.load(pf)
            self.processed_files, self.features = progress["processed_files"], progress["features"]
            logger.info("Resumed progress from checkpoint with %d files.", len(self.processed_files))
        else:
            if args.images_first:
                source_files = sorted(source_files, key=lambda f: 0 if f.suffix.lower() in (IMAGE_EXTENSIONS | RAW_EXTENSIONS) else 1)
            self.processed_files, self.features = self._extract_features(source_files)
            save_progress(self.processed_files, self.features)

        # Cluster and move files.
        clusters = self._cluster_files(self.features)
        self._organize_files(self.processed_files, clusters, output_dir)

        # If destination had scene folders, run adaptive iterative processing.
        if destination_ready:
            self._iterative_noise_processing(input_dir, output_dir)

    def _gather_files(self, input_dir: Path) -> List[Path]:
        """Collect media files using os.walk (skip files starting with '._')."""
        media_files = []
        allowed_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | RAW_EXTENSIONS
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.startswith("._"):
                    continue
                path = Path(root) / file
                if path.suffix.lower() in allowed_exts and self._validate_file(path):
                    media_files.append(path)
        return media_files

    def _validate_file(self, path: Path) -> bool:
        """Perform safety checks on files."""
        global process_large_files
        max_size = self.config['processing']['max_file_size_mb'] * 1024 * 1024
        if path.stat().st_size > max_size:
            if not process_large_files:
                logger.warning("Skipping large file: %s", path)
                return False
            else:
                logger.warning("Large file found, processing: %s", path)
        else:
            logger.info("Processing file: %s", path)
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
        Returns:
          - List of files processed.
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
                        if time.time() - self.last_checkpoint_time > 10:
                            save_progress(processed_files, features)
                            self.last_checkpoint_time = time.time()
                except Exception as e:
                    logger.error("Failed to process file: %s", e)
        return processed_files, features

    def _process_single_file(self, path: Path) -> Optional[Tuple[Path, Tuple[np.ndarray, float]]]:
        """
        Process a single file to obtain (visual_embedding, timestamp).
        Supports images, RAW images, and videos.
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

    def _cluster_files(self, features: List[Tuple[np.ndarray, float]], eps_override: Optional[float] = None) -> np.ndarray:
        """
        Perform hybrid visual-temporal clustering.
        Uses config values (or eps_override if provided) and weights from the configuration.
        """
        visual_features = np.array([feat[0] for feat in features])
        temporal_features = np.array([feat[1] for feat in features]).reshape(-1, 1)
        if visual_features.size == 0:
            logger.warning("No valid visual features extracted; skipping clustering.")
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
        eps = eps_override if eps_override is not None else self.config['clustering']['eps']

        if self.config['clustering']['algorithm'] == 'hdbscan':
            clusterer = HDBSCAN(min_samples=self.config['clustering']['min_samples'], metric='euclidean')
        else:
            clusterer = DBSCAN(eps=eps, min_samples=self.config['clustering']['min_samples'], metric='euclidean')

        clusters = clusterer.fit_predict(combined_features)
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_distribution = dict(zip(unique, counts))
        logger.info("Clustering Summary:\n  Total files processed: %d\n  Number of clusters (excluding noise): %d\n  Cluster distribution: %s",
                    len(combined_features),
                    len(unique) - (1 if -1 in unique else 0),
                    cluster_distribution)
        return clusters

    def _organize_files(self, files: List[Path], clusters: np.ndarray, output_dir: Path):
        """
        Move files into scene directories as determined by clustering.
        Uses optimized file moves (os.replace) and fixes dates if --fix-dates is supplied.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        total_size = sum(f.stat().st_size for f in files)
        self._check_disk_space(output_dir, total_size)
        for cid in set(clusters):
            scene_name = f"scene_{cid}" if cid != -1 else "noise"
            (output_dir / scene_name).mkdir(parents=True, exist_ok=True)
        for path, cluster_id in zip(files, clusters):
            dest_dir = output_dir / (f"scene_{cluster_id}" if cluster_id != -1 else "noise")
            dest_path = self._get_unique_dest_path(dest_dir, path.name)
            if self.config['safety']['dry_run']:
                logger.info("[Dry Run] Would move %s → %s", path, dest_path)
                continue
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(str(path), str(dest_path))
                except Exception:
                    shutil.move(str(path), str(dest_path))
                logger.info("Moved %s → %s", path, dest_path)
                if fix_dates:
                    self._fix_file_dates(dest_path)
            except Exception as e:
                logger.error("Failed to move %s: %s", path, e)
            save_progress(self.processed_files, self.features)

    def _get_unique_dest_path(self, dest_dir: Path, filename: str) -> Path:
        """Ensure the filename in the destination directory is unique."""
        original_stem = Path(filename).stem
        original_suffix = Path(filename).suffix
        dest_path = dest_dir / filename
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{original_stem}_{counter}{original_suffix}"
            counter += 1
        return dest_path

    def _fix_file_dates(self, path: Path):
        """
        If the file's created date is after its modified date,
        update the creation date to match the modified date.
        (Best-effort; behavior may vary by platform.)
        """
        try:
            stat = path.stat()
            created = stat.st_ctime
            modified = stat.st_mtime
            if created > modified:
                os.utime(str(path), (stat.st_atime, modified))
                logger.info("Fixed file dates for %s: set created date to modified date.", path)
        except Exception as e:
            logger.error("Failed to fix file dates for %s: %s", path, e)

    def _check_disk_space(self, output_dir: Path, required_bytes: int):
        """Ensure sufficient disk space exists at the destination."""
        stat = shutil.disk_usage(output_dir)
        min_space = self.config['safety']['min_disk_space_gb'] * 1024**3
        if stat.free < required_bytes + min_space:
            raise RuntimeError(f"Insufficient disk space. Required: {required_bytes/1024**2:.2f}MB, Available: {stat.free/1024**3:.2f}GB")

    def _get_file_datetime(self, path: Path) -> datetime:
        """Return the file's modification time as its datetime."""
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception as e:
            logger.error("Failed to get datetime for %s: %s", path, e)
            return datetime.now()

    def _get_image_embedding(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Process a batch of images (or RAW images) and return embeddings."""
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
        """Process a single image (or RAW image) and return its embedding."""
        embeddings = self._get_image_embedding([path])
        return embeddings[0] if embeddings else None

    def _get_video_embedding(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Process a video by sampling frames and averaging their embeddings.
        Early termination is applied if the standard deviation over a batch is below the threshold.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Could not open video: %s", video_path)
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            frame_interval = int(fps * self.config['processing']['video_interval'])
            frame_batch = []
            embeddings = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_threshold = self.config.get("video_certainty_threshold", None)
            batch_counter = 0
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
                            batch_counter += 1
                            batch_tensor = torch.cat(frame_batch)
                            with self.model_lock:
                                with torch.no_grad():
                                    batch_emb = self.model.encode_image(batch_tensor).cpu().numpy()
                            std_val = float(np.mean(np.std(batch_emb, axis=0))) if len(batch_emb) > 1 else 0.0
                            pbar.set_postfix(std=f"{std_val:.4f}")
                            if video_threshold is not None and std_val < video_threshold:
                                logger.info("Early termination for video %s due to low variance (std=%.4f)", video_path, std_val)
                                embeddings.extend(batch_emb)
                                break
                            embeddings.extend(batch_emb)
                            frame_batch = []
                    pbar.update(1)
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
        """CPU fallback for image processing when GPU memory is exhausted."""
        original_device = self.device
        try:
            self.device = "cpu"
            self.model = self.model.to(self.device)
            logger.info("Falling back to CPU processing for images")
            return self._get_image_embedding(image_paths)
        finally:
            self.device = original_device
            self.model = self.model.to(self.device)

    def _iterative_noise_processing(self, input_dir: Path, output_dir: Path):
        """
        Iteratively process unsorted files into the existing scene folders.
          - If the noise folder has files (excluding metadata files), process those.
          - Otherwise, look for new files in the input directory not already in any scene.
          - Increase epsilon by 0.05 each iteration.
          - If four consecutive iterations result in no file movements, exit the loop.
        """
        noise_folder = output_dir / "noise"
        iteration = 1
        current_eps = self.config['clustering']['eps']
        color_index = 0
        no_change_iterations = 0

        while True:
            # Determine source files.
            if noise_folder.exists() and any(p.is_file() and not p.name.startswith("._") for p in noise_folder.iterdir()):
                files = [p for p in noise_folder.iterdir() if p.is_file() and not p.name.startswith("._")]
                logger.info(COLOR_CYCLE[color_index % len(COLOR_CYCLE)] +
                            f"Iteration {iteration}: Re-processing noise files with eps={current_eps:.2f}" +
                            Style.RESET_ALL)
            else:
                files = self._gather_files(input_dir)
                sorted_files = set()
                for scene in output_dir.glob("scene_*"):
                    sorted_files.update({p.stem for p in scene.glob("*")})
                files = [f for f in files if f.stem not in sorted_files]
                logger.info(COLOR_CYCLE[color_index % len(COLOR_CYCLE)] +
                            f"Iteration {iteration}: Processing new files into existing scenes with eps={current_eps:.2f}" +
                            Style.RESET_ALL)
            if not files:
                logger.info("No unsorted files left to process.")
                break

            # Extract features and cluster.
            processed_files, features = self._extract_features(files)
            clusters = self._cluster_files(features, eps_override=current_eps)
            moved_count = 0
            for path, cluster_id in zip(processed_files, clusters):
                if cluster_id == -1:
                    dest_dir = noise_folder
                else:
                    scene_dir = output_dir / f"scene_{cluster_id}"
                    if scene_dir.exists():
                        dest_dir = scene_dir
                    else:
                        dest_dir = output_dir / f"scene_{cluster_id}"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = self._get_unique_dest_path(dest_dir, path.name)
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    # Check if file already exists at destination to avoid duplication.
                    if dest_path.exists():
                        logger.info("File %s already exists at destination; skipping.", dest_path)
                        continue
                    try:
                        os.replace(str(path), str(dest_path))
                    except Exception:
                        shutil.move(str(path), str(dest_path))
                    logger.info("Moved %s → %s", path, dest_path)
                    moved_count += 1
                    if fix_dates:
                        self._fix_file_dates(dest_path)
                except Exception as e:
                    logger.error("Failed to move %s: %s", path, e)
            # Update no-change counter.
            if moved_count == 0:
                no_change_iterations += 1
                logger.info("No files moved in iteration %d (consecutive no-change iterations: %d).", iteration, no_change_iterations)
            else:
                no_change_iterations = 0
            # If 4 consecutive iterations with no changes, exit.
            if no_change_iterations >= 4:
                logger.info("No changes in 4 consecutive iterations. Exiting iterative processing.")
                break
            iteration += 1
            current_eps += 0.05
            color_index += 1
            time.sleep(1)  # Brief pause between iterations.

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
    parser.add_argument('--process-large-files', action='store_true', help="Process large files even if they exceed the size limit")
    parser.add_argument('--eps', type=float, default=None, help="Epsilon value for clustering (overrides config)")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint if available")
    parser.add_argument('--video-certainty-threshold', type=float, default=None, help="Early termination threshold for video keyframes (e.g., 0.05)")
    parser.add_argument('--images-first', action='store_true', help="Process images first, then videos")
    parser.add_argument('--fix-dates', action='store_true', help="Fix file creation dates if after modified date")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault('safety', {})['dry_run'] = args.dry_run
    if args.eps is not None:
        config['clustering']['eps'] = args.eps
    if args.video_certainty_threshold is not None:
        config['video_certainty_threshold'] = args.video_certainty_threshold

    process_large_files = args.process_large_files
    fix_dates = args.fix_dates

    processor = MediaProcessor(config)
    processor.process_directory(args.input, args.output)

    unique, counts = np.unique(processor._cluster_files(processor.features), return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    summary = f"""
# Analysis Summary:
  Total files processed: {len(processor.processed_files)}
  Number of clusters (excluding noise): {len(unique) - (1 if -1 in unique else 0)}
  Cluster distribution: {cluster_distribution}
"""
    logger.info(summary)
