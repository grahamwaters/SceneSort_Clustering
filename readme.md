
### `media_scene_sorter.py`

---

### `README.md`

# Enhanced Media Scene Sorter

This tool processes media files (images, RAW files, and videos) by extracting visual and temporal features, clustering them into scenes, and organizing them into separate directories. It uses the CLIP model for visual embeddings and supports both DBSCAN and HDBSCAN clustering.

## Features

- **Media Support:** Handles images (JPEG, PNG, etc.), RAW files, and common video formats.
- **Hybrid Clustering:** Combines visual and temporal features for clustering.
- **Resumable Processing:** Resume an interrupted run using a progress pickle file.
- **Video Optimization:** Option for early termination during video keyframe extraction when variance falls below a threshold.
- **Safety Checks:** File size limits, duplicate detection via checksums, and disk space verification.
- **Dry Run Option:** Simulate file moves without affecting the filesystem.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [CLIP](https://github.com/openai/CLIP)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [rawpy](https://github.com/letmaik/rawpy)
- [scikit-learn](https://scikit-learn.org/)
- [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
- [PyYAML](https://pyyaml.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install the required packages using pip:

```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python pillow rawpy scikit-learn hdbscan pyyaml
```

## Setup

1. **Clone or Download the Repository**

   ```bash
   git clone https://github.com/yourusername/media_scene_sorter.git
   cd media_scene_sorter
   ```

2. **(Optional) Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Prepare a Configuration File (Optional)**

   Create a `config.yaml` file to override default settings. For example:

   ```yaml
   clustering:
     algorithm: hdbscan
     eps: 0.15
     min_samples: 3
     visual_weight: 0.7
     temporal_weight: 0.3
   processing:
     batch_size: 32
     max_workers: 4
     video_interval: 30
     max_file_size_mb: 500
   safety:
     dry_run: false
     min_disk_space_gb: 5
     checksum_verify: true
   video_certainty_threshold: 0.05
   ```

## Usage

Run the script from the command line with the required arguments:

```bash
python media_scene_sorter.py --input /path/to/input_directory --output /path/to/output_directory
```

### Command-Line Options

- `--input`: **(Required)** Path to the input directory containing media files.
- `--output`: **(Required)** Path to the output directory where organized scene folders will be created.
- `--config`: Path to a YAML configuration file (default: `config.yaml`).
- `--dry-run`: Simulate file moves without actually moving files.
- `--process-large-files`: Process files that exceed the size limit (default behavior is to skip them).
- `--eps`: Override the `eps` parameter for DBSCAN clustering.
- `--resume`: Resume from a saved progress file (`progress.pkl`) if available.
- `--video-certainty-threshold`: Set an early termination threshold for video processing (e.g., `0.05`).
- `--images-first`: Process images before videos.

### Example

To process media from `/media/input` and organize scenes in `/media/output` while processing large files and setting a video certainty threshold:

```bash
python media_scene_sorter.py \
  --input /media/input \
  --output /media/output \
  --process-large-files \
  --video-certainty-threshold 0.05
```

## Logging

- **General logs:** Written to `media_sorter.log` and also printed to the console.
- **Error logs:** Additional warnings and errors are logged to `error.log`.

## Resuming Interrupted Runs

If the process is interrupted, you can resume by passing the `--resume` flag. The script saves its progress in a file named `progress.pkl`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---

This version of the media scene sorter has been updated to remove the live dashboard functionality. Follow the instructions in the README to set up and run the tool.