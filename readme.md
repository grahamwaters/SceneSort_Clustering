# 📂 Trip Media Sorter by Visual Scene Similarity

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://python.org)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)  
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI_CLIP-000000?logo=openai&logoColor=white)](https://github.com/openai/CLIP)  
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)  
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-FF6F00?logo=scikit-learn&logoColor=white)](https://github.com/scikit-learn-contrib/hdbscan)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 About

This tool processes media files (images, RAW files, and videos) by extracting visual and temporal features, clustering them into scenes, and organizing them into separate directories. It uses the **CLIP** model for visual embeddings and supports both **DBSCAN** and **HDBSCAN** clustering.

## ✨ Features

- 📸 **Media Support:** Handles images (JPEG, PNG, RAW), and common video formats.
- 🔍 **Hybrid Clustering:** Combines visual and temporal features.
- ♻️ **Resumable Processing:** Resume an interrupted run with a progress file.
- 🎞 **Video Optimization:** Early termination for efficient keyframe extraction.
- 🛡 **Safety Checks:** File size limits, duplicate detection, and disk space verification.
- 🧪 **Dry Run Option:** Simulate file organization before actual execution.

---

## 🔧 Installation

### 🛠 Requirements

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

### 🔽 Install Dependencies

```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python pillow rawpy scikit-learn hdbscan pyyaml
```

---

## 🚀 Setup & Usage

### 📥 Clone the Repository

```bash
git clone https://github.com/yourusername/media_scene_sorter.git
cd media_scene_sorter
```

### 🌐 (Optional) Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### ⚙️ (Optional) Custom Configuration

Create a `config.yaml` file to override default settings:

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

### ▶️ Run the Script

```bash
python media_scene_sorter.py --input /path/to/input_directory --output /path/to/output_directory
```

### 🛠 Command-Line Options

| Option | Description |
|--------|-------------|
| `--input` | **(Required)** Input directory path containing media files. |
| `--output` | **(Required)** Output directory for organized scenes. |
| `--config` | Path to custom YAML configuration (default: `config.yaml`). |
| `--dry-run` | Simulate file organization without making changes. |
| `--process-large-files` | Process files exceeding size limit. |
| `--eps` | Override `eps` for DBSCAN clustering. |
| `--resume` | Resume from saved progress (`progress.pkl`). |
| `--video-certainty-threshold` | Set early termination threshold for video processing. |
| `--images-first` | Process images before videos. |

#### 📌 Example Usage

```bash
python media_scene_sorter.py \
  --input /media/input \
  --output /media/output \
  --process-large-files \
  --video-certainty-threshold 0.05
```

---

## 📜 Logging & Debugging

- 📂 **General logs:** Stored in `media_sorter.log` & printed to console.
- ⚠️ **Error logs:** Critical errors are logged in `error.log`.

### 🔄 Resuming Interrupted Runs

If the process is interrupted, restart it with the `--resume` flag:

```bash
python media_scene_sorter.py --input /media/input --output /media/output --resume
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

🚀 **Happy Sorting!** 🎥📸
