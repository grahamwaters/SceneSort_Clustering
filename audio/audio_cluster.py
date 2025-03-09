import os
import shutil
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Paths
audio_folder = "/Volumes/BigBoy/portugal_movie/AUDIO_EXTRACTED"
output_folder = os.path.join(os.path.dirname(audio_folder), "AUDIO_CLUSTERS")
n_clusters = 5  # Adjust as needed
console = Console()

# Ensure ffmpeg is installed
if shutil.which("ffmpeg") is None:
    console.print("[bold red]Error:[/bold red] ffmpeg is not installed. Please install it to process audio files.")
    exit(1)

# Function to clean and convert audio files
def preprocess_audio(file_path):
    """
    Ensures audio is in WAV format and removes potential encoding issues.
    Returns the path to the processed file.
    """
    try:
        if file_path.lower().endswith(".mp3"):
            wav_path = file_path.replace(".mp3", ".wav")
            if not os.path.exists(wav_path):  # Convert only if necessary
                audio = AudioSegment.from_file(file_path, format="mp3")
                audio.export(wav_path, format="wav")
            return wav_path
        return file_path
    except Exception as e:
        console.print(f"[bold red]Failed to process {file_path}:[/bold red] {e}")
        return None

# Function to extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < 512:  # Ignore very short files
            console.print(f"[bold yellow]Skipping short file:[/bold yellow] {file_path}")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        console.print(f"[bold red]Error processing {file_path}:[/bold red] {e}")
        return None

# Collect audio files, filtering out system files
file_names = [
    f for f in os.listdir(audio_folder)
    if f.endswith((".wav", ".mp3", ".flac")) and not f.startswith("._")
]

features, valid_files = [], []

console.print("[bold cyan]Preprocessing audio files...[/bold cyan]")
for file in track(file_names, description="Cleaning & Converting"):
    file_path = os.path.join(audio_folder, file)
    processed_path = preprocess_audio(file_path)
    if processed_path:
        feat = extract_features(processed_path)
        if feat is not None:
            features.append(feat)
            valid_files.append(file)

# Ensure we have valid features before proceeding
if not features:
    console.print("[bold red]No valid audio files found for clustering![/bold red]")
    exit(1)

features = np.array(features)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-means clustering
console.print("[bold green]Clustering audio files...[/bold green]")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)

# Organize files into clusters
clusters = defaultdict(list)
for file, label in zip(valid_files, labels):
    clusters[label].append(file)

# Create output directories and move files
os.makedirs(output_folder, exist_ok=True)
console.print("[bold cyan]Sorting files into cluster folders...[/bold cyan]")

for cluster_id, files in track(clusters.items(), description="Organizing Files"):
    cluster_dir = os.path.join(output_folder, f"Cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    for file in files:
        src = os.path.join(audio_folder, file)
        dst = os.path.join(cluster_dir, file)
        shutil.copy2(src, dst)

# Display results in a table
table = Table(title="Audio Clustering Results")
table.add_column("Cluster ID", justify="center", style="bold yellow")
table.add_column("Audio Files", style="bold magenta")

for cluster_id, files in clusters.items():
    table.add_row(str(cluster_id), "\n".join(files))

console.print(table)
console.print(f"[bold green]Clustered audio files saved in: {output_folder}[/bold green]")
