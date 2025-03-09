#!/usr/bin/env python3
import os
import shutil
import numpy as np
import librosa
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict
import random
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from time import sleep
import sys
import subprocess
import logging

# Console and logging setup
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])

# ASCII Art
BANNER = """
[bold cyan]
    ██████╗ ██╗   ██╗██████╗ ██╗ ██████╗      ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗
    ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗    ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
    ███████║██║   ██║██║  ██║██║██║   ██║    ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝
    ██╔══██║██║   ██║██║  ██║██║██║   ██║    ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
    ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝    ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝      ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
    2025, Graham Waters
[/bold cyan]
"""

# Paths and constants
AUDIO_FOLDER = "/Volumes/BigBoy/portugal_movie/TESTING_AUDIO_EXTRACTED"
OUTPUT_FOLDER = os.path.join(os.path.dirname(AUDIO_FOLDER), "AUDIO_CLUSTERS")
MAX_CLUSTERS_DEFAULT = 30
MIN_CLUSTERS = 5
MIN_DURATION = 0.5
MIN_AMPLITUDE = 0.001
KEEP_ORIGINALS = True
MAX_WORKERS = min(4, os.cpu_count())
EXPECTED_FEATURE_DIM = 42  # 13 MFCC + 12 chroma + 7 spectral contrast + 5 others (centroid, flatness, rms, zero_crossing, tempo) + 5 onset

# Ensure FFmpeg is installed
if shutil.which("ffmpeg") is None:
    console.print("[bold red]Error:[/bold red] FFmpeg is not installed. Please install it first.")
    sys.exit(1)

# Validate audio file integrity and content
def validate_audio(file_path, check_content=False):
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', file_path, '-f', 'null', '-'],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=False
        )
        if result.returncode != 0:
            raise ValueError(f"FFmpeg validation failed: {result.stderr.decode()}")

        duration_line = [line for line in result.stderr.decode().splitlines() if "Duration" in line]
        if not duration_line:
            raise ValueError("No duration info available")
        duration_str = duration_line[0].split("Duration: ")[1].split(",")[0]
        h, m, s = map(float, duration_str.split(":"))
        duration = h * 3600 + m * 60 + s
        if duration < MIN_DURATION:
            raise ValueError(f"Audio too short: {duration:.2f}s")

        if check_content:
            y, sr = librosa.load(file_path, sr=None)
            rms = np.mean(librosa.feature.rms(y=y))
            if rms < MIN_AMPLITUDE:
                raise ValueError(f"No audible content (RMS: {rms:.6f})")
            console.print(f"[cyan]Validated {file_path}:[/cyan] RMS = {rms:.6f}, Size = {os.path.getsize(file_path) / 1024:.2f} KB")
        return True
    except Exception as e:
        console.print(f"[yellow]Skipping {file_path}:[/yellow] {str(e)}")
        return False

# Preprocess audio files
def preprocess_audio(file_path):
    try:
        console.print(f"[cyan]Processing {file_path}...[/cyan]")
        if not validate_audio(file_path, check_content=True):
            return None
        if file_path.lower().endswith(".mp3"):
            wav_path = file_path.replace(".mp3", "_temp.wav")
            if not os.path.exists(wav_path):
                result = subprocess.run(
                    ['ffmpeg', '-i', file_path, '-ac', '1', '-ar', '22050', wav_path],
                    stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True
                )
                console.print(f"[cyan]Converted {file_path} to {wav_path}:[/cyan] {result.stderr.decode().strip()}")
                if not validate_audio(wav_path, check_content=True):
                    os.remove(wav_path)
                    return None
            return wav_path
        return file_path
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]FFmpeg failed for {file_path}:[/bold red] {e.stderr.decode()}")
        return None
    except Exception as e:
        console.print(f"[bold red]Failed to preprocess {file_path}:[/bold red] {e}")
        return None

# Advanced feature extraction
def extract_features(file_path):
    if '._' in file_path:
        console.print(f"[bold red]'._' detected[/bold red]")
        return None, None

    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < 512:
            console.print(f"[yellow]Skipping {file_path}:[/yellow] Too short")
            return None, None

        rms = np.mean(librosa.feature.rms(y=y))
        if rms < MIN_AMPLITUDE:
            console.print(f"[yellow]Skipping {file_path}:[/yellow] No audible content (RMS: {rms:.6f})")
            return None, None

        n_fft = min(1024, len(y))
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        features = []

        try:
            console.print(f"\t[cyan]Processing file:[/cyan] {file_path}")

            # Reduced to 13 MFCCs (standard practice)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft), axis=1)
            console.print(f"\t[green]Extracted MFCCs:[/green] {mfcc.shape}")

            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft), axis=1) if np.any(y) else np.zeros(12)
            console.print(f"\t[green]Extracted Chroma:[/green] {chroma.shape}")

            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft))
            console.print(f"\t[green]Extracted Spectral Centroid:[/green] {spec_centroid}")

            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=n_fft))
            console.print(f"\t[green]Extracted Spectral Flatness:[/green] {spec_flatness}")

            spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft), axis=1) if len(y) >= n_fft else np.zeros(7)
            console.print(f"\t[green]Extracted Spectral Contrast:[/green] {spec_contrast.shape}")

            tempo_array, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = tempo_array[0] if tempo_array.size > 0 else 0.0  # Extract scalar, default to 0 if empty
            console.print(f"\t[green]Extracted Tempo:[/green] {tempo}")

            onset_env = np.mean(librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft))
            console.print(f"\t[green]Extracted Onset Envelope:[/green] {onset_env}")

            # Ensure middle section is a flat array
            middle_features = np.array([spec_centroid, spec_flatness, rms, zero_crossing, tempo])
            console.print(f"\t[green]Middle Features Shape:[/green] {middle_features.shape}")

            features = np.concatenate((mfcc, chroma, middle_features, spec_contrast))
            console.print(f"\t[blue]Final feature vector shape:[/blue] {features.shape}")

            # Adjust feature vector length if needed
            if len(features) != EXPECTED_FEATURE_DIM:
                console.print(f"\t[yellow]Adjusting features for {file_path}:[/yellow] Expected {EXPECTED_FEATURE_DIM}, got {len(features)}")
                features = np.pad(features, (0, EXPECTED_FEATURE_DIM - len(features)), 'constant') if len(features) < EXPECTED_FEATURE_DIM else features[:EXPECTED_FEATURE_DIM]

        except Exception as e:
            console.print(f"\t[red]Error extracting features for {file_path}:[/red] {e}")
            console.print(f"\t[yellow]Using fallback features for {file_path}[/yellow]")
            features = np.zeros(EXPECTED_FEATURE_DIM)
            features[13 + 12 + 2] = rms  # Index 27
            features[13 + 12 + 3] = zero_crossing  # Index 28

        console.print(f"[cyan]Features for {file_path}:[/cyan] Length = {len(features)}")
        return features, file_path
    except Exception as e:
        console.print(f"[bold red]Error extracting features from {file_path}:[/bold red] {e}")
        return None, None

# Find optimal number of clusters
def find_optimal_clusters(features_scaled, max_k=MAX_CLUSTERS_DEFAULT, min_k=MIN_CLUSTERS):
    if len(features_scaled) < min_k:
        return min_k if len(features_scaled) > 1 else 1
    max_k = min(int(np.sqrt(len(features_scaled))) + 1, max_k + 1, len(features_scaled))
    silhouette_scores = []
    for k in range(max(min_k, 2), max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        silhouette_scores.append((k, score))
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else min_k
    return optimal_k

# Dynamic cluster display
def generate_cluster_layout(clusters):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="clusters")
    )
    layout["header"].update(Panel(Text("Clustering in Progress", style="bold cyan", justify="center")))

    cluster_panels = []
    colors = ["blue", "green", "yellow", "magenta", "cyan", "red", "white", "bright_blue"]
    for cluster_id, files in clusters.items():
        color = random.choice(colors)
        title = Text(f"Cluster {cluster_id} ({len(files)} files)", style=f"bold {color}")
        file_list = Text("\n".join(files[:5] + (["..."] if len(files) > 5 else [])), style=f"dim {color}")
        cluster_panels.append(Panel(file_list, title=title, border_style=color))

    layout["clusters"].split_row(*cluster_panels)
    return layout

# Main processing function
def process_audio_files():
    console.print(BANNER)
    console.print("[bold green]Initializing Enhanced Audio Clustering Engine...[/bold green]\n")

    # Collect audio files
    file_names = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith((".wav", ".mp3", ".flac")) and not f.startswith("._")]
    if not file_names:
        console.print("[bold red]No audio files found in the directory![/bold red]")
        sys.exit(1)
    console.print(f"[bold cyan]Found {len(file_names)} audio files to process.[/bold cyan]")

    # Preprocessing
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task1 = progress.add_task("[cyan]Preprocessing Audio", total=len(file_names))
        processed_files = []
        for f in file_names:  # Sequential for stability
            result = preprocess_audio(os.path.join(AUDIO_FOLDER, f))
            if result:
                processed_files.append(result)
            progress.advance(task1)

    if not processed_files:
        console.print("[bold red]No valid audio files after preprocessing![/bold red]")
        sys.exit(1)
    console.print(f"[bold cyan]{len(processed_files)} files successfully preprocessed.[/bold cyan]")

    # Feature Extraction
    features, valid_files = [], []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task2 = progress.add_task("[yellow]Extracting Features", total=len(processed_files))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_features, f): f for f in processed_files}
            for future in as_completed(futures):
                feat, file = future.result()
                if feat is not None:
                    features.append(feat)
                    valid_files.append(file)
                progress.advance(task2)

    if not features:
        console.print("[bold red]No valid features extracted![/bold red]")
        sys.exit(1)
    console.print(f"[bold cyan]{len(features)} files with valid features extracted.[/bold cyan]")

    # Scale and reduce dimensionality
    features_array = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    pca = PCA(n_components=0.9)  # Retain 90% variance
    features_reduced = pca.fit_transform(features_scaled)
    console.print(f"[bold magenta]PCA reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions, explaining {sum(pca.explained_variance_ratio_)*100:.1f}% variance[/bold magenta]")

    # Determine optimal number of clusters
    console.print("\n[bold magenta]Determining Optimal Cluster Count...[/bold magenta]")
    optimal_k = find_optimal_clusters(features_reduced)
    console.print(f"[bold green]Optimal number of clusters: {optimal_k}[/bold green]")

    # Clustering
    console.print("\n[bold magenta]Performing K-Means Clustering...[/bold magenta]")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_reduced)
    silhouette_avg = silhouette_score(features_reduced, labels) if optimal_k > 1 else 1.0

    # Organize clusters
    clusters = defaultdict(list)
    for file, label in zip(valid_files, labels):
        clusters[label].append(os.path.basename(file))

    # Move files with live display
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with Live(generate_cluster_layout(clusters), refresh_per_second=10, console=console) as live:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            move_func = shutil.copy2 if KEEP_ORIGINALS else shutil.move
            for cluster_id, files in clusters.items():
                cluster_dir = os.path.join(OUTPUT_FOLDER, f"Cluster_{cluster_id:02d}")
                os.makedirs(cluster_dir, exist_ok=True)
                for file in files:
                    src = os.path.join(AUDIO_FOLDER, file)
                    dst = os.path.join(cluster_dir, file)
                    if os.path.exists(src) and validate_audio(src, check_content=True):
                        futures.append(executor.submit(move_func, src, dst))
                        console.print(f"[cyan]Moved {src} to {dst}[/cyan]")
                    else:
                        console.print(f"[yellow]Not moving {src}:[/yellow] No audio content")
            for _ in as_completed(futures):
                sleep(0.05)
                live.update(generate_cluster_layout(clusters))

    # Clean up temporary files
    for file in processed_files:
        if file.endswith("_temp.wav") and os.path.exists(file):
            os.remove(file)
            console.print(f"[cyan]Cleaned up {file}[/cyan]")

    # Cluster summary
    cluster_summary = Table(title="Cluster Characteristics", title_style="bold cyan", border_style="bright_green")
    cluster_summary.add_column("Cluster ID", justify="center", style="bold yellow")
    cluster_summary.add_column("File Count", justify="center", style="bold green")
    cluster_summary.add_column("Avg Tempo", justify="center", style="bold blue")
    cluster_summary.add_column("Avg Energy", justify="center", style="bold magenta")

    tempo_idx = 13 + 12 + 4  # Index 29 in original features
    rms_idx = 13 + 12 + 2    # Index 27 in original features
    for cluster_id in sorted(clusters.keys()):
        cluster_features = [f for i, f in enumerate(features) if labels[i] == cluster_id]
        avg_tempo = np.mean([f[tempo_idx] for f in cluster_features]) if cluster_features else 0
        avg_energy = np.mean([f[rms_idx] for f in cluster_features]) if cluster_features else 0
        cluster_summary.add_row(
            f"{cluster_id:02d}",
            str(len(clusters[cluster_id])),
            f"{avg_tempo:.1f} BPM",
            f"{avg_energy:.4f}"
        )

    # Final Results
    table = Table(title="🎵 Audio Clustering Results 🎵", title_style="bold cyan", border_style="bright_green")
    table.add_column("Cluster ID", justify="center", style="bold yellow")
    table.add_column("File Count", justify="center", style="bold green")
    table.add_column("Sample Files", style="bold magenta")
    for cluster_id, files in sorted(clusters.items()):
        table.add_row(
            f"{cluster_id:02d}",
            str(len(files)),
            "\n".join(files[:5]) + ("..." if len(files) > 5 else ""))

    console.print("\n", table)
    console.print("\n", cluster_summary)
    console.print(f"[bold green]Silhouette Score: {silhouette_avg:.3f} (higher is better)[/bold green]")
    console.print(f"[bold green]Success! Clustered audio files saved in: {OUTPUT_FOLDER}[/bold green]")

if __name__ == "__main__":
    try:
        process_audio_files()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user. Cleaning up...[/bold yellow]")
        for f in os.listdir(AUDIO_FOLDER):
            if f.endswith("_temp.wav"):
                os.remove(os.path.join(AUDIO_FOLDER, f))
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        sys.exit(1)