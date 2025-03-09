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
from scipy.stats import skew
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
    █████╗ ██╗   ██╗██████╗ ██╗ ██████╗      ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗
    ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗    ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
    ███████║██║   ██║██║  ██║██║██║   ██║    ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝
    ██╔══██║██║   ██║██║  ██║██║██║   ██║    ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
    ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝    ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝      ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
    2025, Graham Waters
[/bold cyan]
"""

# Paths and constants
AUDIO_FOLDER = "/Volumes/BigBoy/portugal_movie/AUDIO_EXTRACTED"
OUTPUT_FOLDER = os.path.join(os.path.dirname(AUDIO_FOLDER), "AUDIO_CLUSTERS")
MAX_CLUSTERS_DEFAULT = 12
MIN_DURATION = 0.5
MIN_AMPLITUDE = 0.001
KEEP_ORIGINALS = False  # Set to True to copy instead of move, preserving originals

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
            raise ValueError("FFmpeg validation failed")

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
        # Check original file content
        if not validate_audio(file_path, check_content=True):
            return None
        if file_path.lower().endswith(".mp3"):
            wav_path = file_path.replace(".mp3", "_temp.wav")
            if not os.path.exists(wav_path):
                audio = AudioSegment.from_file(file_path, format="mp3")
                audio = audio.set_channels(1).set_frame_rate(22050)
                audio.export(wav_path, format="wav")
                console.print(f"[cyan]Converted {file_path} to {wav_path}[/cyan]")
                if not validate_audio(wav_path, check_content=True):
                    os.remove(wav_path)
                    return None
            return wav_path
        return file_path
    except Exception as e:
        console.print(f"[bold red]Failed to preprocess {file_path}:[/bold red] {e}")
        return None

# Advanced feature extraction with fallback
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < 512:
            console.print(f"[yellow]Skipping {file_path}:[/yellow] Too short")
            return None, None

        rms = np.mean(librosa.feature.rms(y=y))
        if rms < MIN_AMPLITUDE:
            console.print(f"[yellow]Skipping {file_path}:[/yellow] No audible content (RMS: {rms:.6f})")
            return None, None

        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        features = [rms, zero_crossing]

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)
            mfcc_skew = skew(mfcc, axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1) if np.any(y) else np.zeros(12)
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1) if np.any(y) else np.zeros(6)

            features = np.concatenate((
                mfcc_mean, mfcc_var, mfcc_skew, chroma,
                [spec_centroid, spec_flatness, spec_contrast, zero_crossing, rms, tempo, onset_env],
                tonnetz
            ))
        except Exception as e:
            console.print(f"[yellow]Using fallback features for {file_path}:[/yellow] {e}")

        return features, file_path
    except Exception as e:
        console.print(f"[bold red]Error extracting features from {file_path}:[/bold red] {e}")
        return None, None

# Find optimal number of clusters dynamically
def find_optimal_clusters(features_scaled, max_k=MAX_CLUSTERS_DEFAULT):
    if len(features_scaled) < 2:
        return 1
    max_k = min(int(np.sqrt(len(features_scaled))) + 1, max_k + 1, len(features_scaled))
    silhouette_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        silhouette_scores.append((k, score))
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 2
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
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(preprocess_audio, os.path.join(AUDIO_FOLDER, f)): f for f in file_names}
            for future in as_completed(futures):
                result = future.result()
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
        task2 = progress.add_task("[yellow]Extracting Advanced Features", total=len(processed_files))
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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

    features_scaled = StandardScaler().fit_transform(np.array(features))

    # Determine optimal number of clusters
    console.print("\n[bold magenta]Determining Optimal Cluster Count...[/bold magenta]")
    optimal_k = find_optimal_clusters(features_scaled)
    console.print(f"[bold green]Optimal number of clusters: {optimal_k}[/bold green]")

    # Clustering
    console.print("\n[bold magenta]Performing Enhanced K-Means Clustering...[/bold magenta]")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, labels) if optimal_k > 1 else 1.0

    # Organize clusters
    clusters = defaultdict(list)
    for file, label in zip(valid_files, labels):
        clusters[label].append(os.path.basename(file))

    # Move files with live display
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with Live(generate_cluster_layout(clusters), refresh_per_second=10, console=console) as live:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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

    feature_dim = len(features[0])
    tempo_idx = feature_dim - 7
    rms_idx = feature_dim - 8
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