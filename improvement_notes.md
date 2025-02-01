# Processing DJI_0962.MP4:  23%|███████████████████████████████████████████████▏(0.5789)                                                                                                                                 | 1027/4374 [00:52<02:54, 19.20it/s]
# Processing DJI_0931.MP4:  18%|████████████████████████████████████▏(0.957)                                                                                                                                             | 1458/8105 [01:09<05:16, 20.98it/s]
# Processing DJI_0959.MP4:  30%|████████████████████████████████████████████████████████████▏(0.879)                                                                                                                     | 1352/4514 [01:09<02:47, 18.86it/s]
# Processing DJI_0961.MP4:  25%|█████████████████████████████████████████████████▎(0.4333) ...


The lines above show the desired output in the command line.
The numbers at the end of the progress bar should show the current standard deviation of that batch.
I assume this is the correct way of determining how close we are to finding what batch the whole video file belongs to.
Check my logic on that though.
Make sure early termination due to the 0.05 stdev is not causing this warning: 025-01-31 18:15:16,028 - WARNING - No valid frames extracted from /Volumes/BigBoy/portugal trip 2025/1-13-2025/11 AM/DJI_0919.MP4

It would be nice to have a live updating batch dashboard.
| batch number | images | videos | stdev |
| 1 | 13 | 3 | 0.031 |

this is an example of what I was thinking of. Is it possible to run that in a separate terminal window (different from the other progress bars) so both show live progress?

Add a check at the beginning of the code to survey what has already been processed in the output folder and remove the files already processed from the files in queue.
If any files trigger warnings then add them to an error logfile which you create in the main directory.

Add Additional metrics (e.g., model scoring, certainty scores, error ranges) can be computed and added here.


1. add `skip_errors` parameter to indicate user wants to read error.log before tasking, and remove the errored file paths from the queue of files to process.
2. add colorama support to the CLI. each progress bar (alive progress would be good here) should be color coded by the file's extension.
3. add a config file for the project containing essential variables such as:
```py
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}
RAW_EXTENSIONS   = {".raw", ".cr2", ".nef", ".dng", ".arw", ".rw2"}

process_large_files = False  # Updated via CLI

DEFAULT_CONFIG = {
    'clustering': {
        'algorithm': 'hdbscan',  # 'dbscan' or 'hdbscan'
        'eps': 0.15,
        'min_samples': 3, # the smallest number of items per cluster
        'visual_weight': 0.7, # how much the model emphasizes visual similarity over temporal similarity.
        'temporal_weight': 0.3 # how much the model emphasizes temporal similarity over visual similarity.
    },
    'processing': { #todo add more documentation and explanation to these parameters as comments.
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
    'video_certainty_threshold': None  # e.g. 0.05 (set via CLI)
}

```