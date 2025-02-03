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


1. Any time the code is terminated, the progress of the script should be resumable. This means checkpoint saves along the way.
2. When the code reaches the stage where files are being moved implement the following changes.
   1. hyperoptimize the way the files are traversed, read and written, and choose the fastest way to safely move the files to the destination.
   2. If the file's "created date" is AFTER the "date modified" then change the created date to the date modified as you move it to the destination if the user has supplied the CLI argument --fix-dates
   3. Once you have sorted the files into the scenes based on the initial parameters for epsilon, I want your code to iteratively do the following.
      1. check the files in the 'noise' folder, you will perform the same analysis as you did last time for all the files in the input folder, however, this time you must sort them into the same scene folders (if you can) that you already created. Else, you will create a new scene. For each subsequent iteration and rescan of the noise folder modify epsilon to be less strict by 0.05. Repeat this until all noise images are sorted into the folders. NOTE: you will be using the same destination directory as you did for iteration 1.
         1. Each iteration should be colored in the CLI with a new color to indicate which round it is on.
   4. Note: if the code is executed, and the destination directory already has scene folders, then check the noise folder. IF files are in the noise folder then the user wants you to begin with those files and process them into the scenes as discussed above. ELSE if there is nothing in the noise folder, then look for files in the `input` directory to sort into the existing scenes by similarity. Otherwise, sort the new photos from the input direcory into the scenes already present in the destination directory first, adding new scenes as needed when you find new clusters.
   5. Note that the instance above should only apply if there are folders with files in the destination directory. Else proceed as you would normally.