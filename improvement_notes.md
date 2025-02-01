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