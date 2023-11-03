import contextlib
import json
import os
import pathlib
import random

os.environ["DLClight"]="True" # set before importing DLC
import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
import qc

REUSE_DLC_OUTPUT_H5_IN_ASSET = True
"""Instead of re-generating DLC h5 file, use one in a data asset - for quickly testing
qc plotting"""

def main():
    """ find video paths, launch DLC analysis.. """

    # process first matching video found
    input_video_file_path: pathlib.Path = next(
        utils.get_video_paths(), None
        )
    if input_video_file_path is None:
        raise FileNotFoundError("No video files found matching {utils.VIDEO_FILE_GLOB_PATTERN=}, {utils.VIDEO_SUFFIXES=}")
    print(f"Reading video: {input_video_file_path}")
    
    # phase 1: track points in video and generate h5 file ------------------------- #

    if REUSE_DLC_OUTPUT_H5_IN_ASSET:
        # get existing h5 file from data/ 
        temp_files = set()
        with contextlib.suppress(FileNotFoundError):
            existing_h5 = utils.get_dlc_output_h5_path(
                input_video_file_path=input_video_file_path,
                output_dir_path=utils.DATA_PATH,
            ) 
            print(f"{REUSE_DLC_OUTPUT_H5_IN_ASSET=}: using {existing_h5}")
            # - a pickle file exists too
            # - copy everything with matching filename component to results/
            for file in existing_h5.parent.glob(f"{existing_h5.stem}*"):
                temp_files.add(dest := utils.RESULTS_PATH / file.name)
                if not dest.exists(): # during testing we may have already made this 
                    dest.symlink_to(file)
        # no need to skip DLC - it will see the existing h5 and skip itself
 
    print(f"Running DLC analysis and writing to: {utils.RESULTS_PATH}")
    deeplabcut.analyze_videos(
        config=utils.DLC_PROJECT_PATH / 'config.yaml',
        videos=[
          input_video_file_path.as_posix(),
        ],
        gputouse=0,
        destfolder=utils.RESULTS_PATH.as_posix(),
    )
    dlc_output_h5_path = utils.get_dlc_output_h5_path(
                input_video_file_path=input_video_file_path,
                output_dir_path=utils.RESULTS_PATH,
        )
    # qc plots -------------------------------------------------------------------- #

    utils.QC_PATH.mkdir(exist_ok=True, parents=True)

    # example frames with points overlaid
    NUM_FRAMES = 5
    print(f"Writing {NUM_FRAMES} example frames to {utils.QC_PATH}")
    total_frames = utils.get_video_frame_count(input_video_file_path)
    step = total_frames // NUM_FRAMES + 1
    for idx in range(step//2, total_frames, step): # avoid frames at the very start/end
        qc.plot_video_frame_with_dlc_points(
            video_path=input_video_file_path,
            frame_index=idx,
            dlc_output_h5_path=dlc_output_h5_path,
        ).savefig(
            utils.QC_PATH / f"{input_video_file_path.stem}_{idx}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    
if __name__ == "__main__": 
    main()
