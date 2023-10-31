from __future__ import annotations

import datetime
import pathlib
import random
from typing import Iterable, Sequence

import cv2
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils


def plot_video_frame(
    video_path: str | pathlib.Path | cv2.VideoCapture, 
    frame_index: int | None = None,
) -> plt.Figure:
    v = utils.get_video_data(video_path)
    v.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = v.read()
    fig = plt.figure(facecolor="0.5")
    ax = fig.add_subplot()
    im = ax.imshow(frame, aspect="equal", cmap="Greys")
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.set_title(
        (
            f"frame {frame_index}/{utils.get_video_frame_count(v)}"
            " | "
            f"time {datetime.timedelta(seconds=frame_index / v.get(cv2.CAP_PROP_FPS))}"
                " / "
                f"{datetime.timedelta(seconds=utils.get_video_frame_count(v) / v.get(cv2.CAP_PROP_FPS))}"
            f" | {im.get_clim() = }"
        ), 
        fontsize=8,
    )
    return fig

def plot_video_frame_with_dlc_points(
    video_path: str | pathlib.Path | cv2.VideoCapture, 
    dlc_output_h5_path: str | pathlib.Path,
    frame_index: int | None = None,
    ) -> plt.Figure:
    """Single frame with eye, pupil and corneal reflection DLC points overlaid.
    """
    if frame_index is None:
        frame_index = random.randint(0, utils.get_video_frame_count(video_path))
    dlc_df = utils.get_dlc_df(dlc_output_h5_path)
    fig = plot_video_frame(video_path, frame_index)
    for body_part in utils.DLC_LABELS:
        xy = [utils.get_values_from_row(dlc_df.iloc[frame_index], annotation, body_part) for annotation in ('x', 'y')]
        fig.axes[0].plot(*xy, "+",  markersize=1, alpha=1)
    return fig
