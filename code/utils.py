from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import functools
import itertools
import pathlib
import pickle
from typing import (Dict, Iterable, Iterator, Literal, Mapping, NamedTuple,
                    Sequence, Tuple)

import cv2
import numpy as np
import pandas as pd
import tqdm
from typing_extensions import TypeAlias

DATA_PATH = pathlib.Path('/data/')
RESULTS_PATH = pathlib.Path('/results/')
QC_PATH = RESULTS_PATH / "qc" 

DLC_PROJECT_PATH = DATA_PATH / 'np3_side_cam-sam_corbett-2020-03-31'
DLC_SCORER_NAME = 'np3_side_camMar31-trainset95shuffle1'

VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

BodyPart: TypeAlias = Literal[
    "eye_top_l",
    "eye_bottom_l",
    "pupil_left_l",
    "pupil_right_l",
    "pupil_top_l",
    "pupil_bottom_l",
    "whisker_pad_l",
    "tube_bottom",
    "tongue_end_l",
    "tube_top",
    "tongue_end_r",
    "chin_l",
    "pinky_l",
    "ring_finger_l",
    "middle_finger_l",
    "pointer_finger_l",
    "pinky_lh",
    "ring_finger_lh",
    "middle_finger_lh",
    "pointer_finger_lh",
    "ear_tip_l",
    "whisker_pad_r",
    "chin_r",
    "pinky_r",
    "ring_finger_r",
    "middle_finger_r",
    "pointer_finger_r",
    "pinky_rh",
    "ring_finger_rh",
    "middle_finger_rh",
    "pointer_finger_rh",
    "ear_tip_r",
    "nostril_l",
    "nostril_r",
    "nose_tip",
    "tail_tip",
]
Annotation: TypeAlias = Literal['x', 'y', 'likelihood']
AnnotationData: TypeAlias = Dict[Tuple[BodyPart, Annotation], float]

ANNOTATION_PROPERTIES: tuple[Annotation, ...] = ('x', 'y', 'likelihood')
DLC_LABELS = tuple(BodyPart.__args__)

VIDEO_FILE_GLOB_PATTERNS = ('*[sS]ide*', '*[bB]ehavior*')

def get_video_paths() -> Iterator[pathlib.Path]:
    yield from (
        p for p in itertools.chain(
            *[
                DATA_PATH.rglob(VIDEO_FILE_GLOB_PATTERN)
                for VIDEO_FILE_GLOB_PATTERN in VIDEO_FILE_GLOB_PATTERNS
            ]
        )
        if (
            DLC_PROJECT_PATH not in p.parents
            and p.suffix in VIDEO_SUFFIXES
        )
    )

def get_dlc_pickle_metadata(dlc_output_h5_path: str | pathlib.Path) -> dict:
    h5 = pathlib.Path(dlc_output_h5_path)
    pkl = (
        h5
        .with_stem(f'{h5.stem}_meta')
        .with_suffix('.pickle')
    )
    return pickle.loads(pkl.read_bytes())['data']

def get_dlc_df(dlc_output_h5_path: str | pathlib.Path) -> pd.DataFrame:
    # df has MultiIndex 
    return getattr(pd.read_hdf(dlc_output_h5_path), get_dlc_pickle_metadata(dlc_output_h5_path)['Scorer']) 

def get_dlc_output_h5_path(
    input_video_file_path: str | pathlib.Path, 
    output_dir_path: str | pathlib.Path = RESULTS_PATH,
) -> pathlib.Path:
    output = next(
        pathlib.Path(output_dir_path)
        .rglob(
            glob := f"{pathlib.Path(input_video_file_path).stem}*.h5"
        ),
        None
    )
    if output is None:
        raise FileNotFoundError(f"No file matching {glob} in {output_dir_path}")
    return output


def get_video_data(video_path: str | pathlib.Path | cv2.VideoCapture) -> cv2.VideoCapture:
    """Open video file as cv2.VideoCapture object."""
    if isinstance(video_path, cv2.VideoCapture):
        return video_path
    return cv2.VideoCapture(str(video_path))

def get_video_frame_count(video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> int:
    return int(get_video_data(video_path_or_data).get(cv2.CAP_PROP_FRAME_COUNT))

@functools.cache
def get_video_frame_size_xy(video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> Tuple[int, int]:
    return get_video_data(video_path_or_data).read()[1][:,:,0].shape.T

def is_in_frame(x: float, y: float, video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> bool:
    """Check if point is inside video frame."""
    w, h = get_video_frame_size_xy(video_path_or_data)
    return (0 <= x < w) and (0 <= y < h)

def get_values_from_row(row: AnnotationData, annotation: Annotation, body_part: BodyPart) -> np.array:
    return np.array([v for k, v in row.items() if k[1] == annotation and body_part in k[0]])
