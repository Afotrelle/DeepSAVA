import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import DataSet


def test_get_frames_for_sample_accepts_avi_path(tmp_path):
    video_path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
    for _ in range(3):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames, name = DataSet.get_frames_for_sample('UCF101', ['test', 'Basketball', str(video_path), '3'])

    assert len(frames) == 3
    assert name == str(video_path)
    assert all(os.path.isfile(frame) for frame in frames)
