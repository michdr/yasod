from typing import NamedTuple

from numpy import float32, int32
from numpy.core.multiarray import ndarray


class Detection(NamedTuple):
    class_id: int32
    confidence: float32
    box: ndarray


class Detections(NamedTuple):
    class_ids: ndarray
    confidences: ndarray
    boxes: ndarray
