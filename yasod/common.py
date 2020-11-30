from typing import List, NamedTuple

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


def flatten_detections(class_ids, confidences, boxes) -> List[Detection]:
    return [
        Detection(*d)
        for d in zip(
            class_ids.flatten(),
            confidences.flatten(),
            boxes,
        )
    ]
