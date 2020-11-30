import os

import pytest
from numpy import float32, int32
from numpy.ma import array

from yasod import Yasod
from yasod.common import Detections, flatten_detections


@pytest.fixture(scope="session", autouse=True)
def initialize():
    os.chdir("tests")


@pytest.fixture
def yasod_simple_config():
    return Yasod("data/simple-yasod-config.yml")


@pytest.fixture
def detections_example():
    return Detections(
        class_ids=array(
            ([[0]] * 8) + ([[30]] * 6) + ([[11]] * 4),
            dtype=int32,
        ),
        confidences=array(
            ([[0.92]] * 6) + ([[0.56]] * 10) + ([[0.45]] * 2),
            dtype=float32,
        ),
        boxes=array(
            [
                [1377, 308, 75, 170],
                [132, 354, 94, 228],
                [213, 330, 70, 145],
                [1025, 329, 69, 203],
                [1148, 335, 91, 191],
                [8, 318, 47, 89],
                [409, 323, 51, 101],
                [974, 337, 69, 188],
                [79, 324, 52, 109],
                [749, 324, 42, 99],
                [268, 697, 299, 208],
                [576, 373, 65, 302],
                [727, 370, 95, 304],
                [634, 380, 58, 292],
                [528, 368, 66, 317],
                [1347, 457, 152, 29],
                [120, 696, 256, 218],
                [687, 356, 84, 322],
                [789, 365, 114, 305],
            ],
            dtype=int32,
        ),
    )


@pytest.fixture
def default_model(yasod_simple_config):
    return yasod_simple_config.get_default_model()


def test_simple_config(yasod_simple_config):
    models = yasod_simple_config.models
    assert len(models) == 1


def test_simple_detect_and_draw(default_model):
    model = default_model
    sample_images = {
        "data/input/cars.jpg": dict(expected_classes=["car"]),
        "data/input/people.jpg": dict(expected_classes=["person"]),
        "data/input/skis-and-people.jpg": dict(expected_classes=["person", "skis"]),
    }
    for in_img_file, expected_results in sample_images.items():
        img, detections = model.detect(in_img_file)
        out_img_file = in_img_file.replace("input", "output")
        model.draw_results(img, detections, out_img_file)
        assert len(detections)
        for detection in flatten_detections(detections):
            label = model.label_detection(detection)
            any(exp_cls in label for exp_cls in expected_results["expected_classes"])
        assert out_img_file


def test_get_object_detections_class_ids_counts(default_model, detections_example):
    model = default_model
    counts = model.get_object_detections_class_ids_counts(detections_example)
    assert counts == {0: 8, 11: 4, 30: 6}


def test_get_object_detections_class_names_counts(default_model, detections_example):
    model = default_model
    counts = model.get_object_detections_class_names_counts(detections_example)
    assert counts == {"person": 8, "stop sign": 4, "skis": 6}
