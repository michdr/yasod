import os

import pytest

from yasod import Yasod, YasodModel


@pytest.fixture(scope="module", autouse=True)
def initialize():
    os.chdir("tests")


@pytest.fixture
def yasod_simple_config():
    return Yasod("data/simple-yasod-config.yml")


def test_simple_config(yasod_simple_config):
    models = yasod_simple_config.models
    assert len(models) == 1


def test_simple_detect_and_draw(yasod_simple_config):
    model = yasod_simple_config.get_default_model()
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
        for detection in YasodModel.flatten_detections(detections):
            label = model.label_detection(detection)
            any(exp_cls in label for exp_cls in expected_results["expected_classes"])
        assert out_img_file
