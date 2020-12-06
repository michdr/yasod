![CI Status](https://github.com/michdr/yasod/workflows/CI/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/yasod)](https://pypi.org/project/yasod)

# yasod
<!--- Don't edit the version line below manually. Let bump2version do it for you. -->
> Version 0.1.1 
>
> Yet another simple object detector
>
The aim of this library is to provide a very simple functionality of object detection (currently only for images) given a pre-configured model.
As its main battery it depends heavily on `opencv-python`, alongside a few other relatively slim dependencies. No further dependencies are required.
The main advantage of this is to be able achieve that simple goal without much boilerplate.

As for a detection model, it accepts any topology supported by the underlying `cv2.dnn_Model` class and its implementation.

## Installing
```bash
pip install yasod
``` 

## Getting started
An example for a config and models could be found in `tests/data`. 

Here is a simple example how to detect the objects of a given `input-image.jpg` and draw an output image accordingly:
```python
from yasod import Yasod

model = Yasod("simple-yasod-config.yml").get_model("yolov4-tiny")
img, detections = model.detect("input-image.jpg")
model.draw_results(img, detections, "output-image.jpg")
``` 
