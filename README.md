![CI Status](https://github.com/michdr/yasod/workflows/CI/badge.svg)

# yasod
> Yet another simple object detector

## Installing
```bash
pip install yasod
``` 

## Getting started
An example for a config and models could be found in `tests/data`. 

Here is a simple example how to detect the objects of a given `input-image.jpg` and draw an output image accordingly:
```python
from yasod import Yasod

model = Yasod("simple_yasod_config.yml").get_model("yolov4-tiny")
img, detections = model.detect("input-image.jpg")
model.draw_results(img, detections, "output-image.jpg")
``` 
