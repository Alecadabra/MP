# Building Number Finder

This is a Python OpenCV program to find and classify Curtin University building
numbers.

See [the written report](Report.pdf) for discussion on the approach and
performance.

## Environment

This program runs in Python 3.8.2, using opencv-python 3.4.8.

## Run

There are two separate tasks that this assignment runs.

### Building Number (Task 1)

This is for detecting the area and classifying a single building number.

To provide the 10 input images from the [`val/task1/`](val/task1) directory,
with output put into the directory `output/task1/`, use the following.

```sh
python3 src/assignment.py task1 -t val -n 10 -o output
```

## Directional Signage (Task 2)

This is for detecting the area and classifying a sign with mutiple building
numbers and arrows.

To provide the 10 input images from the [`val/task2/`](val/task2) directory,
with output put into the directory `output/task2/`, use the following.

```sh
python3 src/assignment.py task2 -t val -n 10 -o output
```
