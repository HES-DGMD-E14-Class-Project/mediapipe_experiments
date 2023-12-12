# MediaPipe Experiments

Experiment with MediaPipe to extract and display landmarks for Hands and Pose in RealTime use a Video Feed (Webcam)

# Setting up your Environment

To run the code locally, you'll need to install and setup a few things:

* Python 3 (if you don't have a recent version of Python, [grab one here](https://www.python.org/downloads/).  We've tested on Python 3.10)
* Poetry (dependency manager for Python - [read the installation instructions here](https://python-poetry.org/docs/#installation))
* Git command line tools (the `git` command).  Get these from [the Git website](https://git-scm.com/downloads) if needed.

## Cloning this Repository

At the terminal, clone the repository to your local machine:

```bash
git clone https://github.com/HES-DGMD-E14-Class-Project/mediapipe_experiments.git
```

Then, change directory into the repository folder:

```bash
cd mediapipe_experiments
```

## Installing Python Dependencies

We're using the Poetry tool to manage Python virtual environments and dependencies.  Install the dependencies that this workshop uses with the following command:

```bash
poetry install
```

## Running the Experiments

### Pose Landmarker Demo

Based on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python

```bash
poetry run python PoseLandmarkerDemo.py
```

### Hand Landmarker Demo

Based on https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream

```bash
poetry run python HandLandmarkerDemo.py
```

### Hand Landmarker Finger Counting Demo

Reverse engineering of this Video https://www.youtube.com/watch?v=p5Z_GGRCI5s (this was bad code and for an ancient version of MediaPipe)

```bash
poetry run python HandLandmarkerFingerCounterDemo.py
```

### Hand Gesture Classification Demo

Reverse engineering of this Video https://www.youtube.com/watch?v=p5Z_GGRCI5s (this was bad code and for an ancient version of MediaPipe)

```bash
poetry run python HashGestureClassificationDemo.py
```

### Face Landmarker Demo

Reverse engineering of this Video https://www.youtube.com/watch?v=p5Z_GGRCI5s (this was bad code and for an ancient version of MediaPipe)

```bash
poetry run python FaceLandmarkerDemo.py
```

### LandmarksPlayer

A tkinter app to visualize the data set from https://www.kaggle.com/competitions/asl-signs/data - download the zip file
and unzip it somewhere in your computer, say `~/Desktop/asl-signs` then make a copy of `.env-examples`, rename it to `.env`
and change the environment property `ASL_SIGNS_BASE_DIRECTORY` to point to the `asl-signs` folder.

```bash
poetry run python LandmarksPlayer.py
```
### Random Forest Model
To run the MNIST detector real time using your own camera:
```bash
python3 RandomForestModelDemo.py
```




To quit any of the demos, press "q"
