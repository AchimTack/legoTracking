# LEGO FLL Robot Tracking System

This repository contains Python scripts for tracking LEGO FLL robots using a ceiling-mounted webcam. They utilize two different tracking methods: ArUco Markers and CSRT Tracking.

## General Instructions
Ensure the webcam is properly mounted overhead, looking down onto the table. Make sure to have consistent lighting and no over- or underexposure in webcam image. Webcam resolutions should exceed 1024*768px. Be aware code was in parts created using LLMs and is still in heavy need of refactoring.

## Dependencies
Both scripts require the following Python libraries:
- OpenCV
- NumPy
- svgwrite 

## LegoTrackerArucode.py
This script uses ArUco markers for tracking. The script includes functions for setting up the webcam, defining a region of interest by clicking on the video frame, and applying a perspective transformation to track the robot in a transformed top-down view.

Key Features:
- Interactive selection of tracking region on the video feed.
- Real-time perspective transformation for a top-down view.
- Tracking based on ArUco markers.

Usage
Run the script and interact with the webcam feed by clicking to define the region of interest (click 4 points clockwise). Finish by pressing "m" key. Tracking starts automatically. 
Ensure the region includes the ArUco markers. End the tracking by clicking "q". Results are saved into folder "runs".

## LegoTrackerCSRT.py
This script employs CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) for tracking an object selected on the video feed.

Key Features:
- Initialization of tracking with a user-defined bounding box.
- Real-time object tracking with display of tracking status and frame rate.
- Saving tracking results in JPEG and SVG formats.

Usage
Run the script and interact with the webcam feed by clicking to define the object of interest by drawing a rectangle over it. Finish by pressing enter key. Tracking starts automatically. 
Ensure the region includes the ArUco markers. End the tracking by clicking "q". Results are saved into folder "runs".


