# LEGO Tracker with ArUco Markers

This repository contains the code for a LEGO tracking system that utilizes ArUco markers (either to be brick-built or printed as stickers) to detect and visualize LEGO Robots (WRO / FLL) on a tracking mat using computer vision techniques. Be aware code was in parts created using LLMs and is still in heavy need of refactoring.

## Description

The project comprises two main scripts:
- `functions.py`: Includes helper functions to handle marker detection, image manipulation, and export functionalities.
- `LegoTrackerArucode.py`: Implements the tracking logic using the functions defined in `functions.py`. This script sets up the camera parameters, defines the tracking environment, and processes the video stream to identify and record the positions of LEGO pieces based on ArUco markers.

## Features

- **Marker Detection**: Utilizes OpenCV to detect pre-defined ArUco markers.
- **Visualization**: Generates visual outputs in various formats including JPEG, SVG, and CSV files for data logging.
- **Customizable Tracking**: Users can specify which markers to track, reducing false positives and focusing on relevant pieces.

## Setup

### Dependencies:  
   Ensure you have Python installed along with the following packages:
   - `opencv-contrib-python`
   - `numpy`
   - `svgwrite`

   Install them using pip:
   ```bash
   pip install opencv-contrib-python numpy svgwrite
   ```

### Camera Setup:  
   Connect a camera and ensure it is configured as per the requirements specified in `LegoTrackerArucode.py`. Adjust the `cam_id`, `frame_width`, and `frame_height` parameters as needed.


## Prepare Tracking
- Ensure the webcam is properly mounted overhead, looking down onto the table. Make sure to have consistent lighting and no over- or underexposure in webcam image.
- Webcam resolutions should exceed 1024*768px.
- The script requires 4 ArUco markers with distinct IDs (example 91-94) to be positioned at the four corners of the field. Print them from https://chev.me/arucogen/ and place them accordingly before initiating the script.
- Modify `matLength` and `matWidth` in `LegoTrackerArucode.py` to match the dimensions of your tracking mat.
- Update `marker_ids_to_track` to change the set of ArUco markers that should be detected.


## Tracking
Run the `LegoTrackerArucode.py` script to start the tracking system:
```bash
python LegoTrackerArucode.py
```
- Wait for webcam to initialize, tracking starts automatically when all 4 corner markers have been detected.
- Reset / restart tracking using space key.
- Save the results of the current track by pressing 's'.
- End the tracking by clicking "q". 

## Output

Results are saved into folder "runs" (depending which export type is activated in `LegoTrackerArucode.py`)

The system can export:
- JPEG images of the tracked scene.
- SVG files for graphical representation.
- CSV files logging the detected markers and their positions.
- MP4 video file capturing the tracking process.
