from functions import undistort_and_track

# Set mat dimensions
matLength = 236
matWidth = 114

# Set Webcam Parameters
cam_id = 0
frame_width = 1920
frame_height = 1080

# Define which marker range is to be tracked (minimizes false-positives)
marker_ids_to_track = set(range(1, 10))
edge_marker_ids = [91, 92, 93, 94]

# Define jpg and svg output parameters
img_output_width = 1000

undistort_and_track(matLength, matWidth, marker_ids_to_track, edge_marker_ids, cam_id, frame_width, frame_height, img_output_width)
