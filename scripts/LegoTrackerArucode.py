from functions import undistort_and_track

# Set mat dimensions
matLength = 236
matWidth = 114

# Define which marker range is to be tracked (minimizes false-positives)
marker_ids_to_track = set(range(1, 10))

undistort_and_track(matLength, matWidth, marker_ids_to_track)
