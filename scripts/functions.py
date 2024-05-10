import cv2
import numpy as np
from datetime import datetime
import os
import svgwrite
import csv


def generate_color(marker_id):
    predefined_colors = [
        (0, 0, 255),   # Red
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
        (0, 165, 255), # Orange
        (147, 20, 255),# Deep Pink
        (0, 128, 0),   # Dark Green
        (130, 0, 75)   # Indigo
    ]
    return predefined_colors[(marker_id - 1) % len(predefined_colors)]


def save_tracking_results(transformed_frame, tracks):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_folder = f'runs/{timestamp}'
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Create a desaturated version of the transformed frame
    desaturated_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    desaturated_frame = desaturated_frame.astype(float)  # Convert to float for modification
    desaturated_frame[..., 1] *= 0.2  # Reduce saturation
    desaturated_frame = desaturated_frame.astype('uint8')  # Convert back to uint8
    desaturated_frame = cv2.cvtColor(desaturated_frame, cv2.COLOR_HSV2BGR)

    # Draw tracking lines on the desaturated frame
    for color, track in tracks.items():
        for i in range(len(track) - 1):
            start_point = tuple(map(int, track[i]))
            end_point = tuple(map(int, track[i + 1]))
            cv2.line(desaturated_frame, start_point, end_point, color, 2)

    # Rotate the desaturated frame
    save_frame = cv2.rotate(desaturated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f'{run_folder}/{timestamp}.jpg', save_frame)

    # Calculate the aspect ratio and define SVG dimensions
    frame_height, frame_width = transformed_frame.shape[:2]
    aspect_ratio = frame_height / frame_width
    svg_width = 1000
    svg_height = int(svg_width * aspect_ratio)

    # Create an SVG drawing with rotated dimensions
    dwg = svgwrite.Drawing(
        filename=f'{run_folder}/{timestamp}.svg',
        size=(svg_height, svg_width),
        viewBox=f'0 0 {svg_height} {svg_width}'
    )

    # Add a grey rectangle as the background
    grey_background = dwg.rect(insert=(0, 0), size=(svg_height, svg_width), fill='grey')
    dwg.add(grey_background)

    # Draw the tracks in the rotated SVG
    for color, track in tracks.items():
        for i in range(len(track) - 1):
            start_point = (int(track[i][1] * svg_height / frame_height), int(svg_width - (track[i][0] * svg_width / frame_width)))
            end_point = (int(track[i + 1][1] * svg_height / frame_height), int(svg_width - (track[i + 1][0] * svg_width / frame_width)))
            dwg.add(dwg.line(start=start_point, end=end_point, stroke=svgwrite.rgb(*color, '%')))

    # Save the rotated SVG
    dwg.save()


    # Save tracking data to a CSV file
    csv_file_path = f'{run_folder}/{timestamp}.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'X', 'Y'])
        for color, track in tracks.items():
            for i, point in enumerate(track):
                csv_writer.writerow([i, point[0], point[1]])


def transform_points(points, matrix):
    if not points or matrix is None:
        return []
    points = np.array(points, dtype='float32').reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, matrix)
    return [tuple(map(int, point[0])) for point in transformed_points]


def is_point_inside_mask(point, mask_contour):
    return cv2.pointPolygonTest(mask_contour, point, False) >= 0


def detect_and_track_markers(frame, dictionary, parameters, tracking_colors, tracking_points):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    centers = {}
    orientations = {}
    excluded_ids = {91, 92, 93, 94}
    if ids is not None:
        for i, corner in enumerate(corners):
            # Calculate the center of the marker
            center = np.mean(corner[0], axis=0)
            marker_id = ids[i][0]
            centers[marker_id] = center

            # Process only if marker ID is not in excluded range
            if marker_id not in excluded_ids:
                # Calculate the orientation of the marker
                vector = corner[0][1] - corner[0][0]
                angle = int(np.arctan2(vector[1], vector[0]) * 180 / np.pi)
                if angle < 0:
                    angle += 360
                orientations[marker_id] = angle

                # Assign tracking color if not already set
                if marker_id not in tracking_colors:
                    tracking_colors[marker_id] = generate_color(marker_id)
                if marker_id not in tracking_points:
                    tracking_points[marker_id] = []
                tracking_points[marker_id].append(center)

                # Draw the marker outline, marker ID, and orientation
                cv2.polylines(frame, [np.int32(corner)], True, tracking_colors[marker_id], 1)
                cv2.putText(frame, str(marker_id), tuple(np.int32(corner[0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracking_colors[marker_id], 1, cv2.LINE_AA)
                cv2.putText(frame, str(angle), tuple(np.int32(center)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracking_colors[marker_id], 1, cv2.LINE_AA)

    return centers, orientations


def get_perspective_transform_matrix(centers, frame_shape):
    aspect_ratio = 114 / 205
    height = frame_shape[0]
    width = int(height * aspect_ratio)

    src_points = np.array([centers[92], centers[91], centers[93], centers[94]], dtype="float32")
    dst_points = np.array([
        [width, 0],
        [0, 0],
        [width, height],
        [0, height]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    mask_contour = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='int32')

    return matrix, mask_contour, width, height


def draw_transformed_tracks(result, transformed_tracks):
    for color, track in transformed_tracks.items():
        for i in range(len(track) - 1):
            cv2.line(result, track[i], track[i + 1], color, 1)
