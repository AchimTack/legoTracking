import cv2
import numpy as np
from datetime import datetime
import os
import svgwrite
import csv
import time


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


def save_tracking_results(transformed_frame, all_transformed_data):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_folder = f'runs/{timestamp}'
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Rotate the original frame first
    rotated_frame = cv2.rotate(transformed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Create a desaturated version of the rotated frame
    desaturated_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2HSV)
    desaturated_frame = desaturated_frame.astype(float)  # Convert to float for modification
    desaturated_frame[..., 1] *= 0.2  # Reduce saturation
    desaturated_frame = desaturated_frame.astype('uint8')  # Convert back to uint8
    desaturated_frame = cv2.cvtColor(desaturated_frame, cv2.COLOR_HSV2BGR)

    # Create a copy to draw colorful tracks
    colorful_frame = desaturated_frame.copy()

    # Draw the tracks
    tracks = {}
    for frame, marker_id, x, y, _ in all_transformed_data:
        if marker_id not in tracks:
            tracks[marker_id] = []
        # Adjust coordinates due to the rotation (90-degree counterclockwise)
        adjusted_x = y  # New x after rotation
        adjusted_y = rotated_frame.shape[0] - x  # New y after rotation
        tracks[marker_id].append((adjusted_x, adjusted_y))

    # Draw the tracks on the colorful frame
    for marker_id, track in tracks.items():
        color = generate_color(marker_id)
        for i in range(len(track) - 1):
            start_point = tuple(map(int, track[i]))
            end_point = tuple(map(int, track[i + 1]))
            cv2.line(colorful_frame, start_point, end_point, color, 2)

    # Save the final combined image (desaturated background with colorful tracks)
    cv2.imwrite(f'{run_folder}/{timestamp}.jpg', colorful_frame)

    # Calculate the aspect ratio and define SVG dimensions
    frame_height, frame_width = rotated_frame.shape[:2]
    aspect_ratio = frame_height / frame_width
    svg_width = 1000
    svg_height = int(svg_width * aspect_ratio)

    # Create an SVG drawing with rotated dimensions
    dwg = svgwrite.Drawing(
        filename=f'{run_folder}/{timestamp}.svg',
        size=(svg_width, svg_height),
        viewBox=f'0 0 {svg_width} {svg_height}'
    )

    # Add a grey rectangle as the background
    grey_background = dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill='grey')
    dwg.add(grey_background)

    # Draw the tracks in the rotated SVG
    for marker_id, track in tracks.items():
        color = generate_color(marker_id)
        color_str = f'rgb({color[2]},{color[1]},{color[0]})'
        for i in range(len(track) - 1):
            start_point = (
                int(track[i][0] * svg_width / frame_width),
                int(track[i][1] * svg_height / frame_height)
            )
            end_point = (
                int(track[i + 1][0] * svg_width / frame_width),
                int(track[i + 1][1] * svg_height / frame_height)
            )
            dwg.add(dwg.line(start=start_point, end=end_point, stroke=color_str, stroke_width=2))

    dwg.save()

    # Save tracking data to a CSV file
    csv_file_path = f'{run_folder}/{timestamp}.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Marker', 'X', 'Y', 'Orientation'])
        csv_writer.writerows(all_transformed_data)
        

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


def detect_and_track_markers(frame, tracking_colors, tracking_points):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.__version__.startswith('3'):  # For OpenCV 3.x
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeStep = 2
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    else:  # For OpenCV 4.x and above
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

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
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Calculate text position to the right of the marker
                text_position_id = tuple(np.int32(corner[0][0]) + np.array([10, 0]))
                text_position_angle = tuple(np.int32(center) + np.array([10, 15]))

                # Outline color and text color
                outline_color = (255, 255, 255)  # White outline
                text_color = (0, 0, 0)  # Black text

                # Draw text with outline
                cv2.putText(frame, str(marker_id), text_position_id, font, font_scale, outline_color, thickness+2, cv2.LINE_AA)
                cv2.putText(frame, str(marker_id), text_position_id, font, font_scale, text_color, thickness, cv2.LINE_AA)

                cv2.putText(frame, str(angle), text_position_angle, font, font_scale, outline_color, thickness+2, cv2.LINE_AA)
                cv2.putText(frame, str(angle), text_position_angle, font, font_scale, text_color, thickness, cv2.LINE_AA)

                cv2.polylines(frame, [np.int32(corner)], True, tracking_colors[marker_id], 1)
                
    return centers, orientations


def get_perspective_transform_matrix(matLength, matWidth, centers, frame_shape):
    aspect_ratio = matWidth / matLength
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


def draw_transformed_tracks(result, all_transformed_data, tracking_colors):
    tracks = {}
    for frame, marker_id, x, y, _ in all_transformed_data:
        if marker_id not in tracks:
            tracks[marker_id] = []
        tracks[marker_id].append((x, y))

    for marker_id, track in tracks.items():
        color = tracking_colors.get(marker_id, (0, 255, 0))  # Default to green if not found
        for i in range(len(track) - 1):
            cv2.line(result, tuple(map(int, track[i])), tuple(map(int, track[i + 1])), color, 1)


def undistort_and_track(matLength, matWidth, marker_ids_to_track):
    print('starting...')
    cap = cv2.VideoCapture(0)
    frame_width = 1920
    frame_height = 1080

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    matrix = None
    mask_contour = None
    tracking_colors = {}
    tracking_points = {}
    width, height = 0, 0
    result = None
    all_transformed_data = []

    try:
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not captured.")
                continue

            centers, orientations = detect_and_track_markers(frame, tracking_colors, tracking_points)

            # Ensure perspective matrix is set using markers 91-94
            if all(key in centers for key in [91, 92, 93, 94]) and matrix is None:
                matrix, mask_contour, width, height = get_perspective_transform_matrix(matLength, matWidth, centers, frame.shape)
                print('Found Distortion Matrix:')
                print(matrix)

            if matrix is not None:
                result = cv2.warpPerspective(frame, matrix, (width, height))

                for marker_id in marker_ids_to_track:
                    if marker_id in tracking_points:
                        transformed_points = transform_points(tracking_points[marker_id], matrix)
                        filtered_points = [point for point in transformed_points if is_point_inside_mask(point, mask_contour)]
                        if filtered_points:
                            color = tracking_colors[marker_id]
                            orientation = orientations.get(marker_id, '')
                            all_transformed_data.append([frame_counter, marker_id, filtered_points[-1][0], filtered_points[-1][1], orientation])

                draw_transformed_tracks(result, all_transformed_data, tracking_colors)
 
                rotated_frame = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow('Transformed', rotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                frame_counter = 0
                result = None
                all_transformed_data = []

            if key == ord('s'):
                if result is not None:
                    save_tracking_results(result, all_transformed_data)
                frame_counter = 0
                result = None
                all_transformed_data = []

            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            frame_counter += 1

    finally:
        cap.release()