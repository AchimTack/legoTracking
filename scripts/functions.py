import cv2
import numpy as np
from datetime import datetime
import os
import svgwrite


def undistortField():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeStep = 2
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23

    matrix = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not captured.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

            if ids is not None:
                centers = {}

                for i, corner in enumerate(corners):
                    center = np.mean(corner[0], axis=0)
                    centers[ids[i][0]] = center
                    if ids[i][0] not in [91, 92, 93, 94]:
                        cv2.polylines(frame, [np.int32(corner)], True, (0, 255, 0), 2)
                        cv2.putText(frame, str(ids[i][0]), tuple(np.int32(corner[0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                
                if all(key in centers for key in [91, 92, 93, 94]) and matrix is None:
                    src_points = np.array([centers[92], centers[91], centers[93], centers[94]], dtype="float32")
                    aspect_ratio = 100 / 200
                    height = frame.shape[0]
                    width = int(height * aspect_ratio)
                    dst_points = np.array([
                        [width, 0],
                        [0, 0],
                        [width, height],
                        [0, height]
                    ], dtype="float32")

                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    print('Found Distortion Matrix:')
                    print(matrix)

            if matrix is not None:
                result = cv2.warpPerspective(frame, matrix, (width, height))
                cv2.imshow('Transformed', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def apply_perspective_transform(src_points, dst_size=(1280, 1024)):
    dst_points = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M


def save_tracking_results(transformed_frame, midpoints):
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_folder = f'runs/{timestamp}'
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)
            desaturated_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
            desaturated_frame[..., 1] = desaturated_frame[..., 1] * 0.2
            desaturated_frame = cv2.cvtColor(desaturated_frame, cv2.COLOR_HSV2BGR)
            for i in range(len(midpoints) - 1):
                cv2.line(desaturated_frame, tuple(midpoints[i]), tuple(midpoints[i+1]), (0, 0, 255), 2)
            cv2.imwrite(f'{run_folder}/{timestamp}.jpg', desaturated_frame)

            # Save SVG
            dwg = svgwrite.Drawing(f'{run_folder}/{timestamp}.svg', size=(1280, 1024))  # Adjusted to your dst_size

            # Add a grey rectangle as the background
            grey_background = dwg.rect(insert=(0, 0), size=(1280, 1024), fill='grey')
            dwg.add(grey_background)

            # Add lines between points
            for i in range(len(midpoints) - 1):
                start_point = (int(midpoints[i][0]), int(midpoints[i][1]))
                end_point = (int(midpoints[i+1][0]), int(midpoints[i+1][1]))
                dwg.add(dwg.line(start=start_point, end=end_point, stroke=svgwrite.rgb(255, 0, 0, '%')))

            dwg.save()