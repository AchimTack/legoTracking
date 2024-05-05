import cv2
import numpy as np
from datetime import datetime
import os
import svgwrite


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