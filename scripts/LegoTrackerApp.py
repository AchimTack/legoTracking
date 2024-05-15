import tkinter as tk
import cv2
from PIL import Image, ImageTk
import yaml
import numpy as np
from datetime import datetime
import os
import svgwrite
import csv


class LegoTrackerApp:
    def __init__(self, master, config):
        self.master = master
        self.config = config
        master.title("Lego Tracker")

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeStep = 2
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23

        self.matrix = None
        self.mask_contour = None
        self.tracking_colors = {}
        self.tracking_points = {}
        self.width, self.height = 0, 0
        self.all_transformed_data = []
        self.frame_counter = 0
        self.transformed_frame = None  # Initialize transformed_frame

        self.create_widgets()

        # Webcam Selection
        self.webcam_index = tk.IntVar()
        self.create_webcam_selection()

        self.cap = None
        self.start_tracking = False

    def create_widgets(self):
        self.start_button = tk.Button(self.master, text="Start", command=self.start_tracking)
        self.start_button.pack()

        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop_tracking)
        self.stop_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save_results)
        self.save_button.pack()

        self.canvas = tk.Canvas(self.master, width=self.config['camera']['resolution'][0],
                                      height=self.config['camera']['resolution'][1])
        self.canvas.pack()

        self.data_label = tk.Label(self.master, text="Marker Data: ")
        self.data_label.pack()

    def create_webcam_selection(self):
        # Check available webcams
        index = 0
        working_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                working_cameras.append(index)
                cap.release()
                index += 1
            else:
                break

        # Create radio buttons for each available webcam
        for i in working_cameras:
            tk.Radiobutton(self.master, text=f"Webcam {i}", variable=self.webcam_index, value=i).pack()

        # Set default webcam
        if working_cameras:
            self.webcam_index.set(working_cameras[0])  # Default to the first webcam

    def start_tracking(self):
        self.cap = cv2.VideoCapture(self.webcam_index.get())
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution'][1])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        self.start_tracking = True
        self.frame_counter = 0
        self.all_transformed_data = []
        self.matrix = None
        self.update()

    def stop_tracking(self):
        self.start_tracking = False
        if self.cap:
            self.cap.release()

    def save_results(self):
        if self.transformed_frame is not None:
            self.save_tracking_results(self.transformed_frame, self.all_transformed_data)

    def update(self):
        if self.start_tracking:
            ret, frame = self.cap.read()
            if ret:
                if self.matrix is None:
                    processed_frame, data_text = self.preview_and_calibrate(frame)
                else:
                    processed_frame, data_text = self.process_frame(frame)

                photo = ImageTk.PhotoImage(image=Image.fromarray(processed_frame))
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.photo = photo

                self.data_label.config(text=f"Marker Data: {data_text}")

            self.master.after(10, self.update)

    def preview_and_calibrate(self, frame):
        centers, orientations = self.detect_and_track_markers(frame)
        data_text = "Waiting for calibration markers..."

        if all(key in centers for key in [91, 92, 93, 94]):
            self.matrix, self.mask_contour, self.width, self.height = self.get_perspective_transform_matrix(
                self.config['mat']['length'], self.config['mat']['width'], centers, frame.shape
            )
            print('Found Distortion Matrix:')
            print(self.matrix)
            data_text = "Calibration complete. Tracking started."

        return frame, data_text  # Return original frame until calibration is done

    def process_frame(self, frame):
        centers, orientations = self.detect_and_track_markers(frame)

        data_text = ""
        if self.matrix is not None:
            self.transformed_frame = cv2.warpPerspective(frame, self.matrix, (self.width, self.height))

            for marker_id in self.config['markers']['track_range']:
                if marker_id in self.tracking_points:
                    transformed_points = self.transform_points(self.tracking_points[marker_id], self.matrix)
                    filtered_points = [
                        point for point in transformed_points if self.is_point_inside_mask(point, self.mask_contour)
                    ]
                    if filtered_points:
                        color = self.tracking_colors[marker_id]
                        orientation = orientations.get(marker_id, '')
                        self.all_transformed_data.append(
                            [self.frame_counter, marker_id, filtered_points[-1][0], filtered_points[-1][1], orientation]
                        )
                        data_text += f"Marker {marker_id}: ({filtered_points[-1][0]}, {filtered_points[-1][1]}), {orientation}°  "

            self.draw_transformed_tracks(self.transformed_frame, self.all_transformed_data, self.tracking_colors)
            processed_frame = cv2.rotate(self.transformed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            processed_frame = frame  # Show original frame if transformation not ready
            data_text = "Calibration lost. Waiting for markers..."

        self.frame_counter += 1
        return processed_frame, data_text

    def generate_color(self, marker_id):
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

    def save_tracking_results(self, transformed_frame, all_transformed_data, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_folder = os.path.join(self.config['output']['directory'], timestamp) 
        os.makedirs(run_folder, exist_ok=True) 

        rotated_frame = cv2.rotate(transformed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        desaturated_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2HSV)
        desaturated_frame = desaturated_frame.astype(float)
        desaturated_frame[..., 1] *= 0.2
        desaturated_frame = desaturated_frame.astype('uint8')
        desaturated_frame = cv2.cvtColor(desaturated_frame, cv2.COLOR_HSV2BGR)

        colorful_frame = desaturated_frame.copy()
        tracks = {}
        for _, marker_id, x, y, _ in all_transformed_data:
            if marker_id not in tracks:
                tracks[marker_id] = []
            adjusted_x = y
            adjusted_y = rotated_frame.shape[0] - x
            tracks[marker_id].append((adjusted_x, adjusted_y))

        for marker_id, track in tracks.items():
            color = self.generate_color(marker_id)
            for i in range(len(track) - 1):
                start_point = tuple(map(int, track[i]))
                end_point = tuple(map(int, track[i + 1]))
                cv2.line(colorful_frame, start_point, end_point, color, 2)

        cv2.imwrite(f'{run_folder}/{timestamp}.jpg', colorful_frame)

        frame_height, frame_width = rotated_frame.shape[:2]
        aspect_ratio = frame_height / frame_width
        svg_width = 1000
        svg_height = int(svg_width * aspect_ratio)

        dwg = svgwrite.Drawing(
            filename=f'{run_folder}/{timestamp}.svg',
            size=(svg_width, svg_height),
            viewBox=f'0 0 {svg_width} {svg_height}'
        )

        dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill='grey'))

        for marker_id, track in tracks.items():
            color = self.generate_color(marker_id)
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

        csv_file_path = f'{run_folder}/{timestamp}.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame', 'Marker', 'X', 'Y', 'Orientation'])
            csv_writer.writerows(all_transformed_data)


    def transform_points(self, points, matrix):
        if not points or matrix is None:
            return []
        points = np.array(points, dtype='float32').reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, matrix)
        return [tuple(map(int, point[0])) for point in transformed_points]

    def is_point_inside_mask(self, point, mask_contour):
        return cv2.pointPolygonTest(mask_contour, point, False) >= 0

    def detect_and_track_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        centers = {}
        orientations = {}
        excluded_ids = {91, 92, 93, 94}
        if ids is not None:
            for i, corner in enumerate(corners):
                center = np.mean(corner[0], axis=0)
                marker_id = ids[i][0]
                centers[marker_id] = center

                if marker_id not in excluded_ids:
                    vector = corner[0][1] - corner[0][0]
                    angle = int(np.arctan2(vector[1], vector[0]) * 180 / np.pi)
                    if angle < 0:
                        angle += 360
                    orientations[marker_id] = angle

                    if marker_id not in self.tracking_colors:
                        self.tracking_colors[marker_id] = self.generate_color(marker_id)
                    if marker_id not in self.tracking_points:
                        self.tracking_points[marker_id] = []
                    self.tracking_points[marker_id].append(center)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    outline_color = (255, 255, 255)
                    text_color = (0, 0, 0)

                    text_position_id = tuple(np.int32(corner[0][0]) + np.array([10, 0]))
                    text_position_angle = tuple(np.int32(center) + np.array([10, 15]))

                    cv2.putText(frame, str(marker_id), text_position_id, font, font_scale, outline_color, thickness + 2,
                                cv2.LINE_AA)
                    cv2.putText(frame, str(marker_id), text_position_id, font, font_scale, text_color, thickness,
                                cv2.LINE_AA)
                    cv2.putText(frame, str(angle), text_position_angle, font, font_scale, outline_color, thickness + 2,
                                cv2.LINE_AA)
                    cv2.putText(frame, str(angle), text_position_angle, font, font_scale, text_color, thickness,
                                cv2.LINE_AA)

                    cv2.polylines(frame, [np.int32(corner)], True, self.tracking_colors[marker_id], 1)

        return centers, orientations

    def get_perspective_transform_matrix(self, matLength, matWidth, centers, frame_shape):
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

    def draw_transformed_tracks(self, result, all_transformed_data, tracking_colors):
        tracks = {}
        for _, marker_id, x, y, _ in all_transformed_data:
            if marker_id not in tracks:
                tracks[marker_id] = []
            tracks[marker_id].append((x, y))

        for marker_id, track in tracks.items():
            color = tracking_colors.get(marker_id, (0, 255, 0))
            for i in range(len(track) - 1):
                cv2.line(result, tuple(map(int, track[i])), tuple(map(int, track[i + 1])), color, 1)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

root = tk.Tk()
app = LegoTrackerApp(root, config)
root.mainloop()