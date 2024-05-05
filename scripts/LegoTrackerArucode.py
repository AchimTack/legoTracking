import cv2
import numpy as np

from functions import apply_perspective_transform, save_tracking_results

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOCUS, float('inf'))

if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit(0)

clicked_points = []
rect_defined = False
dragging_point = None
mask_locked = False
track_history = []

def get_corners(event, x, y, flags, param):
    global clicked_points, rect_defined, img, dragging_point
    need_to_update = False
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, point in enumerate(clicked_points):
            if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                dragging_point = idx
                return
        if len(clicked_points) < 4 and dragging_point is None:
            clicked_points.append((x, y))
            need_to_update = True
        if len(clicked_points) == 4:
            rect_defined = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging_point is not None:
        clicked_points[dragging_point] = (x, y)
        need_to_update = True
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None
        need_to_update = True

    if need_to_update:
        img = frame.copy()
        for point in clicked_points:
            cv2.circle(img, point, 5, (0, 255, 0), -1)
        if len(clicked_points) > 1:
            for i in range(len(clicked_points)):
                cv2.line(img, clicked_points[i], clicked_points[(i + 1) % len(clicked_points)], (0, 255, 0), 2)
        cv2.imshow('Frame', img)

ret, frame = cap.read()
img = frame.copy()
cv2.imshow('Frame', img)
cv2.setMouseCallback('Frame', get_corners)

while not mask_locked:
    if not rect_defined or dragging_point is not None:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam")
            break
        cv2.imshow('Frame', img if 'img' in locals() else frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('m') and rect_defined:
        mask_locked = True
        cv2.destroyWindow('Frame')
    elif key & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

src_pts = np.array(clicked_points, dtype="float32")
transform_matrix = apply_perspective_transform(src_pts)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

midpoints = []
path = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam")
        break

    if not mask_locked:
        if rect_defined or dragging_point is not None:
            img = frame.copy()
            for point in clicked_points:
                cv2.circle(img, point, 5, (0, 255, 0), -1)
            if len(clicked_points) > 1:
                for i in range(len(clicked_points)):
                    cv2.line(img, clicked_points[i], clicked_points[(i + 1) % len(clicked_points)], (0, 255, 0), 2)
            cv2.imshow('Frame', img)
    else:
        transformed_frame = cv2.warpPerspective(frame, transform_matrix, (1280, 1024))
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(transformed_frame)
        if markerIds is not None:
            cv2.aruco.drawDetectedMarkers(transformed_frame, markerCorners, markerIds)
            for i, markerId in enumerate(markerIds):
                if markerId[0] <=10:
                    corners = markerCorners[i][0]
                    midpoint = np.mean(corners, axis=0).astype(int)
                    track_history.append(midpoint)

        # Draw the track even if no new markers are detected
        if track_history:
            cv2.polylines(transformed_frame, [np.array(track_history, dtype=np.int32)], False, (0, 0, 255), 2)

        cv2.imshow('Transformed Frame', transformed_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('m'):
        mask_locked = True
        cv2.destroyWindow('Frame')
    elif key & 0xFF == ord('q'):
        save_tracking_results(transformed_frame, track_history)
        break

# Exit clean up
cap.release()
cv2.destroyAllWindows()

