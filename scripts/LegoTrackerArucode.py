import cv2
from functions import save_tracking_results, transform_points, is_point_inside_mask, detect_and_track_markers, get_perspective_transform_matrix, draw_transformed_tracks

def undistort_and_track():
    print('starting...')
    cap = cv2.VideoCapture(0)
    frame_width = 1920
    frame_height = 1080

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 90)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeStep = 2
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23

    matrix = None
    mask_contour = None
    tracking_colors = {}
    tracking_points = {}
    transformed_tracks = {}
    width, height = 0, 0
    result = None

    # Define which marker range is to be tracked (minimizes false-positives)
    marker_ids_to_track = set(range(1, 4))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not captured.")
                continue

            centers, orientations = detect_and_track_markers(frame, dictionary, parameters, tracking_colors, tracking_points)

            # Ensure perspective matrix is set using markers 91-94
            if all(key in centers for key in [91, 92, 93, 94]) and matrix is None:
                matrix, mask_contour, width, height = get_perspective_transform_matrix(centers, frame.shape)
                print('Found Distortion Matrix:')
                print(matrix)

            if matrix is not None:
                result = cv2.warpPerspective(frame, matrix, (width, height))

                for marker_id in marker_ids_to_track:
                    if marker_id in tracking_points:
                        transformed_points = transform_points(tracking_points[marker_id], matrix)
                        filtered_points = [point for point in transformed_points if is_point_inside_mask(point, mask_contour)]
                        if filtered_points:
                            transformed_tracks[tracking_colors[marker_id]] = filtered_points

                draw_transformed_tracks(result, transformed_tracks)

                rotated_frame = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow('Transformed', rotated_frame)

            #cv2.imshow('Original', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if result is not None:
            save_tracking_results(result, transformed_tracks)
        cap.release()
        cv2.destroyAllWindows()

# Start detection and tracking
undistort_and_track()
