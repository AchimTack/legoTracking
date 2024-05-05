import cv2 as cv
import numpy as np

# Test different indices if 0 does not work
cap = cv.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit(0)

# Define the ArUco dictionary and parameters
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters =  cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

while True:
    ret, frame = cap.read()
    if ret:
        # Detect ArUco markers
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        
        # If markers are detected, draw them and display their IDs and orientation
        if markerIds is not None:
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            # Dummy camera parameters for pose estimation
            cameraMatrix = np.array([[1000, 0, frame.shape[1]/2], [0, 1000, frame.shape[0]/2], [0, 0, 1]])
            distCoeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            for i, corner in enumerate(markerCorners):
                # Estimate pose of each marker
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corner, 1, cameraMatrix, distCoeffs)
                cv.putText(frame, f"ID: {markerIds[i][0]}", (int(corner[0][0][0]), int(corner[0][0][1] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(frame, f"Orientation: {rvec[0][0][2]:.2f}", (int(corner[0][0][0]), int(corner[0][0][1] + 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv.imshow('Frame', frame)
        
        # Exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Unable to read from webcam")
        break

# Release resources
cap.release()
cv.destroyAllWindows()
