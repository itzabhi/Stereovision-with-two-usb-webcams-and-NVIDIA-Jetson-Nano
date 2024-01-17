xyz coordinate of an object with two camera and jetson nano

import cv2
import numpy as np

# Load calibration data
calibration_data = np.load('calibration_data.npz')
camera_matrix_left = calibration_data['camera_matrix_left']
dist_coeff_left = calibration_data['dist_coeff_left']
camera_matrix_right = calibration_data['camera_matrix_right']
dist_coeff_right = calibration_data['dist_coeff_right']
R, T, E, F = calibration_data['extrinsic_params']

# Open camera capture objects
cap_left = cv2.VideoCapture(0)  # Adjust the camera indices as needed
cap_right = cv2.VideoCapture(1)

# Create stereo vision object
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=5)

while True:
    # Capture frames
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    # Rectify images
    map_left, map_right = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeff_left, R, camera_matrix_left, (640, 480), cv2.CV_32FC1)
    rectified_left = cv2.remap(frame_left, map_left, map_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(frame_right, map_right, map_right, cv2.INTER_LINEAR)

    # Compute disparity map
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right)

    # Convert disparity map to depth map
    depth = cv2.reprojectImageTo3D(disparity, Q=R)

    # Extract 3D coordinates of a specific pixel (adjust x, y as needed)
    x, y = 320, 240  # Example pixel coordinates
    depth_at_pixel = depth[y, x]

    print(f"Depth at pixel ({x}, {y}): {depth_at_pixel}")

    # Display images and disparity map (optional)
    cv2.imshow('Rectified Left', rectified_left)
    cv2.imshow('Rectified Right', rectified_right)
    cv2.imshow('Disparity Map', disparity.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
