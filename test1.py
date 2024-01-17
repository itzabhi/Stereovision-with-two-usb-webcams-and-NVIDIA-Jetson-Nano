import sys
import os
import numpy as np
import argparse
import subprocess
import utilities.red_square as red_square
import copy
import pickle
import stereo_vision.projection
import imageio
import cv2



video_dev0 = 0
video_dev1 = 1
image_width = 1280
image_height = 720
window0 = "camera-0"
window1 = "camera-1"
absolute_path = os.path.dirname(__file__)
inputImagesFilepathPrefix = os.path.join(absolute_path,'./red_square_images/camera_')
outputDirectory =os.path.join(absolute_path,'./output_track_red_square')
redSquareDetectorBlueDelta = 15
redSquareDetectorBlueDilationSize = 45
redSquareDetectorRedDelta = 70
redSquareDetectorRedDilationSize = 13
projectionMatrix1Filepath = os.path.join(absolute_path,"./output_calibrate_stereo/camera1.projmtx")
projectionMatrix2Filepath = os.path.join(absolute_path,"./output_calibrate_stereo/camera2.projmtx")
radialDistortion1Filepath = os.path.join(absolute_path,'./radial_distortion/calibration_left.pkl')
radialDistortion2Filepath = os.path.join(absolute_path,'./radial_distortion/calibration_right.pkl')
red_square_detector = red_square.Detector(
        blue_delta=redSquareDetectorBlueDelta,
        blue_mask_dilation_kernel_size=redSquareDetectorBlueDilationSize,
        red_delta=redSquareDetectorRedDelta,
        red_mask_dilation_kernel_size=redSquareDetectorRedDilationSize,
        debug_directory=None
    )

# Load the projection matrices
P1, P2 = None, None
with open(projectionMatrix1Filepath, 'rb') as proj1_file:
    P1 = pickle.load(proj1_file)
with open(projectionMatrix2Filepath, 'rb') as proj2_file:
    P2 = pickle.load(proj2_file)
# Create the stereo vision system
stereo_system = stereo_vision.projection.StereoVisionSystem([P1, P2])

# Load the radial distortion models
radial_dist1, radial_dist2 = None, None
with open(radialDistortion1Filepath, 'rb') as radial_dist1_file:
    radial_dist1 = pickle.load(radial_dist1_file)
with open(radialDistortion2Filepath, 'rb') as radial_dist2_file:
    radial_dist2 = pickle.load(radial_dist2_file)
radial_distortions = [radial_dist1, radial_dist2]

def main():
    cap0 = cv2.VideoCapture(video_dev0) 
    cap1 = cv2.VideoCapture(video_dev1) 
   
    if not cap0.isOpened() or not cap1.isOpened():
        sys.exit('Failed to open camera!')

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Unable to capture frames from ne or both cameras.")
            break
        
        images = [frame0, frame1]
        
        mosaic_img, undistorted_centers = centers(images)
        XYZ = stereo_system.SolveXYZ(undistorted_centers)
        uv = undistorted_centers[0]
        cv2.putText(mosaic_img, "({:.1f}, {:.1f}, {:.1f})".format(XYZ[0], XYZ[1], XYZ[2]),
                    (round(uv[0]) + 10, round(uv[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
        #cv2.imshow(window0, frame0)
        #cv2.imshow(window1, frame1)
        cv2.imshow("3D Position", mosaic_img)

        print(undistorted_centers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def detect_color(img):
    min_contour_area = 1000
    # Convert the frame to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Define the green color range in HSV
    lower_green = np.array([168, 98, 99])
    upper_green = np.array([98, 98, 32])

    # Create a binary mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store information about the largest contour
    max_contour_area = 0
    max_contour = None

    # Iterate over the contours and find the largest one
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            max_contour = contour

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        center_of_mass = (x+(w//2), y+(h//2))
    else:
        center_of_mass = (-1, -1)      
    return center_of_mass

def centers(images):
    undistorted_centers = []
    img_shapeHWC = images[0].shape
    mosaic_img = np.zeros((img_shapeHWC[0], len(images) * img_shapeHWC[1], img_shapeHWC[2]), dtype=np.uint8)
    for image_ndx in range(len(images)):
        image = images[image_ndx]
        annotated_img = copy.deepcopy(image)
        center = detect_color(image)  #red_square_detector.Detect(image)
        center_rounded = (round(center[0]), round(center[1]))
        cv2.line(annotated_img, (center_rounded[0] - 5, center_rounded[1]), (center_rounded[0] + 5, center_rounded[1]), (255, 0, 0),
                    thickness=3)
        cv2.line(annotated_img, (center_rounded[0], center_rounded[1] - 5),
                    (center_rounded[0], center_rounded[1] + 5), (255, 0, 0),
                    thickness=3)
        # Undistort the coordinates
        undistorted_center = radial_distortions[image_ndx].UndistortPoint(center)
        undistorted_centers.append(undistorted_center)
        mosaic_img[:, image_ndx * img_shapeHWC[1]: (image_ndx + 1) * img_shapeHWC[1], :] = annotated_img
    return mosaic_img, undistorted_centers

if __name__ == '__main__':
    main()