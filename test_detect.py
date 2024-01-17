import os
import sys
import cv2
import copy
import utilities.red_square as red_square

video_dev0 = 0
video_dev1 = 1
image_width = 1280
image_height = 720
window0 = "camera-0"
window1 = "camera-1"
absolute_path = os.path.dirname(__file__)

redSquareDetectorBlueDelta = 15
redSquareDetectorBlueDilationSize = 45
redSquareDetectorRedDelta = 70
redSquareDetectorRedDilationSize = 13
red_square_detector = red_square.Detector(
        blue_delta=redSquareDetectorBlueDelta,
        blue_mask_dilation_kernel_size=redSquareDetectorBlueDilationSize,
        red_delta=redSquareDetectorRedDelta,
        red_mask_dilation_kernel_size=redSquareDetectorRedDilationSize,
        debug_directory=None
    )

def main():
    cap0 = cv2.VideoCapture(video_dev0) 
     
   
    if not cap0.isOpened():
        sys.exit('Failed to open camera!')

    while True:
        ret0, frame0 = cap0.read()

        if not ret0:
            print("Unable to capture frames from ne or both cameras.")
            break
        annotated_img = copy.deepcopy(frame0)
        center = red_square_detector.Detect(frame0)
        center_rounded = (round(center[0]), round(center[1]))
        cv2.line(annotated_img, (center_rounded[0] - 5, center_rounded[1]), (center_rounded[0] + 5, center_rounded[1]), (255, 0, 0),
                    thickness=3)
        cv2.line(annotated_img, (center_rounded[0], center_rounded[1] - 5),
                    (center_rounded[0], center_rounded[1] + 5), (255, 0, 0),
                    thickness=3)
        cv2.imshow("3D Position", annotated_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap0.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()