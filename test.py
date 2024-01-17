import cv2
import os


absolute_path = os.path.dirname(__file__)
image = cv2.imread(os.path.join(absolute_path,"./calibration_images/camera_1_60cm.png"))

if image is not None:
    cv2.imshow('Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('error')