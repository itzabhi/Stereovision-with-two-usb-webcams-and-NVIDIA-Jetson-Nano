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
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.spatial import distance


video_dev0 = 'dev/video0'
video_dev1 = 'dev/video1'
image_width = 1280
image_height = 720
window0 = "camera-0"
window1 = "camera-1"
absolute_path = os.path.dirname(__file__)
virtual_img = os.path.join(absolute_path,'./virtual_images/weld_4.jpg')
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


class App:
    def __init__(self, root, camera_id0, camera_id1):
        self.root = root
        self.root.title("Digital Test Kit")

        # Initialize variables
        self.prev_position = None
        self.times = []
        self.velocities = []
        self.distances = []
        self.trajectory = []

        # Create a video capture object for the camera
        self.cap0 = cv2.VideoCapture(camera_id0)
        self.cap1 = cv2.VideoCapture(camera_id1)

        # Create a label for displaying the Headers
        '''self.label0 = tk.Label(root, width = 40, height = 4, bd = 1, text='Left Camera')
        self.label0.grid(row=0, column=0)
        self.label1 = tk.Label(root, width = 400, height = 4, bd = 1, text='Right Camera')
        self.label1.grid(row=0, column=1)
        self.label2 = tk.Label(root, width = 400, height = 4, bd = 1, text='Digital Object')
        self.label2.grid(row=0, column=2)'''

        # Create a label for displaying the camera feed
        self.label = tk.Label(root, width = 1280, height = 400, bd = 2)
        self.label.grid(row=0, column=0) #pack(padx=10, pady=10)

        self.plotframe = tk.Frame(root, width = 430, height = 1, bd = 1, bg = "gray")
        self.plotframe.grid(row=1, column=0)
        # Create a Matplotlib figure for the velocity plot
        self.figure, self.ax = plt.subplots(figsize=(5,2))
        self.line, = self.ax.plot([], [], label='Velocity')
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plotframe)
        self.canvas.get_tk_widget().grid(row=0, column=0) #pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create a Matplotlib figure for the Distance plot
        self.figure1, self.ax1 = plt.subplots(figsize=(5,2))
        self.line1, = self.ax1.plot([], [], label='Distance')
        self.ax1.legend()
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plotframe)
        self.canvas1.get_tk_widget().grid(row=0, column=1) #pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        if not self.cap0.isOpened() or not self.cap1.isOpened():
            sys.exit('Failed to open camera!')
        # Start updating the camera feed and velocity plot
        self.update()


    def update(self):
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        frame2 = cv2.imread(virtual_img)
        if ret0 and ret1:
            frame0 = cv2.flip(self.resize_img(frame0), 1)
            frame1 = cv2.flip(self.resize_img(frame1), 1)
            frame2 = cv2.resize(frame2, (420, 315))

            images = [frame0, frame1, frame2]
            
            mosaic_img, undistorted_centers = self.centers(images)
            XYZ = stereo_system.SolveXYZ(undistorted_centers)
            uv = undistorted_centers[0]
            cv2.putText(mosaic_img, "({:.1f}, {:.1f}, {:.1f})".format(XYZ[0], XYZ[1], XYZ[2]),
                        (round(uv[0]) + 10, round(uv[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
            
            current_position = (XYZ[0], XYZ[1], XYZ[2])
            #cv2.imshow("3D Position", mosaic_img)
            if current_position is not None:
                # Calculate and update velocity
                velocity = self.calculate_velocity(current_position)
                self.velocities.append(velocity)
                self.times.append(len(self.times))
                self.prev_position = current_position

                # Update the velocity plot
                self.line.set_xdata(self.times)
                self.line.set_ydata(self.velocities)
                self.ax.relim()
                self.ax.autoscale_view()
                
                self.distances.append(XYZ[1])
                # Update the velocity plot
                self.line1.set_xdata(self.times)
                self.line1.set_ydata(self.distances)
                self.ax1.relim()
                self.ax1.autoscale_view()

            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)
            

            # Convert the frame to Tkinter PhotoImage
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)

            # Update the label with the new frame
            self.label.config(image=img)
            self.label.image = img

        # Schedule the next update
        self.root.after(10, self.update)  # Update every 100 milliseconds

    def resize_img(self, img):
        new_width = 420
        aspect_ratio = img.shape[1] / img.shape[0]
        new_ht = int(new_width / aspect_ratio)
        new_dimensions = (new_width, new_ht)
        resized_img = cv2.resize(img, new_dimensions)
        return resized_img

    def detect_color(self, img):
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

    def centers(self, images):
        
        undistorted_centers = []
        img_shapeHWC = images[0].shape
        mosaic_img = np.zeros((img_shapeHWC[0], len(images) * img_shapeHWC[1], img_shapeHWC[2]), dtype=np.uint8)
        for image_ndx in range(len(images)):
            image = images[image_ndx]
            annotated_img = copy.deepcopy(image)

            if image_ndx == 1 and center != (-1, -1):
                # Append the current position to the trajectory
                self.trajectory.append(center)
                # Draw the trajectory on the frame
                self.draw_trajectory(annotated_img, self.trajectory)

            if image_ndx < 2 :
                center = self.detect_color(image)  #red_square_detector.Detect(image)
                center_rounded = (round(center[0]), round(center[1]))
                cv2.line(annotated_img, (center_rounded[0] - 5, center_rounded[1]), (center_rounded[0] + 5, center_rounded[1]), (255, 0, 0), thickness=3)
                cv2.line(annotated_img, (center_rounded[0], center_rounded[1] - 5),(center_rounded[0], center_rounded[1] + 5), (255, 0, 0), thickness=3)
                # Undistort the coordinates
                undistorted_center = radial_distortions[image_ndx].UndistortPoint(center)
                undistorted_centers.append(undistorted_center)
            mosaic_img[:, image_ndx * img_shapeHWC[1]: (image_ndx + 1) * img_shapeHWC[1], :] = annotated_img
        return mosaic_img, undistorted_centers

    # Function to calculate distance between two points
    def calculate_distance(self, point1, point2):
        return int(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

    def calculate_velocity(self, current_position):
        if self.prev_position is not None:
            distance_meters = distance.euclidean(self.prev_position, current_position)
            time_elapsed = len(self.times) * 0.05  # Assuming 20 frames per second
            velocity = distance_meters / time_elapsed
            return velocity
        return 0.0

    # Function to draw the trajectory on the frame
    def draw_trajectory(self, frame, trajectory):
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

if __name__ == '__main__':
    root = tk.Tk()

    app = App(root, video_dev0, video_dev1)
    app.root.mainloop()
