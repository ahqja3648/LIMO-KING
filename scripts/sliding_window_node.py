#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Sliding_Window:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('sliding_window_node')
        self.pub = rospy.Publisher(
            '/sliding_window/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)
        self.self_nothing_key = False
        
    def detect_color_self(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        yellow_lower = np.array([15, 80, 0])
        yellow_upper = np.array([45, 255, 255])

        white_lower = np.array([0, 0, 200])
        white_upper = np.array([179, 64, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        blend_mask = cv2.bitwise_or(yellow_mask, white_mask)
        blend_color = cv2.bitwise_and(img, img, mask=blend_mask)
        return blend_color

    def img_warp_self(self, img, blend_color):
        self_img_x = img.shape[1]
        self_img_y = img.shape[0]

        src_side_offset = round(self_img_x * 0.046875), round(self_img_y * 0.208)
        src_center_offset = round(self_img_x * 0.14), round(self_img_y * 0.008)
        
        src = np.float32([
            [src_side_offset[0], src_side_offset[1]],
            [self_img_x / 2 - src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x / 2 + src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x - src_side_offset[0], src_side_offset[1]]
        ])
        
        # Destination Offset Calculation
        dst_offset = (round(self_img_x * 0.125), 0)
        
        dst = np.float32([
            [dst_offset[0], self_img_y],
            [dst_offset[0], 0],
            [self_img_x - dst_offset[0], 0],
            [self_img_x - dst_offset[0], self_img_y]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        
        blend_line = cv2.warpPerspective(blend_color, matrix, (self_img_x, self_img_y))
        return blend_line

    def img_binary(self, blend_line):
        bin = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        binary_line = np.zeros_like(bin)
        binary_line[bin > 0] = 1
        return binary_line
        
    def detect_nothing_self(self):
        # NOTE: 이 함수는 self_nothing_pixel_left/right 변수를 설정하는 로직으로 보이지만,
        # 이 변수들이 클래스 초기화(__init__) 시 정의되어 있지 않아 에러가 발생할 수 있습니다.
        # 코드의 흐름을 보존하기 위해 그대로 배치합니다.

        self_nothing_left_base = round(self.self_img_x * 0.140625)
        self_nothing_right_base = round(self.self_img_x - round(self.self_img_x * 0.140625))

        self.self_nothing_pixel_left = np.array(np.zeros(self.self_window_height) + round(self.self_img_x * 0.140625))
        self.self_nothing_pixel_right = np.array(np.zeros(self.self_window_height) + round(self.self_img_x - round(self.self_img_x * 0.140625)))
        self.self_nothing_pixel = round(self.self_window_height / 2) # index for index in range(0, self.nwindows)

    def window_search(self, binary_line):
        bottom_half = binary_line.shape[0] / 2
        histogram = np.sum(binary_line[round(bottom_half):, :], axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        # Check for initial line position (noting nothing_key usage)
        if self.self_nothing_key == False:
            left_x_current = left_x_base
            right_x_current = right_x_base
        else:
            left_x_current = self.self_nothing_pixel_left
            right_x_current = self.self_nothing_pixel_right

        out_img = np.dstack((binary_line, binary_line, binary_line)) * 255
        nwindows = self.self_nwindows # self.self_nwindows는 img_CB에서 설정됨
        window_height = self.self_window_height
        margin = 80
        min_pix = round(margin * 2 * window_height * 0.0031) # min_pix
        
        lane_pixel = binary_line.nonzero()
        lane_pixel_y = np.array(lane_pixel[0])
        lane_pixel_x = np.array(lane_pixel[1])

        left_lane_ids = []
        right_lane_ids = []

        for window in range(nwindows):
            # Window boundaries
            win_y_low = binary_line.shape[0] - (window + 1) * window_height
            win_y_high = binary_line.shape[0] - window * window_height
            
            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin

            # Draw the windows for visualization
            if left_x_current != 0: # Check to avoid drawing windows on nothing
                cv2.rectangle(out_img,
                              (win_x_left_low, win_y_low),
                              (win_x_left_high, win_y_high),
                              (0, 255, 0),
                              2)
                
            if right_x_current != midpoint: # Check to avoid drawing windows on nothing
                cv2.rectangle(out_img,
                              (win_x_right_low, win_y_low),
                              (win_x_right_high, win_y_high),
                              (0, 255, 0),
                              2)
            
            # Identify pixels inside the window
            good_left_ids = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & 
                             (lane_pixel_x >= win_x_left_low) & (lane_pixel_x < win_x_left_high)).nonzero()[0]
                             
            good_right_ids = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & 
                              (lane_pixel_x >= win_x_right_low) & (lane_pixel_x < win_x_right_high)).nonzero()[0]
                              
            # Append ids to the list
            left_lane_ids.extend(good_left_ids)
            right_lane_ids.extend(good_right_ids)

            # Recenter the next window if enough pixels are found
            if len(good_left_ids) > min_pix:
                left_x_current = np.int(np.mean(lane_pixel_x[good_left_ids]))
            if len(good_right_ids) > min_pix:
                right_x_current = np.int(np.mean(lane_pixel_x[good_right_ids]))

        # Concatenate all pixel indices
        left_lane_ids = np.concatenate(left_lane_ids)
        right_lane_ids = np.concatenate(right_lane_ids)
        
        # Extract left and right line pixel positions
        left_x = lane_pixel_x[left_lane_ids]
        left_y = lane_pixel_y[left_lane_ids]
        right_x = lane_pixel_x[right_lane_ids]
        right_y = lane_pixel_y[right_lane_ids]

        # Handle case where no line pixels were found in the window search
        if len(left_x) == 0 and len(right_x) == 0:
            if self.self_nothing_key == False:
                self.self_nothing_pixel_left = left_x
                self.self_nothing_pixel_y = left_y
                self.self_nothing_pixel_right = right_x
                self.self_nothing_pixel_y = right_y
                self.self_nothing_key = True
            
            # NOTE: 이 부분은 차선이 없을 때 기본 위치를 반환하는 로직으로 추정됩니다.
            left_x = np.round(self_img_x / 2)
            left_y = right_y
            
            right_x = np.round(self_img_x / 2)
            right_y = left_y
        
        else: # Lines were detected, set nothing_key to False
            self.self_nothing_key = False

        # Polynomial fitting
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate y values for plotting
        plot_y = np.linspace(0, binary_line.shape[0] - 1, binary_line.shape[0])
        
        # Calculate x values based on the fitted polynomial
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        
        # Calculate center line x position
        center_fit_x = (left_fit_x + right_fit_x) / 2

        # Prepare points for drawing
        center = np.array([np.int32(center_fit_x), np.int32(plot_y)]).T
        left = np.array([np.int32(left_fit_x), np.int32(plot_y)]).T
        right = np.array([np.int32(right_fit_x), np.int32(plot_y)]).T

        # Draw the fitted lines onto the output image
        cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [right], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [center], False, (0, 255, 0), thickness=5)

        sliding_window_img = out_img
        
        # Return the visualization image and line data
        return sliding_window_img, left, right, center, left_x, right_x, left_y, right_y

    def img_CB(self, data):
        # Image pre-processing
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        
        self.self_img_x = img.shape[1]
        self.self_img_y = img.shape[0]
        self.self_nwindows = 10 
        self.self_window_height = np.int(img.shape[0] / self.self_nwindows)
        
        # 1. Color Detection
        blend_color = self.detect_color_self(img)
        
        # 2. Perspective Transform (Warp)
        blend_line = self.img_warp_self(img, blend_color)
        
        # 3. Binary Conversion
        binary_line = self.img_binary(blend_line)

        # 4. Handle initial 'nothing' state
        if self.self_nothing_key == False:
            self.detect_nothing_self()
            self.self_nothing_key = True # Prevent repeated initialization on every frame if nothing is found
            
        # 5. Sliding Window Search and Polynomial Fit
        sliding_window_img, left, right, center, left_x, right_x, left_y, right_y = self.window_search(binary_line)

        # Debug/Print information
        print("\n------------------------------")
        print("center:", center)
        print("left_x:", left_x)
        print("left_y:", left_y)
        print("right_x:", right_x)
        print("right_y:", right_y)
        print("------------------------------")
        
        # Publish the visualization image
        sliding_window_msg = self.bridge.cv2_to_compressed_imgmsg(sliding_window_img)
        self.pub.publish(sliding_window_msg)
        
        # Display images
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("sliding_window_img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("sliding_window_img", sliding_window_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    sliding_window = Sliding_Window()
    rospy.spin()

    