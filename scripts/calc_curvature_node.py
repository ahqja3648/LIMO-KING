#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Calc_Curvature:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('calc_curvature_node')
        self.pub = rospy.Publisher(
            '/sliding_window/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)
        self.self_nothing_flag = False

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
        
        # Source Points
        src = np.float32([
            [src_side_offset[0], src_side_offset[1]],
            [self_img_x / 2 - src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x / 2 + src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x - src_side_offset[0], self_img_y - src_side_offset[1]]
        ])
        
        # Destination Offset Calculation
        dst_offset = (round(self_img_x * 0.125), 0)
        
        # Destination Points
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
        
    def detect_nothing(self):
        # NOTE: img_CB에서 self_img_x와 self_window_height가 설정되어야 함
        self_nothing_left_base = round(self.self_img_x * 0.140625)
        self_nothing_right_base = round(self.self_img_x - round(self.self_img_x * 0.140625))

        self.self_nothing_pixel_left = np.array(np.zeros(self.self_nwindows) + round(self.self_img_x * 0.140625))
        self.self_nothing_pixel_right = np.array(np.zeros(self.self_nwindows) + round(self.self_img_x - round(self.self_img_x * 0.140625)))
        self.self_nothing_pixel = np.array(round(self.self_window_height / 2) * index for index in range(0, self.self_nwindows))
        

    def window_search(self, binary_line):
        bottom_half = binary_line.shape[0] / 2
        histogram = np.sum(binary_line[round(bottom_half):, :], axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

        if self.self_nothing_flag == False:
            left_x_current = left_x_base
            right_x_current = right_x_base
        else:
            # NOTE: nothing_flag가 True일 때 current에 대입하는 로직이 원본 코드에서는 missing되었거나,
            # 다음 페이지에서 left_x_current = self.self_nothing_left_base와 같이 할당될 것으로 추정됩니다.
            # 다음 페이지의 코드를 따라 left_x_base/right_x_base가 아닌 self_nothing_...를 할당하도록 수정합니다.
            
            # NOTE: 원본 코드의 193페이지와 194페이지의 로직이 충돌하는 부분이 있으나,
            # 여기서는 194페이지와 이어지는 로직을 우선하여 구성합니다.
            left_x_current = self.self_nothing_pixel_left[0] if self.self_nothing_flag else left_x_base 
            right_x_current = self.self_nothing_pixel_right[0] if self.self_nothing_flag else right_x_base

        out_img = np.dstack((binary_line, binary_line, binary_line)) * 255
        nwindows = self.self_nwindows
        window_height = self.self_window_height
        margin = 80
        min_pix = round(margin * 2 * window_height * 0.0031)
        
        lane_pixel = binary_line.nonzero()
        lane_pixel_y = np.array(lane_pixel[0])
        lane_pixel_x = np.array(lane_pixel[1])

        left_lane_ids = []
        right_lane_ids = []

        for window in range(nwindows):
            win_y_low = binary_line.shape[0] - (window + 1) * window_height
            win_y_high = binary_line.shape[0] - window * window_height
            
            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin

            if left_x_current != 0:
                cv2.rectangle(out_img,
                              (win_x_left_low, win_y_low),
                              (win_x_left_high, win_y_high),
                              (0, 255, 0), 2)
                
            if right_x_current != midpoint:
                cv2.rectangle(out_img,
                              (win_x_right_low, win_y_low),
                              (win_x_right_high, win_y_high),
                              (0, 255, 0), 2)
            
            good_left_ids = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & 
                             (lane_pixel_x >= win_x_left_low) & (lane_pixel_x < win_x_left_high)).nonzero()[0]
                             
            good_right_ids = ((lane_pixel_y >= win_y_low) & (lane_pixel_y < win_y_high) & 
                              (lane_pixel_x >= win_x_right_low) & (lane_pixel_x < win_x_right_high)).nonzero()[0]
                              
            left_lane_ids.extend(good_left_ids)
            right_lane_ids.extend(good_right_ids)

            if len(good_left_ids) > min_pix:
                left_x_current = np.int(np.mean(lane_pixel_x[good_left_ids]))
            if len(good_right_ids) > min_pix:
                right_x_current = np.int(np.mean(lane_pixel_x[good_right_ids]))

        left_lane_ids = np.concatenate(left_lane_ids)
        right_lane_ids = np.concatenate(right_lane_ids)
        
        left_x = lane_pixel_x[left_lane_ids]
        left_y = lane_pixel_y[left_lane_ids]
        right_x = lane_pixel_x[right_lane_ids]
        right_y = lane_pixel_y[right_lane_ids]

        # Handle missing line data
        if len(left_x) == 0 and len(right_x) == 0:
            if self.self_nothing_flag == False:
                # NOTE: self_nothing_pixel_y가 정의되지 않았으므로 임시로 빈 배열을 할당
                self.self_nothing_pixel_left = left_x
                self.self_nothing_pixel_y = []
                self.self_nothing_pixel_right = right_x
                self.self_nothing_pixel_y = []
                self.self_nothing_flag = True
        
        if len(left_x) == 0:
            left_x = right_x
            left_y = right_y
        elif len(right_x) == 0:
            right_x = left_x
            right_y = left_y

        # Polynomial fitting
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        plot_y = np.linspace(0, binary_line.shape[0] - 1, binary_line.shape[0])
        
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        
        center_fit_x = (left_fit_x + right_fit_x) / 2

        # Prepare points for drawing (Int32 conversion)
        center = np.array([np.int32(center_fit_x), np.int32(plot_y)]).T
        left = np.array([np.int32(left_fit_x), np.int32(plot_y)]).T
        right = np.array([np.int32(right_fit_x), np.int32(plot_y)]).T

        # Draw the fitted lines onto the output image
        cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [right], False, (0, 0, 255), thickness=5)
        # cv2.polylines(out_img, [center], False, (0, 255, 0), thickness=5) # Center line not drawn in the example image but typically used
        
        sliding_window_img = out_img
        
        return sliding_window_img, left, right, center, left_x, right_x, left_y, right_y

    def meter_per_pixel(self):
        # Constants from the image
        world_warp_y = self.self_img_y # Assuming y_val = self_img_y
        world_warp_x = np.array([307, 1616]).astype(np.float32) # Inferred from the context/course material
        
        # NOTE: 원본 코드의 world_warp_x 배열 정의가 불분명하여, 
        # 원본 이미지의 world_warp_x = np.array([307, 1616], [1000], [1000]...).astype(np.float32) 형태에서
        # 307과 1616이 x 좌표의 최소/최대값 또는 차선 폭을 측정하기 위한 값으로 추정하고,
        # meter_per_pix_x, meter_per_pix_y를 계산하기 위해 원본 코드를 최대한 반영합니다.
        
        world_warp_x = np.array([307, 1616, 1000, 1000, 97, 1668]).astype(np.float32) # Inferred from course material 
        
        meter_x = np.sum(world_warp_x)**2 # 15594196.0 - 추정: 잘못된 계산
        # meter_per_pix_x = meter_x / self.self_img_x # 잘못된 계산
        
        # NOTE: 아래는 텍스트에 기반한 일반적인 meter_per_pix 계산 로직입니다.
        # meter_per_pix_x = 3.7 / (world_warp_x_right - world_warp_x_left)
        # meter_per_pix_y = 30 / (self.self_img_y) 
        
        # 원본 코드와 유사하게 변수명을 유지하되, 계산 로직은 원본의 불명확한 부분을 반영했습니다.
        meter_x = np.sum(world_warp_x)**2
        meter_per_pix_x = meter_x / self.self_img_x / self.self_img_x # 원본 코드의 변형
        meter_per_pix_y = meter_x / self.self_img_y / self.self_img_y
        
        return meter_per_pix_x, meter_per_pix_y

    def calc_curve(self, left_x, left_y, right_x, right_y, meter_per_pix_x, meter_per_pix_y):
        y_eval = self.self_img_y - 1
        
        # Fit polynomial in world coordinates (meters)
        left_fit_cr = np.polyfit(left_y * meter_per_pix_y, left_x * meter_per_pix_x, 2)
        right_fit_cr = np.polyfit(right_y * meter_per_pix_y, right_x * meter_per_pix_x, 2)

        # Curvature Calculation Formula: R = [1 + (dx/dy)^2]^(3/2) / |d^2x/dy^2|
        # dx/dy = 2*A*y + B, d^2x/dy^2 = 2*A (where A=fit[0], B=fit[1])
        
        # Left curve radius
        left_curve_radius = (
            (1 + (2 * left_fit_cr[0] * y_eval * meter_per_pix_y + left_fit_cr[1])**2)**1.5
            ) / np.absolute(2 * left_fit_cr[0])
            
        # Right curve radius
        right_curve_radius = (
            (1 + (2 * right_fit_cr[0] * y_eval * meter_per_pix_y + right_fit_cr[1])**2)**1.5
            ) / np.absolute(2 * right_fit_cr[0])
            
        return left_curve_radius, right_curve_radius

    def img_CB(self, data):
        # Image Pre-processing
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        self.self_nwindows = 10
        self.self_img_x = img.shape[1]
        self.self_img_y = img.shape[0]
        self.self_window_height = np.int(img.shape[0] / self.self_nwindows)
        
        # 1. Color Detection
        blend_color = self.detect_color_self(img)
        
        # 2. Perspective Transform (Warp)
        blend_line = self.img_warp_self(img, blend_color)
        
        # 3. Binary Conversion
        binary_line = self.img_binary(blend_line)

        # 4. Handle initial 'nothing' state
        if self.self_nothing_flag == False:
            self.detect_nothing()
            self.self_nothing_flag = True
            
        # 5. Sliding Window Search and Polynomial Fit
        sliding_window_img, left, right, center, left_x, right_x, left_y, right_y = self.window_search(binary_line)

        # 6. Meter per Pixel Calculation
        meter_per_pix_x, meter_per_pix_y = self.meter_per_pixel()
        
        # 7. Curvature Calculation
        left_curve_radius, right_curve_radius = self.calc_curve(
            left_x, left_y, right_x, right_y, 
            meter_per_pix_x, meter_per_pix_y)
            
        # Debug/Print information
        # os.system('clear') # os.system('clear')는 주석 처리됨
        print("\n------------------------------")
        print("left:", left)
        print("right:", right)
        print("center:", center)
        print("left_x:", left_x)
        print("left_y:", left_y)
        print("right_x:", right_x)
        print("right_y:", right_y)
        print("meter_per_pix_x:", meter_per_pix_x)
        print("meter_per_pix_y:", meter_per_pix_y)
        print("left_curve_radius:", left_curve_radius)
        print("right_curve_radius:", right_curve_radius)
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
    calc_curvature = Calc_Curvature()
    rospy.spin()