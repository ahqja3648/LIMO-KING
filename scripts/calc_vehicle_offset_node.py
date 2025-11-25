#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os # os.system('clear')를 위해 필요

class Calc_Vehicle_Offset:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('calc_vehicle_offset_node')
        self.pub = rospy.Publisher(
            '/sliding_window/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)
        self.nothing_flag = False

    def detect_color_self(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Yellow color detection range
        yellow_lower = np.array([15, 80, 0])
        yellow_upper = np.array([45, 255, 255])

        # White color detection range
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
            [self_img_x - src_side_offset[0], src_side_offset[1]]
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
        # NOTE: self.self_nwindows와 self.self_window_height는 img_CB에서 설정됨
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

        if self.nothing_flag == False:
            left_x_current = left_x_base
            right_x_current = right_x_base
        else:
            # nothing_flag가 True일 때 초기값 할당 (원본 코드 로직 반영)
            left_x_current = self.self_nothing_pixel_left[0] if self.nothing_flag else left_x_base 
            right_x_current = self.self_nothing_pixel_right[0] if self.nothing_flag else right_x_base

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
                cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
                
            if right_x_current != midpoint:
                cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
            
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
            if self.nothing_flag == False:
                # NOTE: self_nothing_pixel_y가 정의되지 않아 self_nothing_pixel_left/right로 대체
                self.self_nothing_pixel_left = left_x
                self.self_nothing_pixel_right = right_x
                self.nothing_flag = True
        
        if len(left_x) == 0:
            left_x = np.round(self.self_img_x / 2)
            left_y = right_y
        elif len(right_x) == 0:
            right_x = np.round(self.self_img_x / 2)
            right_y = left_y

        # Polynomial fitting
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        plot_y = np.linspace(0, binary_line.shape[0] - 1, binary_line.shape[0])
        
        left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        
        center_fit_x = (left_fit_x + right_fit_x) / 2

        center = np.array([np.int32(center_fit_x), np.int32(plot_y)]).T
        left = np.array([np.int32(left_fit_x), np.int32(plot_y)]).T
        right = np.array([np.int32(right_fit_x), np.int32(plot_y)]).T

        cv2.polylines(out_img, [left], False, (0, 0, 255), thickness=5)
        cv2.polylines(out_img, [right], False, (0, 0, 255), thickness=5)
        
        sliding_window_img = out_img
        
        return sliding_window_img, left, right, center, left_x, right_x, left_y, right_y

    def meter_per_pixel(self):
        # NOTE: 원본 코드의 world_warp_x 배열 정의가 불분명하지만 로직을 유지
        world_warp_x = np.array([307, 1616, 1000, 1000, 97, 1668]).astype(np.float32) 
        
        meter_x = np.sum(world_warp_x)**2 
        meter_per_pix_x = meter_x / self.self_img_x / self.self_img_x
        meter_per_pix_y = meter_x / self.self_img_y / self.self_img_y
        
        return meter_per_pix_x, meter_per_pix_y

    def calc_curve(self, left_x, left_y, right_x, right_y, meter_per_pix_x, meter_per_pix_y):
        y_eval = self.self_img_y - 1
        
        left_fit_cr = np.polyfit(left_y * meter_per_pix_y, left_x * meter_per_pix_x, 2)
        right_fit_cr = np.polyfit(right_y * meter_per_pix_y, right_x * meter_per_pix_x, 2)
        
        left_curve_radius = (
            (1 + (2 * left_fit_cr[0] * y_eval * meter_per_pix_y + left_fit_cr[1])**2)**1.5
            ) / np.absolute(2 * left_fit_cr[0])
            
        right_curve_radius = (
            (1 + (2 * right_fit_cr[0] * y_eval * meter_per_pix_y + right_fit_cr[1])**2)**1.5
            ) / np.absolute(2 * right_fit_cr[0])
            
        return left_curve_radius, right_curve_radius

    def calc_vehicle_offset(self, sliding_window_img, left_x, left_y, right_x, right_y):
        # 픽셀 좌표에서의 다항식 피팅
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        
        # 이미지 맨 아래(차량 위치)의 y값
        bottom_y = sliding_window_img.shape[0] - 1
        
        # 이미지 맨 아래 y값에서의 좌/우 차선 x좌표 계산
        bottom_x_left = left_fit[0] * bottom_y**2 + left_fit[1] * bottom_y + left_fit[2]
        bottom_x_right = right_fit[0] * bottom_y**2 + right_fit[1] * bottom_y + right_fit[2]
        
        # 차선 중앙 x좌표
        lane_center_x = (bottom_x_left + bottom_x_right) / 2
        
        # 이미지 중앙 x좌표
        image_center_x = sliding_window_img.shape[1] / 2
        
        # 픽셀 오프셋
        vehicle_offset_pix = image_center_x - lane_center_x
        
        # 미터/픽셀 변환
        meter_per_pix_x, _ = self.meter_per_pixel()
        vehicle_offset = vehicle_offset_pix * meter_per_pix_x
        
        return vehicle_offset

    def img_CB(self, data):
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        self.self_nwindows = 10
        self.self_img_x = img.shape[1]
        self.self_img_y = img.shape[0]
        self.self_window_height = np.int(img.shape[0] / self.self_nwindows)
        
        blend_color = self.detect_color_self(img)
        blend_line = self.img_warp_self(img, blend_color)
        binary_line = self.img_binary(blend_line)

        if self.nothing_flag == False:
            self.detect_nothing()
            self.nothing_flag = True
            
        sliding_window_img, left, right, center, left_x, right_x, left_y, right_y = self.window_search(binary_line)

        meter_per_pix_x, meter_per_pix_y = self.meter_per_pixel()
        
        left_curve_radius, right_curve_radius = self.calc_curve(
            left_x, left_y, right_x, right_y, 
            meter_per_pix_x, meter_per_pix_y)
            
        vehicle_offset = self.calc_vehicle_offset(
            sliding_window_img, left_x, left_y, right_x, right_y)
            
        # Debug/Print information
        os.system('clear')
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
        print("vehicle_offset:", vehicle_offset) # 추가된 출력
        print("------------------------------")
        
        sliding_window_msg = self.bridge.cv2_to_compressed_imgmsg(sliding_window_img)
        self.pub.publish(sliding_window_msg)
        
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("sliding_window_img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("sliding_window_img", sliding_window_img)
        cv2.waitKey(1)

if __name__ == '__main__':
    calc_vehicle_offset = Calc_Vehicle_Offset()
    rospy.spin()