#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2

class Binary_line:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('binary_line_node')
        self.pub = rospy.Publisher(
            '/binary/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)

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

        # Source Offset Calculation (Perspective Transform)
        src_side_offset = round(self_img_x * 0.046875), round(self_img_y * 0.208)
        src_center_offset = round(self_img_x * 0.14), round(self_img_y * 0.008)

        # Source Points (4 points from the original image)
        src = np.float32([
            [src_side_offset[0], src_side_offset[1]],
            [self_img_x / 2 - src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x / 2 + src_center_offset[0], self_img_y / 2 + src_center_offset[1]],
            [self_img_x - src_side_offset[0], src_side_offset[1]]
        ])

        # NOTE: 원본 코드의 중간에 있던 불필요한/중복된 좌표 리스트는 제거하고 다음 논리로 넘어갑니다.
        
        # Destination Offset Calculation
        dst_offset = (round(self_img_x * 0.125), 0)
        
        # Destination Points (4 points for the warped image - rectangular)
        dst = np.float32([
            [dst_offset[0], self_img_y],
            [dst_offset[0], 0],
            [self_img_x - dst_offset[0], 0],
            [self_img_x - dst_offset[0], self_img_y]
        ])

        # Calculate Transformation Matrices
        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        
        # Apply Perspective Transform to the colored line detection image
        blend_line = cv2.warpPerspective(blend_color, matrix, (self_img_x, self_img_y))
        
        return blend_line

    def img_binary(self, blend_line):
        # Convert color-blended warped image to Grayscale
        bin = cv2.cvtColor(blend_line, cv2.COLOR_BGR2GRAY)
        
        # Create a binary image mask (initialized to zeros)
        binary_line = np.zeros_like(bin)
        
        # Apply thresholding: pixels with intensity > 0 are set to 1
        binary_line[bin > 0] = 1
        
        return binary_line

    def img_CB(self, data):
        # Convert ROS CompressedImage to OpenCV BGR image
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        
        # 1. Color Detection (Yellow and White Lines)
        blend_color = self.detect_color_self(img)
        
        # 2. Perspective Transform (Warp)
        blend_line = self.img_warp_self(img, blend_color)
        
        # 3. Binary Conversion
        binary_line = self.img_binary(blend_line)
        
        # Convert binary image back to ROS CompressedImage and publish
        binary_line_msg = self.bridge.cv2_to_compressed_imgmsg(binary_line)
        self.pub.publish(binary_line_msg)
        
        # Display images
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("binary_line", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("binary_line", binary_line)
        cv2.waitKey(1)

if __name__ == '__main__':
    binary_line_detect = Binary_line()
    rospy.spin()