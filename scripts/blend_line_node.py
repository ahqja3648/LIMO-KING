#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2

class Blend_line_detect:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('blend_line_node')
        self.pub = rospy.Publisher(
            '/blend_line/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)

    def detect_color(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Yellow color detection range
        yellow_lower = np.array([15, 80, 0])
        yellow_upper = np.array([45, 255, 255])
        
        # White color detection range
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([179, 64, 255])
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine yellow and white masks
        blend_mask = cv2.bitwise_or(yellow_mask, white_mask)
        # Apply the mask to the original image (or create a colored mask image)
        blend_color = cv2.bitwise_and(img, img, mask=blend_mask)
        
        return blend_color

    def img_warp_all(self, img, blend_color):
        self_img_x = img.shape[1]
        self_img_y = img.shape[0]

        # Source offset calculation (Perspective Transform)
        src_side_offset = round(self_img_x * 0.046875), round(self_img_y * 0.08)
        src_center_offset = round(self_img_x * 0.14), round(self_img_y * 0.08)

        # Source Points (4 points from the original image)
        # NOTE: The order might need checking, but this follows the logic from the image fragments.
        src = np.float32([
            [src_side_offset[0], src_side_offset[1]],
            [self_img_x / 2 - src_center_offset[0], self_img_y - src_center_offset[1]],
            [self_img_x / 2 + src_center_offset[0], self_img_y - src_center_offset[1]],
            [self_img_x - src_side_offset[0], src_side_offset[1]]
        ])

        # Destination offset calculation (Bird's Eye View rectangular output)
        dst_offset_x = round(self_img_x * 0.125)
        
        # Destination Points (4 points for the warped image - rectangular)
        dst = np.float32([
            [dst_offset_x, 0],
            [self_img_x - dst_offset_x, 0],
            [self_img_x - dst_offset_x, self_img_y],
            [dst_offset_x, self_img_y]
        ])

        # Calculate Transformation Matrices
        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        
        # Apply Perspective Transform to the colored line detection image
        blend_line_warp = cv2.warpPerspective(blend_color, matrix, (self_img_x, self_img_y))
        
        return blend_line_warp

    def img_CB(self, data):
        # Convert ROS CompressedImage to OpenCV BGR image
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        
        # Detect lines by color (yellow and white)
        blend_color = self.detect_color(img)
        
        # Warp the color-detected image (Perspective Transform)
        blend_line_warp = self.img_warp_all(img, blend_color)
        
        # Convert warped image back to ROS CompressedImage and publish
        blend_line_msg = self.bridge.cv2_to_compressed_imgmsg(blend_line_warp)
        self.pub.publish(blend_line_msg)
        
        # Display images
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("blend_line_warp", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("blend_line_warp", blend_line_warp)
        cv2.waitKey(1)

if __name__ == '__main__':
    blend_line_detect = Blend_line_detect()
    rospy.spin()