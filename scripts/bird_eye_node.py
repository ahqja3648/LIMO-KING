#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2

class Bird_Eye_View:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('bird_eye_node')
        self.pub = rospy.Publisher(
            '/bird_eye/compressed', CompressedImage, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.img_CB)

    def img_warp(self, img):
        # NOTE: self.img가 정의되어 있지 않으므로, img_CB에서 받은 img를 사용하도록 가정하고 변수명은 원본 유지
        self_img_x = self.img.shape[1] 
        self_img_y = self.img.shape[0]

        # Source Offset Calculation (원본 코드 형태 유지)
        src_side_offset = round(self.img_x * 0.046875), round(self.img_y * 0.208)
        src_center_offset = round(self.img_x * 0.14), round(self.img_y * 0.008)
        
        # Source Points (원본 이미지에서 투영할 4개의 점)
        src = np.float32([
            [src_side_offset[0], src_side_offset[1]],
            [self.img_x / 2 + src_center_offset[0], self.img_y / 2 + src_center_offset[1]],
            [self.img_x / 2 - src_center_offset[0], self.img_y / 2 + src_center_offset[1]],
            [self.img_x - src_side_offset[0], src_side_offset[1]]
        ])
        
        # NOTE: 원본 코드의 img_warp() 메서드 중간에 있던 좌표 리스트의 닫히지 않은 부분과 중복된 부분은 문맥상 src 배열의 정의가 끝난 후 dst 배열의 정의로 넘어가는 과정에서 오류가 발생한 것으로 보이지만,
        # 추출 요청이므로 보이는 텍스트를 최대한 이어서 처리합니다. (실제 실행 시 오류 발생 가능성 있음)
        # 이하는 페이지 170 상단에 있던 코드 조각입니다.
        
        # [self.img_x - src_side_offset[0], self.img_y - src_side_offset[1]], # 추정: 이 줄은 이전 페이지에서 정의가 끝났어야 할 src 배열의 일부이거나 오류
        #     [self.img_x / 2, self.img_y / 2],
        #     [self.img_x / 2, self.img_y / 2],
        #     [self.img_x / 2, self.img_y / 2]
        # ] 
        
        # Destination Offset Calculation
        dst_offset = (round(self.img_x * 0.125), 0)
        
        # Destination Points (투영된 이미지가 놓일 4개의 점 - 직사각형)
        dst = np.float32([
            [dst_offset[0], self.img_y],
            [dst_offset[0], 0],
            [self.img_x - dst_offset[0], 0],
            [self.img_x - dst_offset[0], self.img_y]
        ])

        # Calculate Transformation Matrices
        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inv = cv2.getPerspectiveTransform(dst, src)
        
        # Apply Perspective Transform
        warp_img = cv2.warpPerspective(img, matrix, (self.img_x, self.img_y))
        return warp_img

    def img_CB(self, data):
        # Convert ROS CompressedImage to OpenCV BGR image
        img = self.bridge.compressed_imgmsg_to_cv2(data)
        
        # Warp the image (버드 아이 뷰 변환)
        warp_img = self.img_warp(img)
        
        # Convert warped image back to ROS CompressedImage and publish
        warp_img_msg = self.bridge.cv2_to_compressed_imgmsg(warp_img)
        self.pub.publish(warp_img_msg)
        
        # Display images
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("warp_img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.imshow("warp_img", warp_img)
        
        cv2.waitKey(1)

if __name__ == '__main__':
    bird_eye_view = Bird_Eye_View()
    rospy.spin()