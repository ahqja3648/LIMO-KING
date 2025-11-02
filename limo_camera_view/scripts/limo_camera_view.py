import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
def image_callback(msg):
    # 1) ROS 메시지로 확인
    w, h = msg.width, msg.height
    rospy.loginfo(f"ROS Image size: {w}x{h}")

    bridge = CvBridge() #ROS 이미지 메세지(sensor_msgs/Image) 를 (numpy array)로 바꿔줌

    # color 모드
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8") #ROS에서 받은 image를 RGB 형태의 OpenCV 이미지로 변환
    cv2.imshow("Camera View", cv_image)
    # 2) OpenCV 배열로 확인
    cv2.waitKey(1)
    H, W = cv_image.shape[:2]
    rospy.loginfo(f"OpenCV Image size: {W}x{H}")

    #터미널에 rostopic echo -n1 /camera/color/image_raw | grep -E "width|height" 로 가로 세로 확인


def main():
    rospy.init_node("camera_viewer")
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
