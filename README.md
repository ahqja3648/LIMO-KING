# Test code
camera view

## Bring up

``` bash
#limo 기본 bringup 실행
roslaunch limo_bringup limo_start.launch
```

## Build from source code

```bash
# package 생성
cd ~/limo_ws/src
catkin_create_pkg camera_viewer roscpp rospy std_msgs sensor_msgs cv_bridge 

# 파일 넣기 

# 실행 권한 python만
chmod +x ~/limo_ws/src/camera_viewer/scripts/limo_camera_view.py

# 빌드
cd ~/limo_ws
catkin_make
source devel/setup.bash

# 실행
roslaunch camera_viewer limo_camera_view.launch
```

## Check wide and height

```bash
rostopic echo -n1 /camera/color/image_raw | grep -E "width|height"
```
