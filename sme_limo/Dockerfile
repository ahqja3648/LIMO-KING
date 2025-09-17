# Ubuntu 20.04 (focal)
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 기본 유틸, 로케일/타임존, GPG/lsb-release
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata locales ca-certificates curl gnupg2 lsb-release \
    bash-completion build-essential git wget \
 && locale-gen en_US.UTF-8 ko_KR.UTF-8 \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ROS 패키지 저장소 추가
RUN curl -sSL 'https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc' | apt-key add - \
 && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list

# ROS Noetic 설치 (desktop-full) + 자주 쓰는 툴
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# rosdep 초기화
# (컨테이너 최초 실행 시 네트워크 필요)
RUN rosdep init || true

# 비루트 사용자 추가 (권장)
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# 기본 catkin 워크스페이스 생성
RUN /bin/bash -lc "mkdir -p ~/catkin_ws/src"

# 셸 진입 시 ROS 자동 source
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc \
 && echo "source ~/catkin_ws/devel/setup.bash || true" >> ~/.bashrc \
 && echo "export ROS_DISTRO=noetic" >> ~/.bashrc

# 엔트리포인트: bash
CMD ["/bin/bash"]
