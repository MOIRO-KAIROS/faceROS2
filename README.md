# moiro_vision using Adaface and YOLOv8
#### This repo is designed to implement real-time face recognition in ROS2
##### [ Ubuntu 22.04 ROS2 humble ver. ]
## 0. YOLOv8_ros
https://github.com/mgonzs13/yolov8_ros
- Use for Person detection
  
## 0. Adaface
https://github.com/mk-minchul/AdaFace
- Use for Face Recognition

## 1. Setup
### 1) Install nvidia driver
#### 참고: https://webnautes.tistory.com/1844 # 레퍼런스

-  Architecture 확인
    ```
    uname -a # x86_64

    sudo apt-get update && sudo apt-get upgrade
    ```
- nvidia driver 설치

    ```
    apt --installed list | grep nvidia-driver # 설치할 수 있는 드라이버 버전을 확인
    sudo apt-get install nvidia-driver-525 # sudo apt install는 옛날버전이므로 X
    sudo reboot
    ```
- 잘 설치했는지 확인
    ```
    nvidia-smi
    sudo apt-get update && sudo apt-get upgrade
    ```
- cuda 설치 (11.8 ver)
    ```
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
    > 참고 페이지 (https://webnautes.tistory.com/1844)에서 터미널 설정 확인: Continue / accept / Driver 해제 / Install

- 환경변수 추가
    ```
    vim ~/.bashrc
    ```
    > export PATH="/usr/local/cuda-11.8/bin:$PATH"
    
    > export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

    ```
    source ~/.bashrc
    ```
- CUDA 잘 설치했는지 확인
    ```
    nvcc --version
    ```

### 2) 가상환경 설치 (pytorch 설치)
```
python3 -m venv py310
source py310/bin/activate
```
```
# When not using requirement.txt

pip install pyyaml
pip install typeguard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-image matplotlib pandas scikit-learn
pip install pytorch pytorch-lightning==1.8.6
pip install tqdm bcolz-zipline prettytable menpo mxnet opencv-python
pip install -U colcon-common-extensions
```
```
pip list | grep torch
```
> pytorch-lightning         1.8.6                    pypi_0    pypi

> torch                     2.2.2+cu118              pypi_0    pypi

> torchaudio                2.2.2+cu118              pypi_0    pypi

> torchmetrics              1.3.2                    pypi_0    pypi

> torchvision               0.17.2+cu118             pypi_0    pypi


### 3) Installation
- (주의) workspace 이름을 moiro_ws로 준수할 것
```
  cd ~/moiro_ws/src
  git clone https://github.com/MOIRO-KAIROS/moiro_vision.git
  pip3 install -r faceRec_ros2/requirements.txt
  cd ~/moiro_ws
  colcon build
```

### 4) Before RUN
#### (1)  ```pretrained``` 폴더 생성 후, weight(.ckpt) 다운로드
- 다운로드 (링크 클릭)
    | Arch | Dataset    | Link                                                                                         |
    |------|------------|----------------------------------------------------------------------------------------------|
    | R50  | MS1MV2     | [gdrive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing) |

- 파일 구조 (추가해야하는 pretrained와 video 부분 폴더만 깊게 표기)
    ```
    faceRec_ros2
    ├── adaface
    │   ├── adaface
    │   │   ├── adaface_ros2.py
    │   │   ├── __init__.py
    │   │   └── script
    │   │       ├── adaface.py
    │   │       ├── embed
    │   │       ├── face_alignment
    │   │       ├── face_dataset
    │   │       ├── __init__.py
    │   │       ├── LICENSE
    │   │       ├── main.py
    │   │       ├── net.py
    │   │       ├── pretrained
    │   │       ├── __pycache__
    │   │       ├── README.md
    │   │       ├── requirements.txt
    │   │       ├── scripts
    │   │       └── utils.py
    │   ├── launch
    │   ├── package.xml
    │   ├── resource
    │   ├── setup.cfg
    │   ├── setup.py
    │   └── test   
    │    
    ├── README.md
    └── yolov8_ros

    ```
### 5) Store Embed
#### (1) ```face_dataset``` 저장

- 아래와 같이 원하는 그룹의 이름으로 폴더를 만든 후 embed, images 폴더를 생성
- images 폴더에 촬영을 원하는 사람들의 이미지를 저장

```
└── face_dataset
   └── IVE
      ├── embed
      └── images
          ├── fall.jpg
          ├── iseo.jpg
          ├── lay.JPG
          ├── liz.jpeg
          ├── won.jpg
          └── yujin.jpg

```
#### (2) ```embed``` 폴더에 ids.pt와 features.pt 생성 
- (1)에서 만든 그룹이름과 동일하게 dataset option을 적용
```
  cd ~/moiro_ws/
  python3 src/moiro_vision/adaface_ros/adaface_ros/script/main.py --option 0 --dataset <group_name>
```
> desired output
```
face recongnition model loaded
얼굴 임베딩 벡터 저장 완료(known face 개수:<# of face>)
```

### 6) (OPTION) Run Demo File (.mp4)

1. video/iAm.zip 압축풀기 > iAm.mp4
    - 파일 구조
    ```
    video
      |
      |_____ iAm.mp4
    ```
2. (1)번 터미널
```ros2 run adaface_ros video_publisher```
(2)번 터미널
```ros2 launch adaface_ros adaface.launch.py video:=1```

## 3. Usage
```
ros2 launch adaface adaface.launch.py person_name:=<wanted_target_person>
```
> Debug Image

<p align="center">
  <img src="https://github.com/MOIRO-KAIROS/faceRec_ros2/assets/114575723/d955b345-c5c8-4efe-b6c2-4ccd349e8470" alt="Debug Image">
</p>

> RQT graph

<p align="center">
  <img src="https://github.com/MOIRO-KAIROS/moiro_vision/assets/114575723/9615cddf-a171-46ab-9847-08cf1c5c7ef2" alt="rosgraph">
</p>


#### (주의) Debug image까지는 잘 보이지만, target의 coordinate를 받기 위해서는 world_node의 service client와 'base_plate'라는 link도 활성화되어야한다.
##### (참고) https://github.com/MOIRO-KAIROS/moiro_ws.git
