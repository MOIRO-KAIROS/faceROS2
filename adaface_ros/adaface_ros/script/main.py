import os
import argparse
import sys

sys_path = os.path.join(os.path.expanduser("~"), "moiro_ws/src/moiro_vision/adaface_ros/adaface_ros/script")

from adaface  import AdaFace

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=1, help='0: 임베딩 저장 1: webcam/video으로 run_video)')
    parser.add_argument('--video', type=str, default='1', help='0: webcam 또는 "video/iAM.mp4" 특정 비디오 path 경로')
    parser.add_argument('--model', type=str, default='ir_50', help='face recognition weight')
    parser.add_argument('--thresh', nargs='+', type=str, default=.2, help='face detection confidence < .2')
    parser.add_argument('--max_obj', type=int, default=6, help='detect 가능한 최대 얼굴의 개수')
    parser.add_argument('--dataset', type=str, default='face_dataset/test', help='face dataset의 경로 (known face dataset)')
    
    opt = parser.parse_args()
    
    adaface = AdaFace(
        model=opt.model,
        option=opt.option,
        dataset=opt.dataset,
        video=opt.video,
        max_obj=opt.max_obj,
        thresh=opt.thresh,
    )
    
    if opt.option == 0:
        adaface.store_embedding()
    elif opt.option == 1:
        adaface.run_video()
    else:
        print("Error: 잘못된 argument 입력")
        sys.exit(1)