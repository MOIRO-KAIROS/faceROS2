import net
import torch
import os
from face_alignment import align
import numpy as np
import cv2
from PIL import Image

import sys
import time

# sys_path = os.path.dirname(__file__)
sys_path = "/home/minha/moiro_ws/src/faceROS2/adaface/adaface/script"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False) # for 메모리

adaface_models = {
    'ir_50': os.path.join(sys_path, "pretrained/adaface_ir50_ms1mv2.ckpt"),
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image, RGB=False):
    np_img = np.array(pil_rgb_image)
    if RGB:
        brg_img = ((np_img / 255.) - 0.5) / 0.5 # rgb 기준 ..?
    else:
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5 # bgr 기준 ..?
    tensor = torch.tensor(np.expand_dims(brg_img.transpose(2,0,1),axis=0)).float()
    return tensor

def load_dataset():
    # 데이터 로드
    database_dir = os.path.join(sys_path, 'embed/features.pt')
    if (os.path.exists(database_dir)):
        known_face_encodings = torch.load(database_dir).to(device)
        known_face_names = torch.load(os.path.join(sys_path, 'embed/ids.pt'))
        print("knwon face list: ", len(known_face_names))
        return known_face_encodings, known_face_names
    else:
        print("Error: Face database not found")
        sys.exit(1)

class AdaFace():
    def __init__(self, **kwargs):
        #  model="ir_50", 
        #  option=1, 
        #  dataset="face_dataset/test",
        #  video="0",
        #  max_obj=6,
        #  thresh=0.2
        
        self.model = load_pretrained_model(kwargs["model"]) # .to(device)
        print('face recongnition model loaded')
        
        self.option = kwargs["option"]
        self.max_obj = kwargs["max_obj"]
        self.thresh = kwargs["thresh"]
        if self.option == 0:
            self.dataset = os.path.join(sys_path, kwargs["dataset"])
        elif self.option == 1:
            # 데이터 로드
            self.known_face_encodings, self.known_face_names = load_dataset()
            print('face encodings & names loaded')
            video = kwargs["video"]
            if (video.isnumeric()):
                self.video = int(video)
            else:
                self.video = os.path.join(sys_path, video)
        else:
            print("Error: 잘못된 argument 입력")
            sys.exit(1)

    def store_embedding(self):
        # 모든 얼굴의 임베딩 계산
        features = []
        ids = []
        for fname in sorted(os.listdir(self.dataset)):
            path = os.path.join(self.dataset, fname)
            aligned_rgb_img = align.get_aligned_face(path)
            bgr_tensor_input = to_input(aligned_rgb_img)
            feature, _ = self.model(bgr_tensor_input)

            # features[fname.split('.')[0]] = feature
            features.append(feature)
            ids.append(fname.split('.')[0])
        # print(features)
        
        if not os.path.exists('embed'):
            os.makedirs('embed')

        # Embeddings와 passage_ids를 저장
        features = torch.squeeze(torch.stack(features), dim=1)
        # print(features.shape)
        torch.save(features, os.path.join(sys_path, 'embed/features.pt'))
        torch.save(ids, os.path.join(sys_path, 'embed/ids.pt'))
        print(f"얼굴 임베딩 벡터 저장 완료(known face 개수: {len(ids)})")
        return features, ids

    def inference(self,
                  frame,
                  ):
        pil_im = Image.fromarray(frame).convert('RGB')
        face_encodings = []

        ## 1. 얼굴 feature 추출
        aligned_rgb_img, bboxes = align.get_aligned_face_for_webcam('', pil_im, self.max_obj)
        bboxes = [[int(xy) for (xy) in bbox] for bbox in bboxes]
        for img in aligned_rgb_img:
            bgr_tensor_input = to_input(img)
            face_encoding, _ = self.model(bgr_tensor_input)
            face_encodings.append(face_encoding)
        
        # if not face_encodings:
        #     continue
        face_names = []
        if len(face_encodings) > 0:
            ## 2. 얼굴 유사도 측정 with tensor
            # start_time = time.time() # 연산에 대한 실행 시간(start) check
            face_encodings = torch.squeeze(torch.stack(face_encodings), dim=1).to(device) # torch.squeeze(torch.stack(face_encodings), dim=1) # torch.squeeze()
            with torch.no_grad():
                face_distances = torch.matmul(face_encodings, self.known_face_encodings.T)
            best_match_index = torch.argmax(face_distances, dim=1)
            face_names = ["unknown" if torch.any(face_distances[i][idx] < self.thresh) else self.known_face_names[idx] for i, idx in enumerate(best_match_index)]
            # face_names = [known_face_names[idx] for idx in best_match_index] # threshold 없는 경우 ('unkown' 처리 안한 경우)
            # end_time = time.time() # 연산에 대한 실행 시간(end) check
            # print("Execution Time:", (end_time - start_time), "sec") # 실행 시간 0.0003 ~
        
        return bboxes, face_names

    def run_video(self): 
        video_capture = cv2.VideoCapture(self.video) # os.path.join(sys_path, "video/iAM.mp4"
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # height, width = frame.shape[:2]
            if not ret:
                print("Warning: no frame")
                break

            frame = cv2.flip(frame, 1)
            bboxes, face_names = self.inference(frame)

            ## 3. bbox 시각화
            if (len(face_names) > 0):
                for (x1, y1, x2, y2, _), f_name in zip(bboxes, face_names):
                    cv2.rectangle(frame,(x1, y1), (x2, y2),(0, 0, 255), 1)
                    cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), cv2.FILLED)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f_name, (x1 + 6, y2 - 6), font, .5, (0, 0, 0), 1)
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


# if __name__ == '__main__':
    