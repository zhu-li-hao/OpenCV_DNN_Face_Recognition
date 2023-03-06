import numpy as np
import cv2 as cv
import os

face_detection_model = "D:/individual project/opencv/DNN_Face_Recognition/models/yunet.onnx"
face_recognition_model = "D:/individual project/opencv/DNN_Face_Recognition/models/face_recognizer_fast.onnx"
images_path = "D:/individual project/OpenCV/DNN_Face_Recognition/images"

faceDetector = cv.FaceDetectorYN.create(face_detection_model, "", input_size=(640, 480)) # 初始化FaceRecognizerYN
recognizer = cv.FaceRecognizerSF.create(face_recognition_model, "") # 初始化FaceRecognizerSF

cosine_similarity_threshold = 0.363
l2_similarity_threshold = 1.128

# 获取图片中的人脸特征,将获取到的人脸特征和ID分别添加到不同的列表中
def Gets_Facial_Features(images_path):
    images_feature_list = []
    ID_list = []
    for image_name in os.listdir(images_path):
        ID_list.append(image_name.split('.')[0])
        image = cv.imread(images_path + '/' + image_name)
        faces1 = faceDetector.detect(image)
        # 在人脸检测部分的基础上, 对齐检测到的首个人脸(faces[1][0]), 保存至aligned_face。
        aligned_face1 = recognizer.alignCrop(image, faces1[1][0])
        # 在上文的基础上, 获取对齐人脸的特征feature。
        image_feature = recognizer.feature(aligned_face1)
        images_feature_list.append(image_feature)
    return ID_list, images_feature_list

if __name__ == '__main__':
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    capture.set(3, 640)  # 设置摄像头的帧的高为640
    capture.set(4, 480)  # 设置摄像头的帧的宽为480

    ID_list, images_feature_list = Gets_Facial_Features(images_path)
    while True:
        ret, frame = capture.read()
        if ret is not True:
            print('摄像头未打开')
            break

        frame_faces = faceDetector.detect(frame)

        if frame_faces[1] is not None:
            for idx, face in enumerate(frame_faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(frame, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), thickness=2)
                cv.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness=2)
                cv.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness=2)
                cv.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness=2)
                cv.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness=2)
                cv.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness=2)

                aligned_face = recognizer.alignCrop(frame, frame_faces[1][idx])
                frame_feature = recognizer.feature(aligned_face)
                for index, image_feature in enumerate(images_feature_list):
                    cosine_score = recognizer.match(frame_feature, image_feature, 0)
                    l2_score = recognizer.match(frame_feature, image_feature, 1)
                    if (cosine_score >= cosine_similarity_threshold):
                        cv.putText(frame, ID_list[index], (coords[0] + 5, coords[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        i = 1
                        break
                    elif (l2_score <= l2_similarity_threshold):
                        cv.putText(frame, ID_list[index], (coords[0] + 5, coords[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        i = 1
                        break
                    else:
                        cv.putText(frame, 'unkown', (coords[0] + 5, coords[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (0, 255, 0), 2)

            cv.imshow('photo-collection-demo', frame)
            c = cv.waitKey(1)
            # 按esc退出视频
            if c == 27:
                break

        else:
            cv.putText(frame, 'face is not detected', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.imshow('photo-collection-demo', frame)
            c = cv.waitKey(1)
            # 按esc退出视频
            if c == 27:
                break