import numpy as np
import cv2 as cv

model_path = "D:/individual project/opencv/DNN_Face_Recognition/models/yunet.onnx"
faceDetector = cv.FaceDetectorYN.create(model_path, "", input_size=(640, 480))

capture = cv.VideoCapture(1, cv.CAP_DSHOW)
capture.set(3, 640) # 设置摄像头的帧的高为640
capture.set(4, 480) # 设置摄像头的帧的宽为480

while True:
    ret, frame = capture.read()
    if ret is not True:
        break

    faces = faceDetector.detect(frame)
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame,(coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (255, 0, 0), thickness=2)

        cv.imshow('photo-collection-demo', frame)
        c = cv.waitKey(1)
        # 按esc退出视频
        if c == 27:
            break
        elif c == 32:  # 按空格键可输入人脸ID并保存视频帧图片
            face_id = input('请输入人脸ID，按回车键后系统自动保存视频帧图片==>  ')  # 对于每一个人，输入一个数字作为人脸IP
            print('ID输入成功，正在保存帧照片......')
            cv.imwrite(r"D:/individual project/OpenCV/DNN_Face_Recognition/images/" + face_id + ".jpg", frame)
            print('照片保存成功！\n按esc可退出视频\n按空格键可继续保存帧图片\n')
    else:
        cv.imshow('photo-collection-demo', frame)
        c = cv.waitKey(1)
        # 按esc退出视频
        if c == 27:
            break
        elif c == 32:
            print('当前未检测到人脸，无法保存视频帧图片')