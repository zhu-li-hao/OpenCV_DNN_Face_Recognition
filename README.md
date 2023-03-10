# OpenCV_DNN_Face_Recognition
这是一个基于OpenCV（4.5.4版本及以上）的深度学习神经网络人脸模块（OpenCV DNN Face）的实时人脸识别程序。

编写这个程序的初衷是大二时跟着实验室的师兄接了个毕业设计的私活儿，毕设题目就是利用OpenCV进行人脸识别。

在OpenCV DNN Face模块推出之前，OpenCV实现人脸检测多数都是使用haar人脸特征检测，利用cv2.CascadeClassifier加载官方haar级联分类器即可实现人脸、眼睛、嘴部等的检测。这种方式操作简便快捷，但是在逆光、人脸侧对镜头等情况下，人脸检测效果大打折扣，检测准确度较低。  

OpenCV4.5.4更新后，收录了一个基于深度学习神经网络的人脸模块，里面包含了人脸检测模型YuNet和人脸识别模型SFace。
在YuNet模型的加持下，OpenCV人脸检测准确度大幅度提升，在弱逆光、人脸侧对镜头的情况下也能准确检测出人脸，检测效率高，效果稳定。此外人脸识别模型SFace可以实现人脸特征提取和特征对比。
两个模型配合使用即可实现高质量的人脸检测。  

本项目包含sample_collection.py和face_recognition.py两个程序，分别用于人脸样本采集和人脸识别。
images文件夹用于保存sample_collection.py运行后采集的人脸样本图片。  

得益于SFace模型强大的人脸特征提取能力，本程序只需采集一张人脸样本图片即可实现人脸识别！！！

在使用本项目的程序时，记得把程序中的文件路径换成你PC中的实际文件路径！！！

参考资料
https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html  
https://zhuanlan.zhihu.com/p/423625566  
https://github.com/opencv/opencv/tree/master/samples/dnn  
https://github.com/opencv/opencv/tree/master/modules/objdetect
