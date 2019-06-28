'''Download the faces.tar file from http://www.vision.caltech.edu/Image_Datasets/faces
Extract to a folder named faces and copy the folder into FaceRecognizer directory
This script processes the faces's images to form the negative part of the dataset
'''
import os
import cv2
import shutil

class BuildNegativeDataset:
    def __init__(self):
        self.src_path = "faces"
        self.target_path = "dataset/train/No"
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.imgnum = 1

    def create_dataset(self):
        try:
            os.makedirs(self.target_path)
        except:
            return
        for r, d, f in os.walk(self.src_path):
            for file in f:
                if '.jpg' in file:
                    img = cv2.imread(str(self.src_path+"/"+str(file)))
                    print("Read "+self.src_path+"/"+str(file))
                    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face = self.detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
                    for x, y, w, h in face:
                        new_img = img[y:y+h, x:x+w]
                        cv2.imwrite(str(self.target_path+"/"+str(self.imgnum)+".jpg"), new_img)
                        self.imgnum += 1
    def reset(self):
        self.imgnum = 1
        shutil.rmtree(self.target_path)
