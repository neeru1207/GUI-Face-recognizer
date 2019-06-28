import time
import cv2
import os
import shutil
'''Opens the webcam, then:
Repeatedly:
    Detects a face from the web cam video stream.
    Draws a rectangle around the face.
    If k is pressed:
        Saves the rectangular part as an image.
    If q is pressed:
        Closes the webcam and terminates.
The saved images form the positive data in the data set for face recognition.
'''
class BuildPositiveFaceDataset:
    def __init__(self):
        self.num_of_images = 0
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.name = None
        self.path = None

    def set_name(self, name):
        self.name = name
        self.path = "dataset/train/" + self.name

    def start_capture(self):
        try:
            os.makedirs(self.path)
        except:
            return
        vid = cv2.VideoCapture(0)
        while True:
            ret, img = vid.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = self.detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
                cv2.putText(img, str(str(self.num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
                new_img = img[y:y+h, x:x+w]
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("k"):
                self.num_of_images += 1
                cv2.imwrite(str(self.path+"/"+str(self.num_of_images)+".jpg"), new_img)
            elif key == ord("q") or key == 27:
                break
        cv2.destroyAllWindows()

    def reset(self):
        self.num_of_images = 0
        try:
            shutil.rmtree(self.path)
        except:
            pass