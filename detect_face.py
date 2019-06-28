import cv2
import time


class DetectFace:
    def __init__(self, name, CNNinst):
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.name = name
        vid = cv2.VideoCapture(0)
        time.sleep(2)
        while True:
            ret, img = vid.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = self.detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                new_img = img[y:y + h, x:x + w]
                cv2.imwrite("temimg.jpg", new_img)
                if CNNinst.make_prediction():
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, self.name, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness = 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, str("Not "+self.name), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
            cv2.imshow("FaceRecognition", img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
