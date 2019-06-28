from keras.preprocessing import image
import keras
from keras.models import load_model
import numpy as np


class LoadCnn:
    def __init__(self, name):
        self.pers_name = name
        filname = (str(name)+".h5")
        self.classifier = load_model(filname)
        self.pos_label = None

    def make_prediction(self):
        img1 = image.load_img('temimg.jpg', target_size=(64, 64))
        img1 = image.img_to_array(img1)
        img1 = np.expand_dims(img1, axis=0)
        result = self.classifier.predict(img1)
        self.pos_label = None
        with open(str(self.pers_name+"poslabel.txt"), "r") as x:
            self.pos_label = int(x.read())
        if result[0][0] == self.pos_label:
            return True
        else:
            return False
