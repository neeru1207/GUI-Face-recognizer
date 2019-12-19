import tensorflow as tf
import cv2
import numpy as np

class CNN:
    def __init__(self):
        self.classifier = tf.keras.models.Sequential()
        self.classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        self.classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(tf.keras.layers.Flatten())
        self.classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
        self.test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        self.training_set = None
        self.test_set = None
        self.predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.pers_name = None

    def create_train_test(self):
        self.training_set = self.train_datagen.flow_from_directory('dataset/train/',
                                                        classes=["No", self.pers_name],
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='binary')
        self.label_map = (self.training_set.class_indices)
        self.pos_class_label = self.label_map[self.pers_name]
        self.test_set = self.test_datagen.flow_from_directory('dataset/test/',classes=["No", self.pers_name],
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
        with open(str(self.pers_name+"poslabel.txt"), "w") as x:
            x.write(str(self.pos_class_label))

    def compile(self):
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def set_name(self, name):
        self.pers_name = name

    def fit_generate(self):
        self.classifier.fit_generator(self.training_set,
                         samples_per_epoch=400,
                         nb_epoch=25,
                         validation_data=self.test_set,
                         nb_val_samples=100)

    def make_prediction(self):
        img1 = tf.keras.preprocessing.image.load_img('temimg.jpg', target_size=(64, 64))
        img1 = tf.keras.preprocessing.image.img_to_array(img1)
        img1 = cv2.resize(img1, (64, 64))
        img1 = np.expand_dims(img1, axis=0)
        result = self.classifier.predict(img1)
        if result[0][0] == self.pos_class_label:
            return True
        else:
            return False

