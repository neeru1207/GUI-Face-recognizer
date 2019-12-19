# GUI Face recognizer
A GUI based webcam realtime video Face Recognizer coded in Python3 for Windows. 
# Screenshots
!["Main UI"](https://github.com/neeru1207/GUI-Face-recognizer/blob/master/Face_Recognizer/MainUI.png)
!["Face Recognition"](https://github.com/neeru1207/GUI-Face-recognizer/blob/master/Face_Recognizer/Face_Recognizer.png)
## Usage:
### Required download:
* Download the faces.tar file from [here](http://www.vision.caltech.edu/Image_Datasets/faces).
* Extract to a folder named faces and move the folder into the Face Recognizer folder.
### Required Packages:
Install [Python3](https://www.python.org/downloads/) and [Anaconda](https://www.anaconda.com/distribution/) if you haven't.
* Create a new conda environment
````shell
conda create --name envname
````
* Activate the environment
````shell
conda activate envname
````
* Install Keras, OpenCV, Numpy, h5py and PIL:
````shell
conda install keras
conda install opencv
conda install numpy
conda install h5py
conda install pillow
````
### Running the application
````shell
cd Face Recognizer
python main_ui.py
````
## Implementation:
### Dataset
#### "NO" class:
* The [faces](http://www.vision.caltech.edu/Image_Datasets/faces) dataset forms the __NO__ class of the dataset.
#### "YES" class:
* [Frontal face Haar cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) is used for webcam based face detection in OpenCV.
* The detected face in every frame of the video stream is cropped and saved as an image.
* This forms the "YES" class of the dataset
#### Augmentation:
* Since the dataset is rather small, __Keras Image data generator__ is used for Image augmentation.
### Model:
* A __Convolutional Neural Network__ is built using Keras. This model is trained on the constructed dataset over __25 epochs__ with __400__ samples per epoch.
* The loss function used is __Binary crossentropy__ and the optimizer used is __adam__.
* The model is saved after training into a __.h5__ file for later use.
* The class indices are also saved into a __poslabel.txt__ file.
### Recognition:
* For every frame in the video stream, face is detected using [frontal_face_haar_cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) in Opencv.
* The detected face is saved to a __tmpimg.jpg__ file.
* Th trained CNN model is used to recognize the face.
* The result is shown using __cv.rectangle__ around the face and text.
* Green colored rectangle and Name text if __YES__ else Red colored rectangle and Not Name text.

