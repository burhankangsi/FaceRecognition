import numpy as np
import cv2
from PIL import Image
import os

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
  #  print(dir(cv2.face))
    classifier = cv2.face.LBPHFaceRecognizer_create()
    #classifier = cv2.face.EigenFaceRecognizer_create()
    classifier.train(faces, ids)
    classifier.write("classifier.yml")

train_classifier("my_data")
