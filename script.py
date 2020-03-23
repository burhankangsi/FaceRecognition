import cv2
import numpy as np

def generate_dataset(img, user_id, img_id):
    cv2.imwrite("my_data/user." + str(user_id) +"." + str(img_id) + ".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighors, color, text, classify):
    #first we convert our image into grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighors)
    coordinates = []
    for(x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, _ = classify.predict(gray_img[y:y+h, x:x+w])
        if id == 1:
            cv2.putText(img, "Burhan", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coordinates = (x,y,w,h)

    return  coordinates


# def draw_boundary(img, classifier, scaleFactor, minNeighors, color, text):
#     #first we convert our image into grayscale image
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighors)
#     coordinates = []
#     for(x, y, w, h) in features:
#         cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#     #    id, _ = classify.predict(gray_img[y:y+h, x:x+w])
#      #   if id == 1:
#         cv2.putText(img, "Burhan", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
#         coordinates = (x,y,w,h)
#
#     return  coordinates

def recognize(img, classify, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coordinates = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", classify)
    return img

def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id):
    color = {"blue":(255, 0, 0), "red":(0, 0, 255), "green":(0, 255, 0), "white":(255, 255, 255)}

    coordinates = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coordinates) == 4:
        roi_img = img[coordinates[1] : coordinates[1] + coordinates[3], coordinates[0]: coordinates[0] + coordinates[2]]

        users_id = 1
        generate_dataset(roi_img, users_id, img_id)
        # coordinates = draw_boundary(roi_img, eyesCascade, 1.1, 14, color['red'], "Eyes")
        # coordinates = draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        # coordinates = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
        # These are most accurate scaling factors after all the testing

    return  img

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

classify = cv2.face.LBPHFaceRecognizer_create()
classify.read("classifier.yml")


video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    _, img = video_capture.read()
    # Capture frame-by-frame
   # img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade, img_id)
    img = recognize(img, classify, faceCascade)

    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
