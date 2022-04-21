import cv2 as cv 
import os
import numpy as np

dir = 'C:\\Users\\rylep\\OneDrive\\Documents\\zlepsFile\\Python\\face-recognition\\python_project\\Photos'
haar_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")
people = []
faces = [] #this is the storage of list of faces...
labels = [] #who owns that face..

for i in os.listdir(dir):
    people.append(i)

for person in people:
    path = os.path.join(dir, person)
    label = people.index(person)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv.imread(img_path)
        imGray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
        face_detect = haar_cascade.detectMultiScale(imGray, 1.2, 6)

        for(x1, y1, x2, y2) in face_detect:
            faceRegi = imGray[y1:y1+y2, x1:x1+x2]
            faces.append(faceRegi)
            labels.append(label)

faces = np.array(faces, dtype="object")
labels = np.array(labels)

#Reference of Face Recognition
#Local Binary Pattern Histogram
faceRecog = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer using faces and labels
faceRecog.train(faces, labels)

np.save('faces.npy', faces)
np.save('labels.npy', labels)

faceRecog.save('faceTrain.yml')