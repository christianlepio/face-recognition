import numpy as np
import cv2 as cv
import os

dir = 'C:\\Users\\rylep\\OneDrive\\Documents\\zlepsFile\\Python\\face-recognition\\python_project\\Photos'
haar_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")
people = []

for i in os.listdir(dir):
    people.append(i)

faceRecog = cv.face.LBPHFaceRecognizer_create()
faceRecog.read('faceTrain.yml')

img = cv.imread("C:\\Users\\rylep\\OneDrive\\Documents\\zlepsFile\\Python\\face-recognition\\python_project\\val\\us.JPG")
imGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_detect = haar_cascade.detectMultiScale(imGray, 1.2, 6)

for(x1, y1, x2, y2) in face_detect:
    cv.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0,255,0), 1)
    faceRegi = imGray[y1:y1+y2, x1:x1+x2]
    label, confidence = faceRecog.predict(faceRegi)
    confidence = "{:.2f}".format(confidence)
    print('Face Recognized: ', people[label])
    print(confidence)
    cv.putText(img, people[label], (x1,y1-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,255,0), 2)
    cv.putText(img, (confidence+"%"), (x1,y1+y2+20), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,255,0), 1)

cv.imshow("Detected Face", img)
cv.waitKey(0)