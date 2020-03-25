import cv2
import numpy as np
from PIL import Image
import pickle
import urllib.request as ur
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

def getProfile(id):
   connect = sqlite3.connect('///SQL///sql.db')
   cursor= connect.execute("SELECT * FROM user WHERE id=?",(str(id), ))
   profile=None
   for row in cursor:
     profile=row
   cursor.close()
   return profile


cam = cv2.VideoCapture(0)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
name = ""


while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(im, str(profile[0]), (x, y-20), font, 1, (255,255,255))
            cv2.putText(im, str(profile[1]), (x, y-50), font, 1, (255,255,255))
            cv2.putText(im, str(profile[2]), (x, y+10), font, 1, (255,255,255))
        else:
            id='Unknown'
    cv2.imshow('im',im)
    if cv2.waitKey(30) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()