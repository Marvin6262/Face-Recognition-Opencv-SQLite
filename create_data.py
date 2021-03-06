import cv2
import sqlite3

connect = sqlite3.connect('SQL//sql.db')
cur = connect.cursor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
id = input("Please Input Your id: ")
name = input("Please Input Your name: ")
nums = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        nums = nums+1
        cv2.imwrite("dataSet\."+str(id)+"."+str(nums)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(100)

    connect.execute("INSERT INTO user(id,name) Values("+str(id)+",' "+str(name)+" ' )")
    connect.commit()
    cur.close()


    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if nums>20:
        break

cap.release()
cv2.destroyAllWindows()
