import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = np.load("face_data.npy")
X = data[:,1:].astype(int)
y = data[:,0]

model = KNeighborsClassifier(5)
model.fit(X, y)

face_list = []

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray)

        areas = []
        for face in faces:
            x, y, w, h = face
            areas.append((w * h, face))

        if len(faces) > 0:
            face = max(areas)[1]
            x, y, w, h = face

            face_img = gray[y:y + h, x:x + w]

            face_img = cv2.resize(face_img, (100, 100))
            face_flatten = face_img.flatten()
            face_list.append(face_flatten)

            res = model.predict([face_flatten])

            cv2.putText(frame, str(res[0]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        break


