### Face Detection Algortihms

#Importing Libraries
import cv2
import numpy as np

#pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    #read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    #convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #result in frame
    cv2.imshow('Face Detection', frame)

    #break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the capture
cap.release()
cv2.destroyAllWindows()