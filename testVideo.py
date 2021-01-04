import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# face recognitor
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # increase brightness, uncomment if u need this
    # frame = frame + 10

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces of each frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    # draw rectangle of each frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()