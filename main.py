import cv2

#Load cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')


#Read input file and greyscale
img = cv2.imread('family.jpg')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
faces = face_cascade.detectMultiScale(grey, 1.1, 4)

#Draws Rectangle around Faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#Display output
cv2.imshow('img', img)
cv2.waitKey()



