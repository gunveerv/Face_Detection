import cv2
import sys

#for camera
cascPath = 'haarcascade_frontalcatface.xml'

#Load cascade
face_cascade = cv2.CascadeClassifier(cascPath)

#capture video
video_capture = cv2.VideoCapture(0)

print('starting...')

while True:
    
    #Frame by frame capture
    ret, frame = video_capture.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    #Draw Rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Display Frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('shutting down...')
        break

#Release the capture when done
video_capture.release()
cv2.destroyAllWindows()



