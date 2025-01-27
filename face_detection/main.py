import cv2 as cv

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    ret, frame = capture.read()

    resized_frame = cv.resize(frame, (500, 400))
    gray = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)

    faced_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faced_rect:
        cv.rectangle(resized_frame, (x, y), (x+w, y+h), (0,255,0), thickness=2)

    cv.imshow("face detection", resized_frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break


capture.release()
cv.destroyAllWindows()