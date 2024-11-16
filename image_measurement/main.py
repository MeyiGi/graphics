import cv2
import numpy as np
import utils
import time

# ----------- CONFIGURATIONS -----------
webcam = True
path = "test.jpg"
cap = cv2.VideoCapture(0)
cap.set(3, 1920) # Width
cap.set(4, 1080) # Height
scale = 3
wP = 210 * scale # Paper width
hP = 297 * scale# Paper height 
# --------------------------------------

def main():
    if webcam: 
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    img, contours = utils.getContours(img=img, minArea=50_000, draw=True, filter=4)
    
    if len(contours) != 0:
        biggest = contours[0][2]
        imgWarp = utils.warpImage(img, biggest, wP, hP)
        cv2.imshow("A4", imgWarp )
        img2, contours2 = utils.getContours(img=imgWarp, draw=False, minArea=50_000, filter=4)
        
        if len(contours) != 0:
            for obj in contours2:
                cv2.polylines(img2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utils.reorder(obj[2])
                nW = round(utils.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10, 1)
                nH = round(utils.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10, 1)
                
                # Draw arrowed lines on the image, not the contours list
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(img2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                
                # Add measurements as text
                x, y, w, h = obj[3]
                cv2.putText(img2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(img2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow("objects", img2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)

    time.sleep(0.1)

    


if __name__ == "__main__":
    while True:
        main()