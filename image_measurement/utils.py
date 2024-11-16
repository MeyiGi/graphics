import cv2
import numpy as np


def getContours(img, threshold=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, threshold1=threshold[0], threshold2=threshold[1])
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErosion = cv2.erode(imgDilation, kernel, iterations=2)
    
    if showCanny:
        cv2.imshow("Canny", imgErosion)
    
    contours, hierarchy = cv2.findContours(imgErosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx )
            if filter > 0:
                if len(approx) == filter:
                    final_contours.append([len(approx), area, approx, bbox, i])
            else:
                final_contours.append([len(approx), area, approx, bbox, i])

    final_contours = sorted(final_contours, key=lambda x: x[1], reverse=True)

    if draw:
        for contour in final_contours:
            cv2.drawContours(img, contour[4], -1, (0, 0, 255), 3)

    return img, final_contours

def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warpImage(img, points, width, height, pad=30):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
     
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5