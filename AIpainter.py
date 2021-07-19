import cv2 as cv
import numpy as np
import HandTrackingModule as track
import os

wCam, hCam = 1280, 720
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = track.HandDetector(maxHands=1, detectionCon=0.8)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)



while True:
    # Capture Live Image
    ret, frame = cap.read()

    img = cv.flip(frame, 1)

    # Find Landmarks
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # print(lmList)

        # tips of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which finger is up
        fingers = detector.fingersUP()
        # print(fingers)

        # Finger Selection:

        # Erase
        if fingers[1] and fingers[2] and fingers[3]:
            # print("Erase")
            cv.circle(img, (x2,y2), 40, (0,0,0), cv.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x2, y2

            cv.line(img, (xp,yp), (x2,y2), (0,0,0), 60)
            cv.line(imgCanvas, (xp,yp), (x2,y2), (0,0,0), 60)
            xp, yp = x2, y2

        # Move
        elif fingers[1] and fingers[2]:
            # print("move")
            xp, yp = 0, 0

        # Draw
        if fingers[1] and fingers[2] == False:
            # print("draw")
            cv.circle(img, (x1,y1), 10, (255,0,255), cv.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv.line(img, (xp,yp), (x1,y1), (0,0,255), 10)
            cv.line(imgCanvas, (xp,yp), (x1,y1), (0,0,255), 10)
            xp, yp = x1, y1
    
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)
    # img = cv.addWeighted(img, 0.5, imgInv, 0.5, 0)
    
    cv.imshow("Image", img)
    # cv.imshow("Canvas", imgCanvas)
    
    

    key = cv.waitKey(1)
    if key == ord('q'):
        break