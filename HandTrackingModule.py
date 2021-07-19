import mediapipe as mp
import cv2 as cv
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.tipIds = [4,8,12,16,20]

    def findHands(self, frame, draw = True):
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            if draw:
                for handlm in self.results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(img, handlm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(23,37,191), thickness=5, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(212,0,0), thickness=2, circle_radius=1),
                        )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = [] 
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # if draw:
                #     cv.circle(img,(cx,cy), 10, (255,0,255), cv.FILLED)
        return self.lmList

    def fingersUP(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Four fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)

    detector = HandDetector()

    while True:
        ret, frame = cap.read()
        img = detector.findHands(frame)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:     
        #     print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 3, (235, 16, 0), 3)
        cv.imshow("Image", img)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()