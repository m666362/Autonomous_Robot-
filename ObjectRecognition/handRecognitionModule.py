# import the opencv library
import cv2
import mediapipe as mp
import numpy as np
import time

class HandDetector():
    def __init__(self, mode= False, maxHands= 2, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode;
        self.maxHands = maxHands;
        self.detectionConfidence = detectionConfidence;
        self.trackConfidence = trackConfidence;
        
        # MP hand objects
        self.mpHands = mp.solutions.hands;
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence);
        self.mpDraw = mp.solutions.drawing_utils;
        
    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(type(results))
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id == 0:
                        cv2.circle(image, (cx, cy), 25, (255,0,255), cv2.FILLED)
                        print(id, cx, cy)
            
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS);
        return image;

    def findPosition(self, image, handNo=0, draw=False):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(type(results))
        # print(results.multi_hand_landmarks)
        lmList = [];
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if id == 0:
                        cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS);

        return lmList;


def main():
    
    # define a video capture object
    cap = cv2.VideoCapture(0)
    # HandDetector Class
    detector = HandDetector();
    
    # fps objects
    cTime = 0
    pTime = 0

    while(True):
        success, image = cap.read()
        
        lmList = detector.findPosition(image, 1, True);
        if(len(lmList)>0):
            print(lmList[0])
        # lmList = detector.findPostiton(image, 1)
        # if len(lmList) !=0:
        #     print(lmList[1])
        
        # To show frame rate per second
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (18, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        
        # show webcam
        cv2.imshow('Web Cam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()