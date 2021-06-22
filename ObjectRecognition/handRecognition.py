# -*- coding: utf-8 -*-

# import the opencv library
import cv2
import mediapipe as mp
import numpy as np
import time

def openWebCam():
    # define a video capture object
    cap = cv2.VideoCapture(0)
    
    # MP hand objects
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    
    # fps objects
    cTime = 0
    pTime = 0

    while(True):
        success, image = cap.read()
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(type(results))
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = image.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id == 0:
                        cv2.circle(image, (cx, cy), 25, (255,0,255), cv2.FILLED)
                        print(id, cx, cy)
                #print(handLms)
                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

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
openWebCam()