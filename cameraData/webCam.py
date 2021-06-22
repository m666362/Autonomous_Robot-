# import the opencv library
import cv2


def openWebCam():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        cv2.imshow('Web Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


openWebCam()
