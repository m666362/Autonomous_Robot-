import cv2
import numpy as np
import time

#Load YOLO Algorithm
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names =  net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes), 3))


def openWebCam():
    # define a video capture object
    cap = cv2.VideoCapture(0)

    # fps objects
    cTime = 0
    pTime = 0

    while (True):
        success, image = cap.read()

        imgrb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channels = image.shape

        # Detect Objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # for b in blob:
        #     for n, img in enumerate(b):
        #         cv2.imshow(str(n), img)

        net.setInput(blob)
        outs = net.forward(output_layers)
        print(outs)

        # showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # cv2.circle(image, (center_x, center_y), 10, (0,255,0), 2);
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(image, (x, y), (x+w, y+h) , (0,255,0), 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # threshold value for duplicate detection
        print(indexes)
        detectedObjects = len(boxes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                # print(label)
                # cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

        # To show frame rate per second
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # show webcam
        print(type(image));
        cv2.imshow('Web Cam', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


openWebCam()