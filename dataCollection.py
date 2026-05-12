import cv2
import math
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#Sautmn
offset=20
imgSize=300

folder = "Data/bai"
counter=0
def close():
    cap.release()
    cv2.destroyAllWindows()
    exit()
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y +  h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
            

        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize
            

        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)
    cv2.imshow("image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f"{folder}/image_{time.time()}.jpg",imgWhite)
        print(counter)
    elif key == ord("q"):
        close()  

