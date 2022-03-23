import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
decorator = htm.handDetector()

while True:
    sucess, img = cap.read()
    img = decorator.findHands(img , connection = True )
    lmList = decorator.findPosition(img, draw = True)

    if len(lmList) !=0:
        print(lmList[4])



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)