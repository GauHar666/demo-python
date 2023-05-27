import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
# import Speech

import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#实用3D摄像头的话会更加准确

#如何计算距离

#如何找到手部识别器

detector = HandDetector(detectionCon=0.8,maxHands=1)

#获得相应的焦距
x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

coff = np.polyfit(x,y,2)


while True:
    success, img = cap.read()
    hands = detector.findHands(img,draw=False)
    a = 0
    if hands:
        lmList = hands[0]["lmList"]
        x, y, w, h = hands[0]['bbox']
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]

        distance = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
        A, B, C = coff
        distanceCM = A*distance**2+B*distance+C

        #print(distanceCM)#找个值是大小，并不是距离

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+5, y-10))
        a = a+1
        if a % 10 == 0:
            speaker.Speak("前方有障碍物距离您"+f'{int(distanceCM)} 厘米')
        # SetReader(Reader_Type["Reader_XiaoPing"])  # 选择播音人晓萍
        # SetVolume(10)
        # Speech_text("你好亚博智能科技", EncodingFormat_Type["GB2312"])
        #
        # while GetChipStatus() != ChipStatus_Type['ChipStatus_Idle']:  # 等待当前语句播报结束
        #     time.sleep(0.1)
        print(f'{int(distanceCM)} cm')


    cv2.imshow("image",img)
    cv2.waitKey(1)