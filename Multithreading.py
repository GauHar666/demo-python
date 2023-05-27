import KeypressModule as kp
from djitellopy import tello
import cv2
import time
import threading
import numpy as np
# from queue import Queue

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

def path1_thread():
    global img
    global lock
    while True:
        get_keyboard_input()
        img = me.get_frame_read().frame
        img = cv2.resize(img, (width, height))
        img = np.vsplit(img, sensors)[-1]

        imgThres = thresholding(img)
        cx = getContours(imgThres, img)  # For translation
        senOut = getSensorOutput(imgThres, sensors)  # For rotation
        sendCommands(senOut, cx)

        cv2.imshow('Output', img)
        cv2.imshow('Path', imgThres)
        cv2.waitKey(1)
#         me.send_rc_control(0, fSpeed, 0, curve)

def path2_thread():
    global lock
    lock.acquire()
    me.move_up(30)
    time.sleep(1)
    me.rotate_clockwise(90)
    time.sleep(1)
    cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
    time.sleep(0.3)
    me.rotate_clockwise(-90)
    time.sleep(1)
    me.move_forward(30)
    time.sleep(1)
    me.move_down(30)
    lock.release()


hsvVals = [0, 46, 162, 31, 213, 255]
sensors = 3
threshold = 0.1
width, height = 480, 360
lock = threading.Lock()

sensitivity = 3  # Less sensitive if number is high

weights = [-25, -15, 0, 15, 25]

fSpeed = 15
curve = 0
global img


def get_keyboard_input():
    if kp.get_key('q'):
        me.land()
        me.move_down(30)
        time.sleep(3)
    elif kp.get_key('e'):
        me.takeoff()
    elif kp.get_key('z'):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        time.sleep(0.3)


def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    #         q.put(mask)
    return mask


def getContours(imgThres, img):
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    else:
        cx = width // 2

    return cx


def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
        cv2.imshow(str(x), im)
    print(senOut)
    return senOut


def sendCommands(senOut, cx):
    global curve
    # Translation
    lr = (cx - width // 2) // sensitivity
    lr = int(np.clip(lr, -10, 10))
    if 2 > lr > -2:
        lr = 0
    # me.send_rc_control(lr, 0, 0, 0)

    # Rotation
    if senOut == [1, 0, 0]:
        curve = weights[0]
    elif senOut == [1, 1, 0]:
        curve = weights[1]
    elif senOut == [0, 1, 0]:
        curve = weights[2]
    elif senOut == [0, 1, 1]:
        curve = weights[3]
    elif senOut == [0, 0, 1]:
        curve = weights[4]
    elif senOut == [0, 0, 0]:
        curve = weights[2]
    elif senOut == [1, 0, 1]:
        curve = weights[2]
    me.send_rc_control(0, fSpeed, 0, curve)


t1 = threading.Thread(target=path1_thread)
t2 = threading.Thread(target=path2_thread)
t1.start()
senOut = getSensorOutput(thresholding(img), sensors)

if senOut == [1, 1, 1]:
    t2.start()
    t2.join()
    t1.join()

