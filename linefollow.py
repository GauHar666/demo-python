import KeypressModule as kp
from djitellopy import tello
import cv2
import time
import threading
kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
global img
me.streamon()

def get_keyboard_input():
        if kp.get_key('q'):
        me.land()
        time.sleep(3)
    elif kp.get_key('e'):
        me.takeoff()

    elif kp.get_key('z'):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        time.sleep(0.3)
def main_thread():
    while True:
        get_keyboard_input()
        img = me.get_frame_read().frame
           cv2.imshow("Image", img)
        cv2.waitKey(1)
def path_thread():
    me.takeoff()
    time.sleep(1)

    me.send_rc_control(0, 0, 20, 0)
    time.sleep(3)
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

    me.send_rc_control(0, 0, 20, 0)
    time.sleep(8)
    me.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    me.land()

def main():
    thread = threading.Thread(target=path_thread)
    thread.start()
    main_thread()

if __name__ == '__main__':
    main()
    
