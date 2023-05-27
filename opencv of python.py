# from PIL import Image
# import pytesseract
#
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\高涵\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
#
# im = Image.open("2.png")
# text = pytesseract.image_to_string((im), lang='chi_sim')
# print(text)

# import cv2
# cap = cv2.VideoCapture(0)
# i = 1
# while(cap.isOpened()):
#     ret, frame = cap.read()
# #     if ret == False:
# #          break
#     if i % 100 == 0:
#         cv2.imwrite("tupian" + str(i/100) + '.jpg', frame)
#     i += 1
#     cv2.imshow("cool",frame)
#     cv2.waitKey(1)
#
# cap.release()
# cv2.destroyAllWindows()


#################实时检测，用pytesseact接口

# import cv2
# from PIL import Image
# import pytesseract
#
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\高涵\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
#
# cap = cv2.VideoCapture(0)
# i = 1
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if i % 100 == 0:
#         cv2.imwrite("tupian" + str(i/100) + '.jpg', frame)
#         im = Image.open("tupian" + str(i/100) + '.jpg')
#         text = pytesseract.image_to_string((im), lang='chi_sim')
#         print(text)
#     i += 1
#     cv2.imshow("cool",frame)
#     cv2.waitKey(1)


# ##############尝试百度接口
#
#
# -*- coding:utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~
    百度文字识别应用案例
"""

# 步骤1：导入百度aip的AipOcr类
from aip import AipOcr
import cv2
import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")


# 步骤2：设置百度云服务的AppID等相关key值
APP_ID = '25330246'
API_KEY = 'HYoI0P63Whn6YIl4kiMfgAuh'
SECRET_KEY = 'Ua3WqCXVeX0cl0cs4sAeHQ4RKjLq16Q0'

# 步骤3：创建百度智能云对象client
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


# 步骤4：创建一个读取图片的函数
def get_file_content(filePath):
    with  open(filePath, "rb") as fp:
        return fp.read()


options = {}
options["language_type"] = "CHN_ENG"
options["detect_direction"] = "true"
options["detect_language"] = "true"
options["probability"] = "true"

cap = cv2.VideoCapture(0)
i = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    if i % 100 == 0:
        cv2.imwrite("tupian" + str(i/100) + '.jpg', frame)
        image = get_file_content("tupian" + str(i/100) + '.jpg')
        data = client.basicGeneral(image, options)
        for item in data['words_result']:
            print(item['words'])
            speaker.Speak(item['words'])
    i += 1
    cv2.imshow("cool",frame)
    cv2.waitKey(1)



# 步骤5：调用自定义函数读取指定路径的图片
# image = get_file_content("realtime.png")

# 步骤6：设置option选项
# options = {}
# options["language_type"] = "CHN_ENG"
# options["detect_direction"] = "true"
# options["detect_language"] = "true"
# options["probability"] = "true"

# 步骤7：使用步骤3创建的client对象完成文字识别的分析

# data = client.basicGeneral(image, options)

# 步骤8：输出结果
# print("识别结果为：")
# print(data)#输出结果是一个字典
# print(data['words_result'])
# for item in data['words_result']:
#     print(item['words'])