import cv2
import os
import numpy as np
from PIL import Image

# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture():
    """调用摄像图实时截取人脸，输出灰度图
    """
    cap = cv2.VideoCapture(0)       # 调用摄像头
    cnt = 1
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # 输出灰度图
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize = (100,100)) # 检测人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x-5, y-30), (x + w+5, y + h+20), (255, 0, 0), 1)  # 调整框体大小，与ORL数据库基本相仿
            roiImg = frame[y-30:y+h+20,x-5:x+w+5]
            gray = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('raw/r{}.png'.format(cnt), gray)
            cnt += 1
        cv2.imshow("camera",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def handle():
    """resize图片，转换到pgm格式
    """
    for i in range(10):
        img = cv2.imread('s41/r' + str(i+1) + '.png')
        new_img = cv2.resize(img, (92, 112))
        cv2.imwrite('s41/{}.png'.format(i+1), new_img)
        Image.open('s41/{}.png'.format(i+1)).convert('L').save('s41/{}.pgm'.format(i+1))


if __name__ == '__main__':
    # capture()
    handle()