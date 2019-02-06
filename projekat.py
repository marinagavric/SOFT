import cv2
import numpy as np
from train_network import getModel
from recognize_number import recognizeNumbers
import recognize_number as rn
import matplotlib.pyplot as plt

brojac=0;
suma=0
video = cv2.VideoCapture("video-0.avi")

ret, frame = video.read()


def DetectGreenLine(frame):
    b,g,r = cv2.split(frame)
   # cv2.imshow("green",g)
    gg= cv2.GaussianBlur(g ,(7,7),0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gg,kernel,iterations= 1)
    dilate = cv2.dilate(erosion,kernel,iterations= 1)
    edges = cv2.Canny(dilate,75,150)
   
    lines = cv2.HoughLinesP(edges,1, np.pi/180,100,maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
    return lines
    
    
def DetectBlueLine(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    edges = cv2.Canny(mask,75,150)
    lines = cv2.HoughLinesP(edges,1, np.pi/180,100,maxLineGap=50)
    if lines is not None:
       # print(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
    return lines

linijePlave=DetectBlueLine(frame)
linijeZelene=DetectGreenLine(frame)

model= getModel()

blueLine=linijePlave[0]
greenLine= linijeZelene[0]


while True: 
    ret, frame = video.read()
    
    if not ret:
        break
   # cv2.imshow("Video",frame)
    brojac+=1
   
    key = cv2.waitKey(27)
    if key == 27:
         break
     
    recognizeNumbers(frame, greenLine, blueLine)
        
    
for r1 in rn.regioniPlava:
        broj = r1.reshape(1, 28, 28, 1)
        number = 0
        niz = model.predict(broj)
        number=niz.argmax()
        suma = suma + number
        print("broj je")
        print(number)
        print('Suma zbir jee')
        print(suma)    

for r2 in rn.regioniZelena:
        broj = r2.reshape(1, 28, 28, 1)
        number = 0
        niz = model.predict(broj)
        number=niz.argmax()
        suma = suma - number
        print("broj je")
        print(number)
        print('Suma razlika jee')
        print(suma)    
      
video.release()
cv2.destroyAllWindows()     
