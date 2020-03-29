import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

SlotNumber = 3

def rect(x, y1 , y2 , z):
    global PData
    xold=x
    avg = list()
    Data = list()
    h1 = y1 + int(z)-30
    h2 = h1 + y2 -50

    for i in range(SlotNumber):
        w = x + int(z / 2)-20
        empty_crop = emptypic[y1:h1, x:w]
        park_crop = parkpic[y1:h1, x:w]
        black_crop = black[y1:h1, x:w]
        gray_crop = cv2.cvtColor(black_crop, cv2.COLOR_BGR2GRAY)
        avg.append(np.average(gray_crop))
        

        if avg[i]>= 150:
            cv2.rectangle(parkpic, (x, y1), (w, h1), (0, 0, 255), 3)
            cv2.rectangle(black, (x, y1), (w, h1), (0, 0, 255), 3)
            Data.append(1)

        else:
            cv2.rectangle(parkpic, (x, y1), (w, h1), (0, 255, 0), 2)
            cv2.rectangle(black, (x, y1), (w, h1), (0, 255, 0), 2)
            Data.append(0)

        x=x+90

    x=xold
    for i in range(SlotNumber,SlotNumber*2):
        w = x + int(z/2)-20
        empty_crop = emptypic[y2:h2, x:w]
        park_crop = parkpic[y2:h2, x:w]
        black_crop = black[y2:h2, x:w]
        gray_crop = cv2.cvtColor(black_crop, cv2.COLOR_BGR2GRAY)
        avg.append(np.average(gray_crop))

        if avg[i]>=150:
            cv2.rectangle(parkpic, (x, y2), (w, h2), (0, 0, 255), 3)
            cv2.rectangle(black, (x, y2), (w, h2), (0, 0, 255), 3)
            Data.append(1)

        else:
            cv2.rectangle(parkpic, (x, y2), (w, h2), (0, 255, 0), 2)
            cv2.rectangle(black, (x, y2), (w, h2), (0, 255, 0), 2)
            Data.append(0)

        x=x+90
        PData = {}
    for Slot in range(0, SlotNumber*2):
        PData["S{0}".format(Slot+1)] = Data[Slot]
    print(Data)
    return PData

empty = cv2.imread('EmptyFloor.jpg')
park = cv2.imread('Floor.jpg')

(h,w) = empty.shape[:2]
center = (w/2,h/2)
M = cv2.getRotationMatrix2D(center, 90, 0.7)

emptypic = cv2.warpAffine(empty, M ,(w,h))
parkpic = cv2.warpAffine(park, M ,(w,h))

UpCrop = 20
DownCrop = 270
m = 21
n = 21

blurpark = cv2.pyrMeanShiftFiltering(parkpic, m, n)
blurempty = cv2.pyrMeanShiftFiltering(emptypic, m, n)
subtract=cv2.subtract(blurempty,blurpark)
_,thresholding = cv2.threshold(subtract,20,255,cv2.THRESH_BINARY)
gray = cv2.cvtColor(thresholding,cv2.COLOR_BGR2GRAY)
contours,hierarchy= cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

black = parkpic*0
k=0
AreaContour = list()
AreaContourIndex = list()
NewContours = list()
PerimeterContour = list()
PerimeterContourIndex = list()

for i in range(len(contours)):
    Area = cv2.contourArea(contours[i])
    if Area>1100:
        AreaContour.append(Area)
        AreaContourIndex.append(k)
        NewContours.append(contours[i])
        k=k+1
        cnt = contours[i]
        rectangle = cv2.minAreaRect(cnt)              
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        cv2.drawContours(blurempty, [box], 0, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(black , (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(blurpark , (x, y), (x + w, y + h), (255, 0, 0), 2)


Thread = threading.Thread(target=rect,args=[200,70,230,200])
Thread.start()
Thread.join()

cv2.imshow('Original image (before rotation)' , park)
cv2.imshow('Detected cars' , parkpic)
cv2.imshow('Making detected region white and calculating its probability', black)
cv2.imshow('Blurred parking' , blurpark)
cv2.imshow('Blurred empty parking with contours on it' , blurempty)
cv2.waitKey(0)
cv2.destroyAllWindows()
