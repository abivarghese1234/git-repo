# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:41:38 2017
test123   uses pic and place file to optimize the templates matched
@author: westghats123
"""
import cv2
import numpy as np
import missingComponent as mc
import templateMatch1 as t

imgFull = cv2.imread("/home/abi-westghats/PCB inspection/pcbInspection test/pcb3/claheres.jpg",0)
testImg = cv2.imread("/home/abi-westghats/PCB inspection/pcbInspection test/pcb3/new1.jpg",0)
template = cv2.imread("/home/abi-westghats/PCB inspection/pcbInspection test/pcb3/template6.jpg",0)
test123 = cv2.imread("/home/abi-westghats/PCB inspection/pcbInspection test/pcb3/new1.jpg",0)
d,abc = t.templateMatching(imgFull,template)
with open('/home/abi-westghats/PCB inspection/pcbInspection test/pcb3/picplace.txt','r') as in_file:
    content = in_file.read()
array = content.split()
#numbers = map(int,array)
d1 = np.zeros((len(array),2),np.uint32)
for  i in range(0,len(array)):
    d1[i] = array[i].split(',')
k=0

temp = np.zeros((len(d)*len(d1),2),np.uint32)
for i in range(0,len(d1)):
    flag = 0
    for j in range(0,len(d)):
        absx = abs(d[j][0]-d1[i][0])
        absy = abs(d[j][1]-d1[i][1])
        if (absx<10 &  absy<10):
            if d1[i] in temp[k]:
                flag =1
            else:
                temp[k] = d1[i]
                k+=1

w,h = template.shape[::-1]
a= temp
b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
_, idx = np.unique(b, return_index=True)
c = a[idx]
for i in range(0,len(c)):
        if (c[i][0] ==0) &(c[i][1]==0):
            flag=1
        else:
            tl = c[i][0],c[i][1]
            br = c[i][0]+w,c[i][1]+h
            cv2.rectangle(imgFull,tl,br,255,5)

imgFull = cv2.resize(imgFull,(0,0),fx=0.5,fy=0.5)
'''
cv2.imshow('matched image',imgFull)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.imwrite("templateMatchingResult1.jpg",imgFull)

#d,features = mc.findmissingcomponents(imgFull,testImg,template,test123)