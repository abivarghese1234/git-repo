# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:36:57 2017
template matching 1
@author: westghats123
"""
import cv2
import numpy as np
from numpy import array

def templateMatching(imgFull,template):
    w,h = template.shape[::-1]
    inde = 10*350
    res = cv2.matchTemplate(imgFull,template,cv2.TM_SQDIFF)


    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)


    top_left = min_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    #cv2.rectangle(imgFull,top_left,bottom_right,255,5)


    result = np.reshape(res,res.shape[0]*res.shape[1])

    sortedVal = np.argsort(result)

    # np.unravel index used to map template matching output value to x,y points
    #y2, x2 = np.unravel_index(sortedVal[1], res.shape) #second best match
    #inde = 35
    flag =0
    a = np.zeros((inde,2),int)
    b = np.zeros((inde,2),int)
    c = np.zeros((inde,2),int)
    x2 = 0

    # taking only top 200 values of the resulting matching template and converting it to coordinate values
    for i in range (0,inde):
        y1, x1 = np.unravel_index(sortedVal[i], res.shape) 
        a[i] = x1,y1

    #sorting the coordinate values
    a = sorted(a,key=lambda l:l[0])
    a = array(a)
    b[0] = a[0]  
    i=0 

#selecting only distinct values from the sorted list
    for j in range(1,inde):
        if (abs(a[j][0]-b[i][0])>=int(w/2)) | (abs(a[j][1]-b[i][1])>=int(h/2.5)):
            i = i+1
            b[i] = a[j]        
            
    b = sorted(b,key=lambda l:l[1])   
    b = array(b)       
    #c[0] = b[0]
    i=0
    for j in range(1,inde):
        if (abs(b[j][0]-c[i][0])>=int(w/2)):
            i = i+1
            c[i] = b[j]                

    #drawing rectangles around the matched area
    count =0
    c = sorted(c,key=lambda l:l[1],reverse= True)   
    c = array(c)        
    for i in range(0,inde):
        if (c[i][0] ==0) &(c[i][1]==0):
            flag=1
        else: 
            count +=1
            
       
    d = np.zeros((count,2),int)
    
    for i in range(0,count):
        if (c[i][0] ==0) &(c[i][1]==0):
            flag=1
        else:
            d[i]=c[i]
            #tl = c[i][0],c[i][1]
            #br = c[i][0]+w,c[i][1]+h
            #cv2.rectangle(imgFull,tl,br,255,5)
    
    imgFull = cv2.resize(imgFull,(0,0),fx=0.5,fy=0.5)
    #print(d)
    #print(count)
    # print the result
    '''
    cv2.imshow('matched image',imgFull)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return d,count
    