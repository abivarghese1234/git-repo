# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:07:04 2017
Code to find the missing components
@author: westghats123
"""

import cv2
import numpy as np
from numpy import array
from sklearn import neighbors
import templateMatch1 as t
   
def featureExtran(template,flag):
    hist = cv2.calcHist([template],[0],None,[256],[0,256])
    x,y = template.shape
  #percent of histogram bins with >0.5%
    total = int(0.005*(x*y))
    a = np.where(hist>total)
    numBins = len(a[0])/2.56
  # pct2pk percent of histogram range within largest 2 peaks
    srt = sorted(hist,reverse=True)
    srt = array(srt)
    temp = srt[0]
    temp1 = np.where(srt == temp)
    l = len(temp1[0])
    temp2 = srt[l]
    temp3 = np.where(srt==temp2)
    l2 = len(temp3[0])
    pct2pk = (l+l2)/2.56
  #average of variums sums of nearest neighbor pixel difference
    tempHist = hist
    sum =0
    for i in range(0,255):
        a = int(tempHist[i])
        b = int(tempHist[i+1])
        c =  abs(a-b)
        sum = sum+c
    avgSum = sum/255
    return numBins,pct2pk,avgSum,flag

def findmissingcomponents(goldenTemplate1,goldenTemplate2, componentTemplate, testImage):
    imgFull = goldenTemplate1
    template= componentTemplate
    testImg = goldenTemplate2
    test123 = testImage
    w,h = template.shape[::-1]
    d,count = t.templateMatching(imgFull,template)
    print(d,count)
            #tl = d[i][0],d[i][1]
            #br = d[i][0]+w,d[i][1]+h
                  #cv2.rectangle(imgFull,tl,br,255,5)
    temp = np.zeros((w,h),int)
    features = np.zeros((1,3),float)
    temp1 = np.zeros((w,h),int)
    features1 = np.zeros((1,3),float)
    #count = int(count)
    for i in range(0,count):                    #extracting the features for all the templates
        temp = imgFull[d[i][1]:d[i][1]+h,d[i][0]:d[i][0]+w]
        temp1 = testImg[d[i][1]:d[i][1]+h,d[i][0]:d[i][0]+w]
        feat = featureExtran(temp,0)
        feat1 = featureExtran(temp1,1)
        if i==0:
           features = feat
           features1 = feat1
        else:
            features = np.vstack([features,feat])
            features1 = np.vstack([features1,feat1])

    allFeatures = features
    allFeatures = np.vstack([allFeatures,features1])
    
    #********************************************************************************
    # Learning using the KNN classification algorithm
    X = allFeatures[:,:3]
    y = allFeatures[:,3]
    n_neighbors = count
    weights = 'uniform'
    
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights)
    clf.fit(X,y)
    print('**********************Fitting Complete****************')
#********************************************************************************
#Test image setup
    #test123 = cv2.imread("C:/New Folder/PCB inspection/test board2.jpg",0)
    
    temp2 = np.zeros((w,h),int)
    features2 = np.zeros((1,3),float)
    for i in range(0,count):                    #extracting the features for all the templates
        #print (i)
        temp2 = test123[d[i][1]:d[i][1]+h,d[i][0]:d[i][0]+w]
        feat2 = featureExtran(temp2,1)
        if i==0:
           features2 = feat2
        else:
           features2 = np.vstack([features2,feat2])
        #print(i,features2)
    #print(i+1)
    if i==0:
        features3 = features2[0],features2[1],features[2]
    else:
        features3 = features2[:,:3]
    Z = 0
    Z = clf.predict(features3)
    abc = np.nonzero(Z)
    #return d,abc 
    ''' 
    for i in range(0,len(abc[0])):
              ind = abc[0][i]
              t1 = d[ind][0],d[ind][1]
              b1 = d[ind][0]+w,d[ind][1]+h
              #cv2.rectangle(test123,t1,b1,255,5)
              
    #cv2.imshow("test123",test123)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''
    return d,abc 

       