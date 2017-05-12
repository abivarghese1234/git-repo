# -*- coding: utf-8 -*-
"""
Spyder Editor
WaterShed Algorithm for segmentation of template image
This is a temporary script file.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import Counter

"----------------------------------------------------------------------------------------------------------------------"
def cannyEdg(img):
    img1 = img
    img1 = (255-img1)
     
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel,iterations =1)
    #im2,contours,hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.Canny(opening,0,255)
    # Plot the images using pyplot
    return edges
"----------------------------------------------------------------------------------------------------------------------"
def findCont(img,temp2):
    templateImage = np.multiply(img,temp2)
    ret2,img4 = cv2.threshold(templateImage,70,255,cv2.THRESH_BINARY)
    (img5,contours,hierarchy) = cv2.findContours(img4.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours,key = cv2.contourArea,reverse=True)
    return templateImage,img4,cnts
"----------------------------------------------------------------------------------------------------------------------"
def preProc(img,im1):
    # otsu threshold
    ret,img1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # noise removal 
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel,iterations =2)
    
    #sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations =3)
    
    # finding sure foreground area
    
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret1,sure_fg = cv2.threshold(dist_transform,0.35*dist_transform.max(),255,0)    #0.35
                                
    #finding unknown area
                                
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
                                
    #Marker Labelling
    ret,markers = cv2.connectedComponents(sure_fg)
                                
    
    #mark the regions with unknown with zero
    markers[unknown==255] = 0
                                     
    # apply the watershed segmentation algorithm
                                      
    markers = cv2.watershed(im1,markers)  
    A = markers.ravel() 
    B = Counter(A)
    lm,lc = B.most_common(1)[-1]                                                       
    im1[markers==-1] = [255,0,0]

    temp = np.uint8(np.ones(img.shape))  
    temp[markers==-1]=0
    temp[markers==lm]=255
    temp1 = temp
    temp2 = np.zeros(img.shape,np.uint8)
    temp2[temp1==255] = 1
    return temp2
"-----------------------------------------------------------------------------------------------------------------"                 
def resCont(cnts):
    for c in cnts:                  #resulting contour  : contour with > 4 points
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)>=4:
            res = c
            break
    return res
"------------------------------------------------------------------------------------------------------------------"

def drawBox(templateImage,res):
    newImg = np.zeros(templateImage.shape,np.uint8)  
    rect = cv2.minAreaRect(res)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(newImg,[box],0,(255,255,255),2)
    return rect,newImg
"------------------------------------------------------------------------------------------------------------------"   
data = np.load('C:/New Folder/PCB inspection/Inspecting  SMD/template1.npy')            #Template locations
testImg = cv2.imread("C:/New Folder/PCB inspection/pcbInspection test/AV_PCB_FRONT_test5.jpg")   #Test Image
GoldenImg = cv2.imread("C:/New Folder/PCB inspection/Inspecting  SMD/123.jpg")       #Template image
ind = 12
h=90
w=40
im1 = GoldenImg[data[ind][1]:data[ind][1]+h,data[ind][0]:data[ind][0]+w]  
im2 = testImg[data[ind][1]:data[ind][1]+h,data[ind][0]:data[ind][0]+w]              
img = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)            
img2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)    

# get the component area of the template
temp2 = preProc(img,im1)

templateImage,im4,cnts = findCont(img,temp2)
templateImage2,im42,cnts2 = findCont(img2,temp2)     
res = resCont(cnts)
res2 = resCont(cnts2)
rect,newImg = drawBox(templateImage,res)
rect2,newImg2 = drawBox(templateImage2,res2)


plt.subplot(2,2,1), plt.imshow(im4)
plt.title(" threshold golden")
plt.subplot(2,2,2), plt.imshow(im42)
plt.title("threshold test")
plt.subplot(2,2,3), plt.imshow(newImg)
plt.title("contour golden")
plt.subplot(2,2,4), plt.imshow(newImg2)
plt.title("contour test")
    
#print "Ok";
