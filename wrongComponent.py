# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 10:56:13 2017
backpropagation learning for the templates.
@author: westghats123
"""
import cv2
import numpy as np
from scipy import ndimage
from numpy import array
from collections import Counter
from matplotlib import pyplot as plt
from math import exp 
from random import seed
from random import random



MIN_DESCRIPTOR = 6

def TemplateMatching(imgFull,template,inde):
    w,h = template.shape[::-1]
    res = cv2.matchTemplate(imgFull,template,cv2.TM_SQDIFF)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
   #top_left = min_loc
    #bottom_right = (top_left[0]+w,top_left[1]+h)
    #cv2.rectangle(imgFull,top_left,bottom_right,255,5)
    result = np.reshape(res,res.shape[0]*res.shape[1])
    sortedVal = np.argsort(result)
    # np.unravel index used to map template matching output value to x,y points
    #y2, x2 = np.unravel_index(sortedVal[1], res.shape) #second best match
    #inde = 100
    flag =0
    a = np.zeros((inde,2),int)
    b = np.zeros((inde,2),int)
    c = np.zeros((inde,2),int)
    #x2 = 0

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
            tl = c[i][0],c[i][1]
            br = c[i][0]+w,c[i][1]+h
            cv2.rectangle(imgFull,tl,br,255,5)
       
        d = np.zeros((count,2),int)
    for i in range(0,count):
        if (c[i][0] ==0) &(c[i][1]==0):
            flag=1
        else:
            d[i]=c[i]
    return d

def watershedSegmentation(im1):
    img = im1
    im1 = cv2.cvtColor(im1,cv2.COLOR_GRAY2BGR)
    #img = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    ret,img1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # noise removal 
    kernel = np.ones((1,2),np.uint8)
    opening = cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel,iterations =1)
        
    #sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations =2)
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
    lm,lc = B.most_common(1)[-1]    # to find out the largest area(component area)                                                   
    im1[markers==-1] = [255,0,0]
    
    temp = np.uint8(np.ones(img.shape))  
    temp[markers==-1]=0
    temp[markers==lm]=255
    temp1 = temp
    temp2 = np.zeros(img.shape,np.uint8)
    temp2[temp1==255] = 255
    return temp2
def CannyEdgeDetection(img):
     
    img1 = img
    img1 = (255-img1)
     
    kernel = np.ones((1,2),np.uint8)
    opening = cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel,iterations =1)
    #im2,contours,hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.Canny(opening,0,255)
    return edges
def findDescriptor(img,degree):
    """ findDescriptor(img) finds and returns the
    Fourier-Descriptor of the image contour"""
    contour = []
    img = cv2.convertScaleAbs(img)
    img, contour, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    fourier_result = truncate_descriptor(fourier_result, degree)
    fourier_result = np.absolute(fourier_result)
    return fourier_result
def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors
def NormalizeFourier(fd):
    norFou = np.zeros(0,np.float)
    bv = max(fd)+1000
    sv = min(fd)/2
    for i in range(len(fd)):
        nv = (fd[i]-sv)/(bv-sv)
        norFou = np.append(norFou,nv)
    return norFou
def FourierDescriptor(data, GoldenImg,template,key):
    h,w = template.shape
    edges = np.zeros((h,w,len(data)),np.uint8)
    fourierData = np.zeros(0,np.float)
    for ind in range(len(data)):
        im1 = GoldenImg[data[ind][1]:data[ind][1]+h,data[ind][0]:data[ind][0]+w]
        edges[:,:,ind] = watershedSegmentation(im1)
        fourier_result = findDescriptor(edges[:,:,ind],MIN_DESCRIPTOR)
        fourierData = np.append(fourierData,fourier_result)
        fourierData = np.append(fourierData,key)
    fourierData = np.reshape(fourierData,(len(data),MIN_DESCRIPTOR+1))
    return fourierData
'************************INITIALIZING ,ACTIVATION FUNCTION*************'
def activate(weights, inputs):                      #ACTIVATION FUNCTION
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation +=weights[i]*inputs[i]   #Sum(weights*input)
    return activation

def transfer(activation):                   #TRANSFER FUNCTION SIGMOID
    return 1.0/(1.0 + exp(-activation))   ## Using the sigmoid function to find out the output value 
    
def initialize_network(n_input,n_hidden,n_output):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_input+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_output)]
    network.append(output_layer)
    return network
    
'**************Train Network**************************'
# Update weights  weight = weight+learning rate *error*input
def update_weights(network,learning_rate,row):
    inputs = row[:-1]
    for i in range(len(network)):
        if i !=0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate*inputs[j]*neuron['delta']
            neuron['weights'][-1] += learning_rate*neuron['delta']

def train_network(network,train_data,learn_rate,n_epoch,n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train_data:
            outputs = forward_propagation(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[np.uint8(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backpropagation_error(network,expected)
            update_weights(network,learn_rate,row)
        print('>epoch=%d, lrate=%.3f, error=%.3f' %(epoch,learn_rate,sum_error))
    #print expected
'***************************BACKPROPAGATION *******'    
def transfer_derivative(output):
    return output*(1.0-output)

def backpropagation_error(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i!=len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron  = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
'**************************FORWARD PROPAGATION*************************'
def forward_propagation(network,row):
    inputs = row
    for layer in network:
        new_input = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = transfer (activation)
            new_input.append(neuron['output'])
        inputs = new_input
    return inputs
'******************************PREDICTION************************'
def predict(network,row):
    outputs = forward_propagation(network,row)
    return outputs.index(max(outputs))
'*****************************************************************'
def testFrameDesc(testframe,templates,datapack):
    database = []
    for i in range(0,len(templates)):
        data = datapack[i]
        template = templates[i]
        GoldenImg = testframe
        fourier_desc = FourierDescriptor(data,GoldenImg,template,i)
        if i ==0:
            database = fourier_desc
        else:
            database = np.vstack((database,fourier_desc))
    normalizedFourier = np.zeros(database.shape,np.float)
    for i in range(0,MIN_DESCRIPTOR):
        normalizedFourier[:,i] = NormalizeFourier(database[:,i])
    normalizedFourier[:,MIN_DESCRIPTOR] =  database[:,MIN_DESCRIPTOR]
    
    dataset = normalizedFourier
    return dataset



def testWrongComponent(golden,templates,datapack,testframe):
    database = []
    for i in range(0,len(templates)):
        data = datapack[i]
        template = templates[i]
        GoldenImg = golden[0]
        fourier_desc = FourierDescriptor(data,GoldenImg,template,i)
        if i ==0:
            database = fourier_desc
        else:
            database = np.vstack((database,fourier_desc))
    normalizedFourier = np.zeros(database.shape,np.float)
    for i in range(0,MIN_DESCRIPTOR):
        normalizedFourier[:,i] = NormalizeFourier(database[:,i])
    normalizedFourier[:,MIN_DESCRIPTOR] =  database[:,MIN_DESCRIPTOR]  
    dataset = normalizedFourier
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 30, n_outputs)
    train_network(network, dataset, 0.4,5000, n_outputs) 
    
    
    frameData = testFrameDesc(testframe,templates,datapack)
    prediction =[]
    for row in frameData:
        prediction.append(predict(network,row))
        #print('Expected = %d, Result = %d' % (row[-1],prediction))
    
    return frameData[:,-1],prediction
    
'''        
imgFull = cv2.imread("/home/abi-westghats/PCB inspection/Inspecting  SMD/123clahe.jpg",0)
template = cv2.imread("/home/abi-westghats/PCB inspection/Inspecting  SMD/template12.jpg",0)
GoldenImg = cv2.imread("/home/abi-westghats/PCB inspection/Inspecting  SMD/123.jpg")
template1 = cv2.imread("/home/abi-westghats/PCB inspection/Inspecting  SMD/template2NewCol.jpg",0)

data = TemplateMatching(imgFull,template,100)
data1 = TemplateMatching(imgFull,template1,45)
rotated = ndimage.rotate(template,90)
data2 = TemplateMatching(imgFull,rotated,10)
fourier_desc = FourierDescriptor(data, GoldenImg,template,0)
fourier_desc1 = FourierDescriptor(data1, GoldenImg,template1,1)
fourier_desc2 = FourierDescriptor(data2, GoldenImg,rotated,2)
database = np.vstack((fourier_desc,fourier_desc1,fourier_desc2))
normalizedFourier = np.zeros(database.shape,np.float)
for i in range(0,MIN_DESCRIPTOR):
    normalizedFourier[:,i] = NormalizeFourier(database[:,i])
normalizedFourier[:,MIN_DESCRIPTOR] =  database[:,MIN_DESCRIPTOR]  
dataset = normalizedFourier
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 30, n_outputs)
train_network(network, dataset, 0.4,5000, n_outputs) 

for row in dataset:
    prediction = predict(network,row)
    print('Expected = %d, Result = %d' % (row[-1],prediction))

plt.subplot(121), plt.imshow(edges[:,:,2])
plt.title(" mask")  
plt.subplot(122), plt.imshow(edges[:,:,1])
plt.title(" golden")
'''