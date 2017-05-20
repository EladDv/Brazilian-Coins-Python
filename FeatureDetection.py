#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:38:46 2017

@author: Elad Dvash
"""
import os
import random

import numpy as np
from scipy import ndimage as image
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import skimage

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def prepareImage(img,cimg,returnMaxDims = False):

    BrightImg = cimg
    
    NormalizedImg = rgb2gray(img)
    seed = np.copy(NormalizedImg)
    seed[1:-1, 1:-1] = NormalizedImg.min()
    mask = NormalizedImg
    fill = -skimage.morphology.reconstruction(seed, mask, method='dilation')

    #fig = plt.figure()
    #plt.imshow(fill,cmap = plt.get_cmap('gray'))

    
    seed2 = np.copy(fill)
    seed2[1:-1, 1:-1] = fill.min()
    mask = fill
    
    dilated = skimage.morphology.reconstruction(seed2, mask, method='dilation')
    fill = fill - dilated
    thresh = skimage.filters.threshold_otsu(fill)
    fill = fill > thresh  
    #fill = skimage.feature.canny(fill,sigma = 2.5, low_threshold=10, high_threshold=50)
    
    
    fill = image.binary_dilation(fill, structure = np.ones((3,3)))
    fill = image.binary_fill_holes(fill)
    fill = image.binary_erosion(fill, structure = np.ones((7,7)))


    #fig = plt.figure()
    #plt.imshow(fill,cmap = plt.get_cmap('gray'))
    
    distance = image.distance_transform_edt(fill)
    distance = image.gaussian_filter(distance,1.1)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7,7)),
                            labels=fill)
    markers = image.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=fill)
    label_image = skimage.measure.label(labels)
    #print(np.unique(labels))
    TempCoins = np.zeros((len(np.unique(labels)) , 128 , 128, 3)).astype(np.uint8)
    counter = 0
    for region in skimage.measure.regionprops(label_image):
        xLen = np.abs(-min(region.coords[:,0])+max(region.coords[:,0]))
        yLen = np.abs(-min(region.coords[:,1])+max(region.coords[:,1]))
        if(region.area >= 900 and xLen<=128 and yLen<=128 and xLen/yLen < 2 and xLen/yLen > 1/2 ):
             mySlice = (slice(min(region.coords[:,0]) , max(region.coords[:,0]) ),
                        slice(min(region.coords[:,1]) , max(region.coords[:,1]) ) )
             TempCoins[counter, int((128-xLen)/2) : int((128-xLen)/2) + xLen 
                         , int((128-yLen)/2): int((128-yLen)/2) + yLen,:] = BrightImg[mySlice]
             #fig= plt.figure()             
             #plt.imshow(TempCoins[counter])
             counter = counter + 1

    Coins = TempCoins[ : counter ,:,:,:]
    #Debugging - Midway Images   
    '''
    fig = plt.figure()
    plt.imshow(cimg)

    fig = plt.figure()
    plt.imshow(fill,cmap = plt.get_cmap('gray'))

    fig = plt.figure()
    plt.imshow(labels,cmap = plt.get_cmap('jet'))
    '''
    if returnMaxDims:
	    return maxDims
    return Coins



def SaveSegmentedImage(libPath,path, addition = ''):
    Images = SegmentImage(os.getcwd()+libPath+'/'+path)
    fig, ax = plt.subplots(4, 5)
    ax = ax.ravel()
    ax[0].imshow(image.imread(os.getcwd()+libPath+'/'+path))
    for i,im in enumerate(Images):
        ax[i+1].imshow(im)
    plt.close("all")
    fig.savefig(os.getcwd()+'/reg_data/'+addition+path)    

def SegmentAllImages(libPath):
    RegImages = os.listdir(os.getcwd()+libPath)
    random.shuffle(RegImages)
    for path in RegImages:
        SaveSegmentedImage(libPath,path)

def SegmentImage(imPath):
    img = image.imread(imPath, mode = 'L')
    cimg = image.imread(imPath)
    img2 = prepareImage(cimg,cimg)
    return img2


#SegmentAllImages('\\all_reg')
#res = np.load('Results.npy').astype(int)
#SegmentAllImages('/all_reg')   
#SegmentImage('105.jpg')




























def MergeDefectiveBounds(mList):
    ToRet = mList[:]
    TempSliceHolder = mList[:]
    for i,j in enumerate(TempSliceHolder):
        for k,m in enumerate(TempSliceHolder):
            if (m in ToRet and not (k == i)):
                if(((np.abs(TempSliceHolder[k][0].start - j[0].start) <= 8 or
                       np.abs(TempSliceHolder[k][0].stop - j[0].stop) <= 8) and
                    (np.abs(TempSliceHolder[k][1].stop - j[1].start) <= 8 or
                     np.abs(TempSliceHolder[k][1].start - j[1].stop) <= 8) )):
                
                    stop0  = max(TempSliceHolder[k][0].start,TempSliceHolder[k][0].stop,j[0].start,j[0].stop)
                    start0 = min(TempSliceHolder[k][0].start,TempSliceHolder[k][0].stop,j[0].start,j[0].stop)
                    stop1 = max(TempSliceHolder[k][1].start,TempSliceHolder[k][1].stop,j[1].start,j[1].stop)
                    start1 = min(TempSliceHolder[k][1].start,TempSliceHolder[k][1].stop,j[1].start,j[1].stop)
                    mSlice = (slice(start0,stop0),slice(start1,stop1)) 
                    ToRet.remove(TempSliceHolder[k])
                    if(j in ToRet):
                        ToRet.remove(j)
                    if(not mSlice in ToRet):
                        ToRet.append(mSlice)
                        
                elif(((np.abs(TempSliceHolder[k][1].start - j[1].start) <= 8 or
                       np.abs(TempSliceHolder[k][1].stop - j[1].stop) <= 8) and 
                      (np.abs(TempSliceHolder[k][0].start - j[0].stop) <= 8 or
                       np.abs(TempSliceHolder[k][0].stop - j[0].start) <= 8))):
                   
                    stop0  = max(TempSliceHolder[k][0].start,TempSliceHolder[k][0].stop,j[0].start,j[0].stop)
                    start0 = min(TempSliceHolder[k][0].start,TempSliceHolder[k][0].stop,j[0].start,j[0].stop)
                    stop1 = max(TempSliceHolder[k][1].start,TempSliceHolder[k][1].stop,j[1].start,j[1].stop)
                    start1 = min(TempSliceHolder[k][1].start,TempSliceHolder[k][1].stop,j[1].start,j[1].stop)
                    mSlice = (slice(start0,stop0),slice(start1,stop1)) 
                    ToRet.remove(TempSliceHolder[k])
                    if(j in ToRet):
                        ToRet.remove(j)
                    if(not mSlice in ToRet):
                        ToRet.append(mSlice)
    TempSliceHolder = None
    return ToRet