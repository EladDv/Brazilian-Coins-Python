#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:38:46 2017

@author: Elad Dvash
"""

import numpy as np
from scipy import ndimage as image
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed

def prepareImage(img,cimg):
    NormalizedImg = img #np.sum(img**2,axis = 2)
    #fig = plt.figure()
    #plt.imshow(NormalizedImg,cmap = plt.get_cmap('gray'))
    norm = image.gaussian_laplace(NormalizedImg,2)
    #fill = image.binary_closing(norm)
    fill = image.binary_fill_holes(norm)
    fill = image.binary_erosion(fill,structure=np.ones((3,3)))
    distance = image.distance_transform_edt(fill)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7, 7)),
                            labels=fill)
    markers = image.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=fill)
    #print(np.unique(labels))
    finL = image.find_objects(labels)
    TempCoins = np.zeros((len(np.unique(labels)) , 128 , 128, 3)).astype(np.uint8)
    counter = 0
    finL = MergeDefectiveBounds(finL)
    for i,j in enumerate(finL):
        if((j[0].stop-j[0].start > 30 and j[1].stop-j[1].start > 30 ) and
           (j[0].stop-j[0].start <120 and j[1].stop-j[1].start<120)):
            xLen = j[0].stop-j[0].start
            yLen = j[1].stop-j[1].start
            TempCoins[counter, int((128-xLen)/2) : int((128-xLen)/2) + xLen 
                         , int((128-yLen)/2): int((128-yLen)/2) + yLen,:] = cimg[j]
            counter = counter + 1
            #plt.imsave('175-'+ str(counter) + '.jpg',Coins[counter-1])
            #print(j)
            #fig= plt.figure()
            #plt.imshow(TempCoins[counter-1])
    Coins = TempCoins[ : counter ,:,:,:]
    #Debugging - Midway Images   
    '''
    fig = plt.figure()
    plt.imshow(local_maxi,cmap = plt.get_cmap('gray'))

    fig = plt.figure()
    plt.imshow(fill,cmap = plt.get_cmap('gray'))

    fig = plt.figure()
    plt.imshow(labels,cmap = plt.get_cmap('jet'))
    '''
    return Coins

def MergeDefectiveBounds(mList):
    ToRet = mList[:]
    TempSliceHolder = mList[:]
    for i,j in enumerate(TempSliceHolder):

        if (not i == 0 and TempSliceHolder[i-1] in ToRet):
            if(((np.abs(TempSliceHolder[i-1][0].start - j[0].start) <= 10 or
                   np.abs(TempSliceHolder[i-1][0].stop - j[0].stop) <= 10) and
                (np.abs(TempSliceHolder[i-1][1].stop - j[1].start) <= 10 or
                 np.abs(TempSliceHolder[i-1][1].start - j[1].stop) <= 10) )):
            
                stop0  = max(TempSliceHolder[i-1][0].start,TempSliceHolder[i-1][0].stop,j[0].start,j[0].stop)
                start0 = min(TempSliceHolder[i-1][0].start,TempSliceHolder[i-1][0].stop,j[0].start,j[0].stop)
                stop1 = max(TempSliceHolder[i-1][1].start,TempSliceHolder[i-1][1].stop,j[1].start,j[1].stop)
                start1 = min(TempSliceHolder[i-1][1].start,TempSliceHolder[i-1][1].stop,j[1].start,j[1].stop)
                mSlice = (slice(start0,stop0),slice(start1,stop1)) 
                ToRet.remove(TempSliceHolder[i-1])
                ToRet.remove(j)
                if(not mSlice in ToRet):
                    ToRet.append(mSlice)
                    
            elif(((np.abs(TempSliceHolder[i-1][1].start - j[1].start) <= 10 or
                   np.abs(TempSliceHolder[i-1][1].stop - j[1].stop) <= 10) and 
                  (np.abs(TempSliceHolder[i-1][0].start - j[0].stop) <= 10 or
                   np.abs(TempSliceHolder[i-1][0].stop - j[0].start) <= 10))):
               
                stop0  = max(TempSliceHolder[i-1][0].start,TempSliceHolder[i-1][0].stop,j[0].start,j[0].stop)
                start0 = min(TempSliceHolder[i-1][0].start,TempSliceHolder[i-1][0].stop,j[0].start,j[0].stop)
                stop1 = max(TempSliceHolder[i-1][1].start,TempSliceHolder[i-1][1].stop,j[1].start,j[1].stop)
                start1 = min(TempSliceHolder[i-1][1].start,TempSliceHolder[i-1][1].stop,j[1].start,j[1].stop)
                mSlice = (slice(start0,stop0),slice(start1,stop1)) 
                ToRet.remove(TempSliceHolder[i-1])
                ToRet.remove(j)
                if(not mSlice in ToRet):
                    ToRet.append(mSlice)
    TempSliceHolder = None
    return ToRet

def SegmentImage(imPath):
    img = image.imread(imPath, mode = 'L')
    cimg = image.imread(imPath)
    img2 = prepareImage(img,cimg)
    return img2
    
#SegmentImage('125.jpg')
