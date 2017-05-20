# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:40:28 2017

@author: Elad Dvash
"""

import os
import random

import FeatureDetection
import matplotlib.pyplot as plt
import scipy
import numpy as np

import keras
from keras.layers import Flatten , Dropout , Dense
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.models import load_model

def CreateCNN():
    model = keras.models.Sequential()
    
    # First Cluster
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3) ) )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[metrics.categorical_accuracy])
    model.summary()
    return model
  
def createSegmentedClassImages():
    print('Generating classification ready images')
    if('ClassificationData' in os.listdir() and 
       len(os.listdir(os.getcwd()+'/ClassificationData')) != 0):
        return
    elif(not 'ClassificationData' in os.listdir()): 
        os.mkdir('ClassificationData')
    classificationImages = os.listdir(os.getcwd()+'/all')
    CoinCounter = 0
    for path in classificationImages:
        #print (path)
        Coins = FeatureDetection.SegmentImage('all/'+path)
        for image in Coins:
            plt.imsave('ClassificationData/'+path.split('_')[0] +'_'+str(CoinCounter) + '.jpg',image)
            #print (CoinCounter)
            CoinCounter += 1
            if(CoinCounter%500 == 0 ):
                print('Coins so far:'+str(CoinCounter))
    print('Done!\n')
def createClassDataset():
    if(not 'ClassificationData' in os.listdir()):
        createSegmentedClassImages()
    print('Creating dataset for classification')
    SegmentedImages = os.listdir(os.getcwd()+'/ClassificationData')
    Dataset = np.zeros((len(SegmentedImages),128,128,3),dtype = np.uint8)
    Labels = np.zeros((len(SegmentedImages),5), dtype = np.uint8)
    random.shuffle(SegmentedImages)
    
    for i,path in enumerate(SegmentedImages):
        Dataset[i,:,:,:] = scipy.ndimage.imread('ClassificationData/' + path, mode = 'RGB')
        if(int(path.split('_')[0]) == 5):
            Labels[i,0] = 1
        elif(int(path.split('_')[0]) == 10):
            Labels[i,1] = 1
        elif(int(path.split('_')[0]) == 25):
            Labels[i,2] = 1
        elif(int(path.split('_')[0]) == 50):
            Labels[i,3] = 1
        elif(int(path.split('_')[0]) == 100):
            Labels[i,4] = 1
        
    print(Dataset.shape)
    train_x = Dataset[ :int( Dataset.shape[0] * 0.8) ,:,:,:]
    train_y = Labels[ :int( Dataset.shape[0] * 0.8) ,:]

    test_x = Dataset[int( Dataset.shape[0] * 0.8) : ,:,:,:]
    test_y = Labels[int( Dataset.shape[0] * 0.8) : ,:]
    print('Done!\n')
    return train_x,train_y,test_x,test_y


def GetCoinValue(pos):
    if(pos == 0):
        return 5
    elif(pos == 1):
        return 10
    elif(pos == 2):
        return 25
    elif(pos == 3):
        return 50
    elif(pos == 4):
        return 100

def GetRegResults(saved_weights = ''):
    model = CreateCNN()
    if(saved_weights == ''):
        model.load_weights("coins-weights-improvement-New.h5")
    else:
        model.load_weights(saved_weights)
    
    RegImages = os.listdir(os.getcwd()+'/all_reg')
    FinalData = np.zeros((len(RegImages),2))
    random.shuffle(RegImages)
    AccuracyCounter = 0
    savingCounter = 0
    for i,path in enumerate(RegImages):
        temp_data = FeatureDetection.SegmentImage('all_reg/' + path)
        data = np.zeros((temp_data.shape[0],1,128,128,3))
        data[:,0,:,:,:]= temp_data[:,:,:,:]
        test_y = int(path.split('_')[0])
        sum = 0
        savedPreds = np.zeros((data.shape[0])).astype(int)
        for k in range(data.shape[0]):
            predicted = model.predict(data[k], batch_size = 1)
            predicted = predicted.argmax()
            sum = sum + GetCoinValue(predicted)
            savedPreds[k] = GetCoinValue(predicted)
        #print(sum == test_y)
        FinalData[i] = [sum,test_y]
        if(FinalData[i][0] == FinalData[i][1]):
            AccuracyCounter = AccuracyCounter + 1
            #for k,image in enumerate(temp_data):
            #    plt.imsave('reg_data/'+str(savedPreds[k]) +'_a'+str(savingCounter)+'.jpg',image)
            #    savingCounter = savingCounter + 1
        if((i+1)%200 == 0 or i+1 == len(RegImages)):
            print()
            print('Accuracy = ' + str(100*AccuracyCounter/(i+1))+'%')
            print('Accuracy = ' + str(AccuracyCounter) +'/' + str(i+1))
            print('Progress = ' + str(i+1) +'/' + str(len(RegImages)))
        np.save('Results.npy',FinalData)

def RunClassifier(saved_weights = ''):
    createSegmentedClassImages()
    train_x,train_y,test_x,test_y = createClassDataset()
    filepath="coins-weights-improvement-New.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model = CreateCNN()
    if(not saved_weights == ''):
        model.load_weights(saved_weights)
    model.fit(train_x, train_y, epochs=30, batch_size=96, callbacks=callbacks_list)
    model.save_weights("coins-weights-improvement-Last.h5")
    scores = model.evaluate(test_x, test_y)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    GetRegResults()


#CreateCNN()
RunClassifier()
