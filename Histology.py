#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:28:37 2021

@author: nicolasbioul
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2

def compare(a,b):
    if len(a)!=len(b):
        return False
    else:
        for i in range(len(a)):
            if a[i]!=b[i]:
                return False
    return True

def WhiteLine(img,L):
    y = img.shape[1]
    for i in range(y):
        if not compare(img[L][i],[255,255,255]):
                   return False
    return True

def WhiteCol(img,C):
    x = img.shape[0]
    for i in range(x):
        if not compare(img[i][C],[255,255,255]):
                   return False

    return True

def Reshape(img,tol=1):
    X,Y=img.shape[0],img.shape[1]
    
    xDebut = 0
    xFin = X
    yDebut = 0
    yFin = Y
    
    if tol<30:
        tol=30
        
    # TOP
    if WhiteLine(img, tol):
        
        incr = 1
        boo = True

        while boo:
            print("top",incr)
            if not WhiteLine(img, tol+incr):
                xDebut = incr
                boo = False
            incr+=int(tol/10)
 
    # BOTTOM
    if WhiteLine(img,X-tol):
        
        incr = 1
        boo = True

        while boo:
            print("bottom",incr)
            if not WhiteLine(img, X-(tol+incr)):
                xFin = X-incr
                boo = False
            incr+=int(tol/10)

            
    # LEFT
    if WhiteCol(img, tol):
        
        incr = 1
        boo = True

        while boo:
            print("Left",incr)
            if not WhiteCol(img, tol+incr):
                yDebut = incr
                boo = False
            incr+=int(tol/10)

    # RIGHT
    if WhiteCol(img, Y-tol):
        
        incr = 1
        boo = True

        while boo:
            print("Right",incr)
            if not WhiteCol(img, Y-(tol+incr)):
                yFin = Y-incr
                boo = False
            incr+=int(tol/10)
    
    img = img[xDebut:xFin,yDebut:yFin,:]
    
    return img
        

    
def cut(pathIn,pathOut,n,ptol=1000):
    cutl=np.zeros(n+1,dtype=int)
    img=cv2.imread(pathIn)
    x=img.shape[0]
    cutl[n]=x
    part=int(x/n)
    tol =int(x/ptol)
    
    for i in range(1,n):
        
        boo=True
        incr=0
        p=-1
        while boo:
            p+=1      
            incr=(-1)**p*int(p/2)*int(tol/10)
            #print(incr)
            
            if WhiteLine(img, (i*part)+incr):
                if not WhiteLine(img, (i*part)+incr+tol):
                    incr = incr-tol
                elif not WhiteLine(img, (i*part)+incr-tol):
                    incr = incr+tol
                else:
                    boo = False
        
        cutl[i] = (i*part)+incr
        
    for i in range(n):
        range(n)
        print(cutl[i])
        cv2.imwrite(pathOut+'/Cut_'+str(i)+'.tiff',Reshape(img[cutl[i]:cutl[i+1],:,:],tol))
    
    
    
    
cut('/Users/nicolasbioul/Desktop/Thesis/Histological_data/1/1_Wholeslide_Default_Extended.tif','/Users/nicolasbioul/Desktop/Thesis/split_histo/',3)
 
    
