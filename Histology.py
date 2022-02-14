#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:28:37 2021

@author: nicolas bioul & eliott henaut
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
"""       
def Reshape(img,tol=1,mini=10):
    X,Y=img.shape[0],img.shape[1]
    
    xDebut = 0
    xFin = X
    yDebut = 0
    yFin = Y
    
    #TOP
    boolean=True
    j=0
    while boolean==True:
        j+=10
        for i in range(Y):
            if not compare(img[j][i],[255,255,255]): 
                j-=10
                boolean = False
                if j>tol+mini:
                    xDebut=j-tol
                break
            
    #BOTTOM
    boolean=True
    j=0
    while boolean==True:
        j+=10
        for i in range(Y):
            if not compare(img[X-j][i],[255,255,255]): 
                j-=10
                boolean = False
                if j>tol+mini:
                    xFin=X-j+tol
                break
            
    #LEFT
    boolean=True
    j=0
    while boolean==True:
        j+=10
        for i in range(X):
            if not compare(img[i][j],[255,255,255]): 
                j-=10
                boolean = False
                if j>tol+mini:
                    yDebut=j-tol
                break
        
    #RIGHT
    boolean=True
    j=0
    while boolean==True:
        j+=10
        for i in range(X):
            if not compare(img[i][Y-j],[255,255,255]): 
                j-=10
                boolean = False
                if j>tol+mini:
                    yFin=Y-j+tol
                break
    
    img = img[xDebut:xFin,yDebut:yFin,:]
    #plt.imshow(img)
    return img
"""
def Reshape(img,tol=1,incr1=1):
    X,Y=img.shape[0],img.shape[1]
    
    xDebut = 0
    xFin = X
    yDebut = 0
    yFin = Y
    
    if tol<25:
        tol=25
        
    # TOP
    bo = True
    for i in range(Y):
            if not compare(img[tol][i],[255,255,255]): 
                print("tolerance trop grande pour TOP")
                bo = False
                break
    
    incr = 1
    boo = True
    if bo:
        while boo:
            for i in range(Y):
                if not compare(img[tol+incr][i],[255,255,255]): 
                    xDebut = incr-incr1
                    boo = False
                    break
            incr+=incr1
 
    # BOTTOM
    bo = True
    for i in range(Y):
            if not compare(img[X-tol][i],[255,255,255]): 
                print("tolerance trop grande pour BOTTOM")
                bo = False
                break
            
    incr = 1
    boo = True
    if bo:
        while boo:
            for i in range(Y):
                if not compare(img[X-(tol+incr)][i],[255,255,255]): 
                    xFin = X-incr
                    boo = False
                    break
            incr+=incr1
            
    # LEFT
    bo = True
    for i in range(X):
            if not compare(img[i][tol],[255,255,255]): 
                print("tolerance trop grande pour LEFT")
                bo = False
                break
            
    incr = 1
    boo = True
    if bo:
        while boo:
            for i in range(X):
                if not compare(img[i][tol+incr],[255,255,255]): 
                    yDebut = incr-incr1
                    boo = False
                    break
            incr+=incr1

    # RIGHT
    bo = True
    for i in range(X):
            if not compare(img[i][X-tol],[255,255,255]): 
                print("tolerance trop grande pour RIGHT")
                bo = False
                break
            
    incr = 1
    boo = True
    if bo:
        while boo:
            for i in range(X):
                if not compare(img[i][X-(tol+incr)],[255,255,255]): 
                    yFin = Y-incr
                    boo = False
                    break
            incr+=incr1
            
    
    img = img[xDebut:xFin,yDebut:yFin,:]
    
    return img

def is_white(line):
    y=line.shape[0]
    for j in range(y):
        if not compare(line[j],[255,255,255]):
            return False
    return True
                
    
def cut(pathIn,pathOut,n,tol=1000):
    cutl=[0]*(n+1)
    img=cv2.imread(pathIn)
    x,y=img.shape[0],img.shape[1]
    cutl[-1]=x
    part=int(x/n)
    for i in range(1,n):
        if is_white(img[(i*part)]):
            #test 40 AU DESSUS ET EN DESSOUS
            bln=False
            incr=int(x/tol)
            while bln==False and incr>0:
                if is_white(img[(i*part)+incr]):
                    l=img[(i*part)+incr
                    #test 40 AU DESSOUS
                    #TODO
                    bln=True
                if is_white(img[(i*part)-incr]):
                    l=img[(i*part)-incr
                    #test 40 AU DESSUS
                    #TODO
                    bln=True
                incr-=10
            pass
        
        else:
            bln=False
            incr=1
            while bln==False:
                if is_white(img[(i*part)+incr]):
                    l=img[(i*part)+incr
                    #test 40 AU DESSOUS
                    #TODO
                    bln=True
                
                if is_white(img[(i*part)-incr]):
                    l=img[(i*part)-incr
                    #test 40 AU DESSUS
                    #TODO
                    bln=True
                incr+=1

"""
def cut(pathIn,pathOut,n,tol=1000):
    cutl=[0]*(n+1)
    img=cv2.imread(pathIn)
    x,y=img.shape[0],img.shape[1]
    cutl[-1]=x
    part=int(x/n)
    for i in range(1,n):
        boo=True
        incr=0
        p=0
        con=0
        while boo:
            bo=True
            for j in range(y):
                if not compare(img[(i*part)+incr][j],[255,255,255]):
                    boo=True
                    bo=False
                    break
            if bo==True:
                con+=1
            if con>=int(x/tol):
                boo=False   
            p+=1
            incr=(-1)**p*int(p/2)
        cutl[i]=(i*part)+incr
    for i in range(n):
        cv2.imwrite(pathOut+'/Cut_'+str(i+1)+'.tiff',Reshape(img[cutl[i]:cutl[i+1],:,:],int(x/tol),10))
"""
cut('/Users/nicolasbioul/Desktop/Thesis/Histological_data/1/1_Wholeslide_Default_Extended.tiff','/Users/nicolasbioul/Desktop/Thesis/split_histo',3)
