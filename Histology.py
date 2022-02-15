#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:28:37 2021

@author: nicolas bioul & eliott henaut
"""

import sys
import importlib.util
import os
import re
import shutil
import time

imports=['easygui','dipy','matplotlib.pyplot','os','re','nibabel','numpy','cv2','bruker2nifti.converter','optparse']
for name in imports:
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")
    
import easygui as egui
import dipy as dp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from bruker2nifti.converter import Bruker2Nifti
from optparse import OptionParser

def compare(a,b):
    a,b
    if len(a)!=len(b):
        return False
    else:
        for i in range(len(a)):
            if a[i]!=b[i]:
                return False
    return True

def WhiteLine(img,L):
    y = img.shape[1]
    L
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
            if not WhiteLine(img, tol+incr):
                xDebut = incr
                boo = False
            incr+=tol//5
            print('Top: ',tol+incr)
 
    # BOTTOM
    if WhiteLine(img,X-tol):
        
        incr = 1
        boo = True

        while boo:
            if not WhiteLine(img, X-(tol+incr)):
                xFin = X-incr
                boo = False
            incr+=tol//5
            print('Bottom:',X-(tol+incr))

            
    # LEFT
    if WhiteCol(img, tol):
        
        incr = 1
        boo = True

        while boo:
            if not WhiteCol(img, tol+incr):
                yDebut = incr
                boo = False
            incr+=tol//5
            print('LEFT: ',tol+incr)

    # RIGHT
    if WhiteCol(img, Y-tol):
        
        incr = 1
        boo = True

        while boo:
            if not WhiteCol(img, Y-(tol+incr)):
                yFin = Y-incr
                boo = False
            incr+=tol//10
            print('RIGHT: ',Y-(tol+incr))
    
    img = img[xDebut:xFin,yDebut:yFin,:]
    
    return img
        

    
def cut(pathIn,pathOut,n,ptol=1000):
    cutl=np.zeros(n+1,dtype=int)
    img=cv2.imread(pathIn)
    x=img.shape[0]
    cutl[-1]=x
    part=x//n
    tol =x//ptol
    
    for i in range(1,n):
        boo=True
        incr=0
        p=-1
        while boo:
            p+=1      
            incr=(-1)**p*int(p/2)*int(tol/5)

            if WhiteLine(img, (i*part)+incr):
                if not WhiteLine(img, (i*part)+incr+tol):
                    incr = max(incr-tol,0)
                elif not WhiteLine(img, (i*part)+incr-tol):
                    incr = incr+tol
                else:
                    boo = False
        
        cutl[i] = (i*part)+incr
    print(cutl)  
    for i in range(n):
        out=pathOut+'/Cut_'+str(i)+'.tiff'
        print(i)
        time.sleep(3)
        cv2.imwrite(out,Reshape(img[cutl[i]:cutl[i+1],:,:],tol))
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Beginning of main")
    parser = OptionParser()
    parser.add_option('-n','--name',dest = 'name',
                      help='path of data')
    parser.add_option('-r','--replace',dest = 'replace',
                      help='replace data True/False')
    parser.add_option('-s','--Stop',dest = 'Stop',
                      help='number of file convert')
    #parser.add_option('-e','--elikopy',dest = 'elikopy',
    #                  help='pre-processing data with elikopy: True/False')
    
    (options,args) = parser.parse_args()
    
    Stop = vars(options)['Stop']
    if Stop!=None and int(Stop)>=1:
        Stop = int(Stop)
    else:
        Stop = 100

    replace = vars(options)['replace']
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No','N']:
        replace = False
    else:
        replace = True

    Base = vars(options)['name']
    if type(Base)==type(None):
        while Base == None or (not os.path.isdir(Base)):
            print("The path of dicom is not correct. pls try again. \n")
            Base = egui.diropenbox()
    else:
        if os.path.exists(Base) and os.path.isdir(Base):
            pass
        else:
            while(not os.path.exists(Base) and not os.path.isdir(Base)):
                Base = egui.diropenbox()    
    
    print("Beginning of the loop")
    Output=os.path.join(Base,'Cuts')
    if replace==False and os.path.isdir(Output):
        pass
    elif replace==True and os.path.isdir(Output):
        shutil.rmtree(Output)
        os.mkdir(Output)
    else:
        os.mkdir(Output)
    
    for file in os.listdir(Base):
        if (os.path.isdir(os.path.join(Base, file)) and file != '.DS_Store' and file != 'Cuts'):
            Local_Output=os.path.join(Output,file)
            if not os.path.isdir(Local_Output):
                os.mkdir(Local_Output)
            Input = os.path.join(Base, file)
            for im in os.listdir(Input):
                if (im != '.DS_Store') and '.tif' in im:
                    cut(os.path.join(Input,im),Local_Output,3)
                    print(im,' has been cut and placed in directory: ', Local_Output)
            