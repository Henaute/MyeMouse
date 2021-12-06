#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:22:03 2021

@author: eliotthenaut
"""
import os
import re
import shutil
import elikopy
import json
import time 
import sys
import numpy as np
import nibabel as nib

def create_folder(path,replace=True):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path): 
            os.mkdir(path)
  
def ni_creator(size,out): #out is a path: '/Users/nicolasbioul/Desktop/Thesis/example/data_1/test.nii.gz'
    output=nib.Nifti1Image(np.zeros(size),affine=np.eye(4))
    output.to_filename(out)

def get_name(path_to_file,default,search):
    file=open(path_to_file,'r')
    line=file.readline()
    name=default
    while line:
        if search in line:
            name=file.readline()
            name=name.replace('<','')
            name=name.replace('>','')
            name=name.replace('\n','')
            break
        line=file.readline()
    file.close()
    return name  

def merge(inlist, out):
    if 'T2map_MSME' in inlist[0]:
        img = nib.concat_images(inlist,axis=0)
        nib.save(img, out)
    else:
        img = nib.concat_images(inlist,axis=3)
        nib.save(img, out)
    
def json_merge(jlist,Out):
    for i in range(len(jlist)):
        if '.nii.gz' in jlist[i]:
            jlist[i]=jlist[i].replace('.nii.gz','.json')
        elif '.nii' in jlist[i]: 
            jlist[i]=jlist[i].replace('.nii','.json')
    start='/Groupe'
    end='/'
    name='/Groupe'+(jlist[0].split(start))[1].split(end)[0]
    shutil.copyfile(jlist[0], Out+name+'.json')
    f=open(Out+name+'.json')
    data=json.load(f)
    f.close()
    f=open(Out+name+'.json','w')
    for key in data:
        attribute=[]
        for item in jlist:
            current=open(item)
            current_data=json.load(current)
            attribute.append(current_data[key])
            current.close()
        if not all(x == attribute[0] for x in attribute):
            data[key]=attribute
    json.dump(data, f, ensure_ascii=False, indent=4)
    f.close()