#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:22:03 2021

@author: eliotthenaut and nicolasbioul
"""
import os
import shutil
import json
import time 
import numpy as np
import nibabel as nib
from termcolor import colored

def create_folder(path,replace=True):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)
        try:
            os.mkdir(path)
        except OSError:
            print("Directory creation %s failed" % path)
    else:
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Directory creation %s failed" % path)

def ni_creator(data,output): #out is a path: '/Users/nicolasbioul/Desktop/Thesis/example/data_1/test.nii.gz'
    out=nib.Nifti1Image(np.zeros(data),affine=np.eye(4))
    out.to_filename(output)

def ni_writer(data,output):
    data.to_filename(output)

def merge(inlist, out_file):
    assert len(inlist)==0, "Your list is empty"
    assert os.path.isfile(out_file), "Your output file appears inexistant. Please check the path"

    if 'T2map_MSME' in inlist[0]:
        try:
            img = nib.concat_images(inlist,axis=0)
            nib.save(img, out_file)
        except:
            shutil.rmtree(out_file)
            out_file.replace('.nii.gz','_error.txt')
            print(colored('Error file generated: T2 merging failed','red'))
            time.sleep(2)
            f=open(out_file,'w+')
            f.write('An error occured during merging multi T2. Please check the input data in order to retry merging. Common errors in merging include wrong dimensions in files, empty files or corrupted files given as an argument.')
            f.close()
    else:
        try:
            img = nib.concat_images(inlist,axis=3)
            nib.save(img, out_file)
        except:
            shutil.rmtree(out_file)
            out_file.replace('.nii.gz','_error.txt')
            print(colored('Error file generated: Diffusion merging failed','red'))
            time.sleep(2)
            f=open(out_file,'w+')
            f.write('An error occured during diffusion merging. Please check the input data in order to retry merging. Common errors in merging include wrong dimensions in files, empty files or corrupted files given as an argument.')
            f.close()

def json_merge(jlist,Out):

    assert len(jlist)==0, "Your list is empty" 
    assert os.path.isdir(Out), "Your output directory appears inexistant. Please check the path"
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
