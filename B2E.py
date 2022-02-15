# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:57:42 2021

@author: Eliott Henaut & Nicolas Bioul
"""

import sys
import importlib.util
import os
import re
import shutil
import time

imports=['easygui','dipy','matplotlib.pyplot','nibabel','numpy','cv2','bruker2nifti.converter','optparse']
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

def create_folder(path, replace=True):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

def convertAndMerge(Input, Output, subject, error_file):
    try:
        # instantiate a converter
        bru = Bruker2Nifti(Input, Output, study_name=subject)

        # select the options (attributes) you may want to change - the one shown below are the default one:
        bru.verbose = 0
        # converter settings for the nifti values
        bru.nifti_version = 1
        bru.qform_code = 1
        bru.sform_code = 2
        bru.save_human_readable = True
        bru.save_b0_if_dwi = (True)  # if DWI, it saves the first layer as a single nfti image.
        bru.correct_slope = True
        bru.correct_offset = True
        # advanced sample positioning
        bru.sample_upside_down = False
        bru.frame_body_as_frame_head = False
        # chose to convert extra files:
        bru.get_acqp = False
        bru.get_method = False
        bru.get_reco = False

        # Check that the list of scans and the scans names automatically selected makes some sense:
        #print(bru.scans_list)
        #print(bru.list_new_name_each_scan)
    
        # call the function convert, to convert the study:
        bru.convert()

    except IOError:
        print(Input,"is not a Bruker study")
        error_file.write(Input+" is not a Bruker study! \n")
        return -1

    #Merge all DWI volumes:
    bvecs = None
    bvals = None
    mergedDWI = None

    for dir in os.listdir(os.path.join(Output,subject)):
        basedir = os.path.join(Output,subject,dir)
        if not os.path.isfile(os.path.join(basedir,"acquisition_method.txt")):
            continue
        with open(os.path.join(basedir,"acquisition_method.txt")) as f:
            method = f.readlines()[0]

        if (method == "DtiEpi"):
            print("DtiEpi", dir)
            bvec = np.load(os.path.join(basedir, dir + "_DwGradVec.npy"))
            bval = np.load(os.path.join(basedir, dir + "_DwEffBval.npy"))
            dwi = nib.load(os.path.join(basedir, dir + ".nii.gz"))
            if any(x is None for x in [bvecs, bvals, mergedDWI]):
                bvecs = bvec
                bvals = bval
                mergedDWI = dwi
            else:
                bvecs = np.concatenate((bvecs, bvec), axis=0)
                bvals = np.concatenate((bvals, bval), axis=0)
                mergedDWI = nib.concat_images([mergedDWI, dwi], axis=3)
        elif(method == "FLASH"):
            print("FLASH")
        elif(method == "RARE"):
            print("RARE")
        elif (method == "MSME"):
            print("MSME")
        elif (method == "FieldMap"):
            print("FieldMap")
        elif (method == "nmrsuDtiEpi"):
            print("nmrsuDtiEpi", dir)
            create_folder(os.path.join(Output, subject, "reverse_encoding"), replace=False)
            dwi = nib.load(os.path.join(basedir, dir + ".nii.gz"))
            bval = np.load(os.path.join(basedir,dir + "_DwEffBval.npy"))
            np.savetxt(os.path.join(Output, subject, "reverse_encoding", subject + ".bvec"), np.load(os.path.join(basedir, dir + "_DwGradVec.npy")), fmt="%.42f")
            np.savetxt(os.path.join(Output, subject, "reverse_encoding", subject + ".bval"), bval, newline=' ', fmt="%.42f")
            dwi.to_filename(os.path.join(Output, subject, "reverse_encoding", subject + ".nii.gz"))
            f = open(os.path.join(Output, subject, "reverse_encoding", "nVol.txt"), "w")
            f.write(str(bval.shape[0]))
            f.close()
        else:
            print("Unknow acquisition method:", method)

    np.savetxt(os.path.join(Output, subject, subject + ".bvec"), bvecs, fmt="%.42f")
    np.savetxt(os.path.join(Output, subject, subject + ".bval"), bvals, newline=' ', fmt="%.42f")
    mergedDWI.to_filename(os.path.join(Output, subject, subject+".nii.gz"))
    print("Number of total volumes: ", bvals.shape)

    f = open(os.path.join(Output, subject, "nVol.txt"), "w")
    f.write(str(int(bvals.shape[0])))
    f.close()
    return int(bvals.shape[0])


def link(Input, Out, nVol, subjectName, error_file):
    if nVol<=0:
        pass
    else:
        pattern = re.compile("data_\\d")
        patternNum = re.compile("\\d")
    
        match = False
        dataFolderIndex = []
        for typeFolder in os.listdir(Out):
            if pattern.match(typeFolder):
                nVolDataFolder = int(np.loadtxt(os.path.join(Out,typeFolder,"nVol.txt")))
                dataFolderIndex.append(patternNum.search(typeFolder).group(0))
                if nVol == nVolDataFolder:
                    print("match")
                    match = True
                    break
    
        if not match:
            dataFolderIndex.sort()
            if len(dataFolderIndex) > 0:
                typeFolder = "data_" + str(dataFolderIndex[-1]+1)
            else:
                typeFolder = "data_1"
            create_folder(os.path.join(Out, typeFolder))
            create_folder(os.path.join(Out, typeFolder, "reverse_encoding"))
            index = open(os.path.join(Out, typeFolder, "index.txt"), "w")
            for i in range(nVol):
                index.write('1 ')
            index.close()
            index = open(os.path.join(Out, typeFolder, "reverse_encoding", "index.txt"), "w")
            for i in range(4):
                index.write('1 ')
            index.close()
            f = open(os.path.join(Out,typeFolder,"nVol.txt"), "w")
            f.write(str(nVol))
            f.close()
    
        subjectPath = os.path.join(Input,subjectName)
        shutil.copyfile(subjectPath + ".nii.gz", os.path.join(Out,typeFolder, subjectName + ".nii.gz"))
        shutil.copyfile(subjectPath + ".bvec", os.path.join(Out, typeFolder, subjectName + ".bvec"))
        shutil.copyfile(subjectPath + ".bval", os.path.join(Out, typeFolder, subjectName + ".bval"))
    
        subjectPath_reverse = os.path.join(Input, "reverse_encoding", subjectName)
        shutil.copyfile(subjectPath_reverse + ".nii.gz", os.path.join(Out,typeFolder, "reverse_encoding", subjectName + ".nii.gz"))
        shutil.copyfile(subjectPath_reverse + ".bvec", os.path.join(Out, typeFolder, "reverse_encoding", subjectName + ".bvec"))
        shutil.copyfile(subjectPath_reverse + ".bval", os.path.join(Out, typeFolder, "reverse_encoding", subjectName + ".bval"))

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
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No']:
        replace = False
    else:
        replace = True

    Base = vars(options)['name']
    if Base==None:
        while Base == None or (not os.path.isdir(Base)):
            print("The path of dicom is not correct. pls try again. \n")
            Base = egui.diropenbox()
    else:
        if os.path.exists(Base) and os.path.isdir(Base):
            pass
        else:
            while(not os.path.exists(Base) and not os.path.isdir(Base)):
                Base = egui.diropenbox()    
                
    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'Convert')
    create_folder(BaseOUT,replace)
    ProcIN = os.path.join(Base,'Merge')
    create_folder(ProcIN,replace)
    error_file = open(Base+'/error_file.txt','w+')
    
    """
    fastMode = True #???
    BaseIN = '/Users/eliotthenaut/Desktop/Mémoire/raw/'
    BaseOUT = '/Users/eliotthenaut/Desktop/Mémoire/Convert/'
    ProcIN = '/Users/eliotthenaut/Desktop/Mémoire/Merge/'
    create_folder(BaseOUT, replace=False)
    create_folder(ProcIN, replace=False)
    """
    
    print("Beginning of the loop")
    d=-1
    for file in os.listdir(BaseIN):
        
        if (os.path.isdir(os.path.join(BaseIN, file)) and file != 'cleaning_data'):
            Input = os.path.join(BaseIN, file)
            #print(Input)
            if not os.path.exists(os.path.join(BaseOUT, file)):
                nVol = convertAndMerge(Input, BaseOUT, file, error_file)
            else:
                nVol = int(np.loadtxt(os.path.join(BaseOUT,file,"nVol.txt")))
            link(os.path.join(BaseOUT, file), ProcIN, nVol, file, error_file)
        d=d+1
        if d>=Stop:
            print("break")
            break
        
    print("number of file = ",d)
    
    