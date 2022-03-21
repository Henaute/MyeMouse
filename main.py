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
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
imports=['easygui','dipy','matplotlib.pyplot','nibabel','numpy','opencv-python','bruker2nifti.converter','optparse','nipype','nilearn','datetime','sklearn','threading']
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
        print(f"can't find the {name!r} module--> pip install")
        try:
            install(name)
            
        except:
            print('The module "',name,'" you are trying to install doesnt seem to exist. Please check!!' )
            sys.exit()
    
import easygui as egui
import dipy as dp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from bruker2nifti.converter import Bruker2Nifti
from optparse import OptionParser
import nipype
import nilearn
import json
import datetime as dt
import elikopy


def create_folder(path,logs,replace=True):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)
            logs.write('You have chosen to replace the existing directory. The old directory has been removed'+'\n')
        os.mkdir(path)
        logs.write('New directory has been created at '+ path+'\n')
    else:
        logs.write('You have chosen not to replace the directory'+'\n')
        if not os.path.exists(path):
            logs.write('Creating the directory '+ path +'\n')
            os.mkdir(path)

from myemouse_preproc import preprocessing

def convertAndMerge(Input, Output, subject,logs):
    # instantiate a converter
    print("Start of convertAndMerge for subject"+ subject+'\n')
    logs.write('Subject'+subject+' from'+ Input+' has been loaded to convert and merge function'+ str(dt.datetime.now())+'\n')
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
    print(bru.scans_list)
    print(bru.list_new_name_each_scan)

    # call the function convert, to convert the study:
    print("\n","[convertAndMerge]", "Conversion step initiated")
    logs.write('\n'+'[convertAndMerge]  '+ ' Conversion step initiated'+ str(dt.datetime.now())+'\n')
    logs.write('Any problems accuring between here and the next log issue from the bruker2nifti library\n')
    bru.convert()
    logs.write('\n'+'[convertAndMerge]  '+' Conversion step complete '+str(dt.datetime.now())+'\n')
    #Merge all DWI volumes:
    bvecs = None
    bvals = None
    mergedDWI = None
    shell_index = []

    print("\n","[convertAndMerge]", 'Merging step initiated')
    logs.write('\n'+'[convertAndMerge]  '+' Merging step initiated '+str(dt.datetime.now())+'\n')
    for dir in os.listdir(os.path.join(Output,subject)):
        basedir = os.path.join(Output,subject,dir)
        if not os.path.isfile(os.path.join(basedir,"acquisition_method.txt")):
            continue
        with open(os.path.join(basedir,"acquisition_method.txt")) as f:
            method = f.readlines()[0]

        if (method == "DtiEpi"):
            print("DtiEpi", dir)
            logs.write('[convertAndMerge]  Treating DtiEpi at '+dir+'   '+str(dt.datetime.now())+'\n')
            bvec = np.load(os.path.join(basedir, dir + "_DwGradVec.npy"))
            bval = np.load(os.path.join(basedir, dir + "_DwEffBval.npy"))
            dwi = nib.load(os.path.join(basedir, dir + ".nii.gz"))
            zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
            bvec[zerobvec] = [1, 0, 0]
            if any(x is None for x in [bvecs, bvals, mergedDWI]):
                bvecs = bvec
                bvals = bval
                mergedDWI = dwi
                shell_index.append(0)
            else:
                bvecs = np.concatenate((bvecs, bvec), axis=0)
                bvals = np.concatenate((bvals, bval), axis=0)
                shell_index.append(mergedDWI.shape[-1])
                mergedDWI = nib.concat_images([mergedDWI, dwi], axis=3)
        elif(method == "FLASH"):
            print("FLASH")
            logs.write('[convertAndMerge] Treating FLASH at'+dir+'   '+str(dt.datetime.now())+'\n')
        elif(method == "RARE"):
            print("RARE")
            logs.write('[convertAndMerge] Treating RARE at'+dir+'   '+str(dt.datetime.now())+'\n')
        elif (method == "MSME"):
            print("MSME")
            logs.write('[convertAndMerge] Treating MSME at'+dir+'   '+str(dt.datetime.now())+'\n')
        elif (method == "FieldMap"):
            print("FieldMap")
            logs.write('[convertAndMerge] Treating FieldMap at'+dir+'   '+str(dt.datetime.now())+'\n')
        elif (method == "nmrsuDtiEpi"):
            print("nmrsuDtiEpi", dir)
            logs.write('[convertAndMerge] Treating nmrsuDtiEpi at'+dir+'   '+str(dt.datetime.now())+'\n')
            create_folder(os.path.join(Output, subject, "reverse_encoding"), replace=False,logs=logs)
            dwi = nib.load(os.path.join(basedir, dir + ".nii.gz"))
            bval = np.load(os.path.join(basedir,dir + "_DwEffBval.npy"))
            bvec = np.load(os.path.join(basedir, dir + "_DwGradVec.npy"))
            zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
            bvec[zerobvec] = [1, 0, 0]
            np.savetxt(os.path.join(Output, subject, "reverse_encoding", subject + ".bvec"), bvec, fmt="%.42f")
            np.savetxt(os.path.join(Output, subject, "reverse_encoding", subject + ".bval"), bval, newline=' ', fmt="%.42f")
            dwi.to_filename(os.path.join(Output, subject, "reverse_encoding", subject + ".nii.gz"))
            f = open(os.path.join(Output, subject, "reverse_encoding", "nVol.txt"), "w")
            f.write(str(bval.shape[0]))
            f.close()
        else:
            print("Unknow acquisition method:", method)
            logs.write('[convertAndMerge]'+dir+'  is an unknow acquisition method  '+str(dt.datetime.now())+'\n')
            

    np.savetxt(os.path.join(Output, subject, subject + ".bvec"), bvecs, fmt="%.42f")
    np.savetxt(os.path.join(Output, subject, subject + ".bval"), bvals, newline=' ', fmt="%.42f")
    mergedDWI.to_filename(os.path.join(Output, subject, subject+".nii.gz"))
    print("Number of total volumes: ", bvals.shape)

    f = open(os.path.join(Output, subject, "nVol.txt"), "w")
    f.write(str(int(bvals.shape[0])))
    f.close()

    f = open(os.path.join(Output, subject, "shell_index.txt"), "w")
    for element in shell_index:
        f.write(str(element) + "\n")
    f.close()
    logs.write('[convertAndMerge] ran successfully \n')
    return int(bvals.shape[0])


def link(Input, Out, nVol, subjectName,logs):
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
        create_folder(os.path.join(Out, typeFolder),logs=logs)
        create_folder(os.path.join(Out, typeFolder, "reverse_encoding"),logs=logs)
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
        shutil.copyfile(os.path.join(Input, "shell_index.txt"), os.path.join(Out,typeFolder,"shell_index.txt"))

    subjectPath = os.path.join(Input,subjectName)
    shutil.copyfile(subjectPath + ".nii.gz", os.path.join(Out,typeFolder, subjectName + ".nii.gz"))
    shutil.copyfile(subjectPath + ".bvec", os.path.join(Out, typeFolder, subjectName + ".bvec"))
    shutil.copyfile(subjectPath + ".bval", os.path.join(Out, typeFolder, subjectName + ".bval"))

    subjectPath_reverse = os.path.join(Input, "reverse_encoding", subjectName)
    shutil.copyfile(subjectPath_reverse + ".nii.gz", os.path.join(Out,typeFolder, "reverse_encoding", subjectName + ".nii.gz"))
    shutil.copyfile(subjectPath_reverse + ".bvec", os.path.join(Out, typeFolder, "reverse_encoding", subjectName + ".bvec"))
    shutil.copyfile(subjectPath_reverse + ".bval", os.path.join(Out, typeFolder, "reverse_encoding", subjectName + ".bval"))


def gen_Nifti(BaseIN,BaseOUT,ProcIN,fastMode,logs):
    """
    :param BaseIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/raw"
    :param BaseOUT: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/bruker_elikopy"
    :param ProcIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/test_b2e_study"
    :param fastMode: True
    :return:
    """

    create_folder(BaseOUT, logs,replace=False)
    create_folder(ProcIN, logs,replace=False)
    print("Beginning of the gen_Nifti loop")
    logs.write('The Nifti generating function has been launched '+str(dt.datetime.now())+'\n')
    for file in os.listdir(BaseIN):
        if (os.path.isdir(os.path.join(BaseIN, file)) and file != 'cleaning_data'):
            Input = os.path.join(BaseIN, file)
            print(Input)
            if not (fastMode and os.path.exists(os.path.join(BaseOUT, file))):
                logs.write('Fast mode off\n')
                nVol = convertAndMerge(Input, BaseOUT, file,logs)
                logs.write('Now launching link function'+str(dt.datetime.now())+'\n')
            else:
                logs.write('Fast mode on\n')
                nVol = int(np.loadtxt(os.path.join(BaseOUT,file,"nVol.txt")))
                logs.write('Launching link function  '+str(dt.datetime.now())+'\n')
            link(os.path.join(BaseOUT, file), ProcIN, nVol, file,logs)
            logs.write('link function ended successully  '+str(dt.datetime.now())+'\n')
            

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Beginning of main")
    dic_path = "/CECI/proj/pilab/Software/elikopy_static_files/mf_dic/dictionary_fixed_rad_dist_StLuc-GE.mat"

    fastMode = True
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
    logs = open(Base+'/logs.txt','w+')
    logs.write('Program launched at'+ str(dt.datetime.now())+'\n')
    
    Stop = vars(options)['Stop']
    if Stop!=None and int(Stop)>=1:
        Stop = int(Stop)
    else:
        Stop = 100

    replace = vars(options)['replace']
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No']:
        replace = False
        logs.write('You have chosen not to replace the data\n')
    else:
        replace = False
        logs.write('You have chosen to replace the data\n')

   
                
    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'Convert')
    create_folder(BaseOUT,logs,replace)
    ProcIN = os.path.join(Base,'Merge')
    create_folder(ProcIN,logs,replace)
    

    patient_list = None

    gen_Nifti(BaseIN,BaseOUT,ProcIN,fastMode,logs)

    study = elikopy.core.Elikopy(ProcIN, slurm=False, slurm_email='name.surname@student.uclouvain.be', cuda=False)

    # Generate elikopy architecture
    study.patient_list()

    # Preprocessing: Motion Correction, Brain extraction, (TODO: Topup)
    study.patientlist_wrapper(preprocessing, {}, folder_path=ProcIN, patient_list_m=None, filename="myemouse_preproc",
                            function_name="preprocessing", slurm=False, slurm_timeout=None, cpus=None,
                            slurm_mem=None)

    # Microstructural metrics
    study.dti(patient_list_m=patient_list,report=False)
    #study.noddi(use_wm_mask=False, patient_list_m=patient_list, cpus=4)
    study.fingerprinting(dic_path, patient_list_m=patient_list, cpus=8, CSD_bvalue=6000,report=False)
    logs.close()