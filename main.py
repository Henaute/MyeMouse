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
import subprocess


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

import nibabel as nib
import numpy as np
from bruker2nifti.converter import Bruker2Nifti
from optparse import OptionParser
import datetime as dt
import elikopy

def write(file_path,message):
    f=open(file_path,'a+')
    f.write(message)
    f.close()
    
def create_folder(path,logs,replace=True):
    if replace:
        if os.path.exists(path):
            print(path)
            try:
                shutil.rmtree(path)
                write(logs,'You have chosen to replace the existing directory. The old directory has been removed'+'\n')
            except FileNotFoundError:
                write(logs,'WARNING!!   '+BaseOUT+' could not be removed from your computer!! Try deleting it manually')
                quit()

        os.mkdir(path)
        write(logs,'New directory has been created at '+ path+'\n')
    else:
        write(logs,'You have chosen not to replace '+path+'\n')
        if not os.path.exists(path):
            write(logs,'Creating the directory '+ path +'\n')
            os.mkdir(path)

from myemouse_preproc import preprocessing

def convertAndMerge(Input, Output, subject,logs):
    # instantiate a converter
    print("Start of convertAndMerge for subject"+ subject+'\n')
    write(logs,'Subject'+subject+' from'+ Input+' has been loaded to convert and merge function  '+ str(dt.datetime.now())+'\n')
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
    write(logs,'\n'+'[convertAndMerge]  '+ ' Conversion step initiated'+ str(dt.datetime.now())+'\n')
    write(logs,'[convertAndMerge] Any problems occuring now until the next log issue from the bruker2nifti library\n')
    bru.convert()
    write(logs,'\n'+'[convertAndMerge]  '+' Conversion step complete '+str(dt.datetime.now())+'\n')
    #Merge all DWI volumes:
    bvecs = None
    bvals = None
    mergedDWI = None
    shell_index = []

    print("\n","[convertAndMerge]", 'Merging step initiated')
    write(logs,'\n'+'[convertAndMerge]  '+' Merging step initiated '+str(dt.datetime.now())+'\n')
    for dirr in os.listdir(os.path.join(Output,subject)):
        basedir = os.path.join(Output,subject,dirr)
        if not os.path.isfile(os.path.join(basedir,"acquisition_method.txt")):
            continue
        with open(os.path.join(basedir,"acquisition_method.txt")) as f:
            method = f.readlines()[0]

        if (method == "DtiEpi"):
            print("DtiEpi", dirr)
            write(logs,'[convertAndMerge]  Treating DtiEpi at '+dirr+'   '+str(dt.datetime.now())+'\n')
            bvec = np.load(os.path.join(basedir, dirr + "_DwGradVec.npy"))
            bval = np.load(os.path.join(basedir, dirr + "_DwEffBval.npy"))
            dwi = nib.load(os.path.join(basedir, dirr + ".nii.gz"))
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
            print("FLASH",dirr)
            write(logs,'[convertAndMerge] Treating FLASH at'+dirr+'   '+str(dt.datetime.now())+'\n')
        elif(method == "RARE"):
            print("RARE",dirr)
            write(logs,'[convertAndMerge] Treating RARE at'+dirr+'   '+str(dt.datetime.now())+'\n')
        elif (method == "MSME"):
            print("MSME",dirr)
            write(logs,'[convertAndMerge] Treating MSME at'+dirr+'   '+str(dt.datetime.now())+'\n')
        elif (method == "FieldMap"):
            print("FieldMap",dirr)
            write(logs,'[convertAndMerge] Treating FieldMap at'+dirr+'   '+str(dt.datetime.now())+'\n')
        elif (method == "nmrsuDtiEpi"):
            print("nmrsuDtiEpi", dirr)
            write(logs,'[convertAndMerge] Treating nmrsuDtiEpi at'+dirr+'   '+str(dt.datetime.now())+'\n')
            create_folder(os.path.join(Output, subject, "reverse_encoding"), replace=False,logs=logs)
            dwi = nib.load(os.path.join(basedir, dirr + ".nii.gz"))
            bval = np.load(os.path.join(basedir,dirr + "_DwEffBval.npy"))
            bvec = np.load(os.path.join(basedir, dirr + "_DwGradVec.npy"))
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
            write(logs,'[convertAndMerge]'+dirr+'  is an unknow acquisition method  '+str(dt.datetime.now())+'\n')
            

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
    write(logs,'[convertAndMerge] ran successfully \n')
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
    


def gen_Nifti(BaseIN,BaseOUT,ProcIN,logs):
    """
    :param BaseIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/raw"
    :param BaseOUT: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/bruker_elikopy"
    :param ProcIN: "/CECI/proj/pilab/PermeableAccess/souris_MKF3Hp7nU/test_b2e_study"
    """
    print("[gen_Nifti] Beginning of the function "+str(dt.datetime.now()))
    write(logs,'The Nifti generating function has been launched \n')
    for file in os.listdir(BaseIN):
        if (os.path.isdir(os.path.join(BaseIN, file)) and file != 'cleaning_data'):
            Input = os.path.join(BaseIN, file)
            if not (os.path.exists(os.path.join(ProcIN,'subjects',file,'dMRI','raw')) and len(os.listdir(os.path.join(ProcIN,'subjects',file,'dMRI','raw')))>=3):
                nVol = convertAndMerge(Input, BaseOUT, file,logs)
                write(logs,'Now launching link function  for '+file+'  '+str(dt.datetime.now())+'\n')
                
                link(os.path.join(BaseOUT, file), ProcIN, nVol, file,logs)
                write(logs,'Link function ended successully  for '+file+'  '+str(dt.datetime.now())+'\n')
                

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Beginning of main")


    parser = OptionParser()
    parser.add_option('-n','--name',dest = 'name',
                      help='path of data')
    parser.add_option('-r','--replace',dest = 'replace',
                      help='replace data True/False')
    parser.add_option('-p','--preprocessing',dest = 'preprocessing',
                      help='preprocessing data True/False')
    
    (options,args) = parser.parse_args()
    
    Base = vars(options)['name']
    if Base==None:
        while Base == None or (not os.path.isdir(Base)):
            print("The path of dicom is not correct. Please select the directory containing the raw dicoms. \n")
            Base = egui.diropenbox()
    else:
        if os.path.exists(Base) and os.path.isdir(Base):
            pass
        else:
            while(not os.path.exists(Base) and not os.path.isdir(Base)):
                print("Your path neither exists or isn't a directory,  please select a directory with a bruker study. The directory must contain the raw directory \n")
                Base = egui.diropenbox()
    f = open(Base+'/logs.txt','w+')
    logs=Base+'/logs.txt'
    f.write('Program launched at'+ str(dt.datetime.now())+'\n')
    f.close()

    replace = vars(options)['replace']
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No','N','n']:
        replace = False
        write(logs,'You have chosen not to replace the data\n')
    else:
        replace = True
        write(logs,'You have chosen to replace the data\n')
         
    preproc = vars(options)['preprocessing']
    if preproc in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No','N','n']:
        preproc = False
        write(logs,'You have chosen not to preprocess the data\n')
    else:
        preproc = True
        write(logs,'You have chosen to preprocess the data\n')
          
    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'Convert')
    create_folder(BaseOUT,logs,replace)
    ProcIN = os.path.join(Base,'Merge')
    create_folder(ProcIN,logs,replace)
    

    patient_list = None

    gen_Nifti(BaseIN,BaseOUT,ProcIN,logs)
    write(logs,'[gen_Nifti] has ended '+str(dt.datetime.now()))
    
    study = elikopy.core.Elikopy(ProcIN, slurm=False, slurm_email='name.surname@student.uclouvain.be', cuda=False)

    # Generate elikopy architecture
    study.patient_list()
    if os.path.exists(BaseOUT) and len(os.listdir(BaseOUT))!=0:
        for subject in os.listdir(BaseOUT):
          sub=os.path.join(BaseOUT,subject)
          new_subjectName=subject
          for acquisition in os.listdir(sub):
              if os.path.isdir(os.path.join(sub,acquisition)) and acquisition+'_visu_pars.txt' in os.listdir(os.path.join(sub,acquisition)):
                  try:
                      txt=open(os.path.join(sub,acquisition,acquisition+'_visu_pars.txt'),'r')
                  except FileNotFoundError:
                      pass
                  line=txt.readline()
                  while line:
                     if 'VisuStudyId' in line:
                            new_subjectName=line.split(' ',4)[2]
                            break
                     line=txt.readline()
                  txt.close()
                  break
          ntxt=open(os.path.join(ProcIN,'subjects',subject,subject+'.txt'),'w')
          ntxt.write(new_subjectName)
          ntxt.close()
          for acq in os.listdir(sub):
              acqpath=os.path.join(sub,acq)
              if os.path.isdir(acqpath) and acq!='reverse_encoding' and os.path.isfile(os.path.join(acqpath,"acquisition_method.txt")):
                  with open(os.path.join(acqpath,"acquisition_method.txt")) as f:
                      method = f.readlines()[0]
                  
                  if(method == "FLASH"):
                      create_folder(os.path.join(ProcIN,'subjects',subject,"FLASH"),logs,False)
                      if acq not in os.listdir(os.path.join(ProcIN,'subjects',subject)):
                          shutil.move(acqpath,os.path.join(ProcIN,'subjects',subject,"FLASH"))
                          write(logs,"FLASH acquisition added to  "+ProcIN+'/subjects'+subject+'\n')
                  
                  elif(method == "RARE"):
                      create_folder(os.path.join(ProcIN,'subjects',subject,"RARE"),logs,False)
                      if acq not in os.listdir(os.path.join(ProcIN,'subjects',subject)):
                          shutil.move(acqpath,os.path.join(ProcIN,'subjects',subject,"RARE"))
                          write(logs,"RARE acquisition added to  "+ProcIN+'/subjects'+subject+'\n')
    
    
                  elif (method == "MSME"):
                      create_folder(os.path.join(ProcIN,'subjects',subject,"MSME"),logs,False)
                      if acq not in os.listdir(os.path.join(ProcIN,'subjects',subject)):
                          shutil.move(acqpath,os.path.join(ProcIN,'subjects',subject,"MSME"))
                          write(logs,"MSME acquisition added to  "+ProcIN+'/subjects'+subject+'\n')
    
    
                  elif (method == "FieldMap"):
                      create_folder(os.path.join(ProcIN,'subjects',subject,"FieldMap"),logs,False)
                      if acq not in os.listdir(os.path.join(ProcIN,'subjects',subject)):
                          shutil.move(acqpath,os.path.join(ProcIN,'subjects',subject,"FieldMap"))
                          write(logs,"FieldMap acquisition added to  "+ProcIN+'/subjects'+subject+'\n')
    
              elif acq=='reverse_encoding':
                  if acq not in os.listdir(os.path.join(ProcIN,'subjects',subject)):
                      shutil.move(acqpath,os.path.join(ProcIN,'subjects',subject))
                      write(logs,"reverse_encoding acquisition added to  "+ProcIN+'/subjects'+subject+'\n')
              else:
                  if os.path.isdir(acqupath):
                      write(logs,"⚠️⚠️⚠️WARNING!!! "+acqpath+" doesn't seem to contain an aquisition_method.txt \n")

                  
    
        write(logs,'Conversion complete  '+str(dt.datetime.now()))
    else:
        write(logs,'Replace mode = off! Fast forwarding to preprocessing  '+str(dt.datetime.now()))

    if os.path.isdir(BaseOUT):
        try:
            shutil.rmtree(BaseOUT)
            write(logs,BaseOUT+' was removed from your computer')
        except FileNotFoundError:
            write(logs,'WARNING!!   '+BaseOUT+' could not be removed from your computer!! Try deleting it manually')
            

    # Preprocessing: Motion Correction, Brain extraction,
    if preproc:
        if replace:
            study.patientlist_wrapper(preprocessing, {}, folder_path=ProcIN, patient_list_m=None, filename="myemouse_preproc",
                                            function_name="preprocessing", slurm=False, slurm_timeout=None, cpus=None,
                                            slurm_mem=None)
        else:
            for dirr in os.listdir(ProcIN+'/subjects'):
                if os.path.isdir(ProcIN+'/subjects/'+dirr) and os.path.exists(ProcIN+'/subjects/'+dirr+'/dMRI/preproc/'+dirr+'_dmri_preproc.bval') and os.path.exists(ProcIN+'/subjects/'+dirr+'/dMRI/preproc/'+dirr+'_dmri_preproc.bvec') and os.path.exists(ProcIN+'/subjects/'+dirr+'/dMRI/preproc/'+dirr+'_dmri_preproc.nii.gz'):
                    continue
                elif os.path.isdir(ProcIN+'/subjects/'+dirr):
                    study.patientlist_wrapper(preprocessing, {}, folder_path=ProcIN, patient_list_m=None, filename="myemouse_preproc",
                                            function_name="preprocessing", slurm=False, slurm_timeout=None, cpus=None,
                                            slurm_mem=None)
                    
                    
        # Microstructural metrics
        dic_path = '/Volumes/LaCie/Thesis/fingerprinting/dictionary_fixed_rad_dist_Bruker_StLuc.mat'
        
        study.dti(patient_list_m=patient_list)
        study.noddi(use_wm_mask=False, patient_list_m=patient_list, cpus=4)
        study.fingerprinting(dic_path, patient_list_m=patient_list, cpus=8, CSD_bvalue=6000)
        
        logs.close()
