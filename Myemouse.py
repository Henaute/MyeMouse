#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:57:42 2021

@author: Eliott Henaut & Nicolas Bioul
"""
import utils as u
import Elikopy as Ep
import os
import re
import shutil
import time 
from termcolor import colored
from optparse import OptionParser
from bruker2nifti.converter import Bruker2Nifti

  

def convert(Input,Out,name,error_file):

    # instantiate a converter
    bru = Bruker2Nifti(Input, Out, study_name=name)

    # select the options (attributes) you may want to change - the one shown below are the default one:
    bru.verbose = 0
    # converter settings for the nifti values
    bru.nifti_version = 2
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
    
    

def link(Input,OutDti,OutT2,name,error_file):
    list_Dti=[]
    O1 = os.path.join(OutDti,name)
    #O2 = os.path.join(OutT2,name)
    fbvec = open(O1+'.bvec', 'w+')  # open file bvec in write mode
    os.chmod(O1+'.bvec',0o777)
    fbvec2 = open(O1+'.txt', 'w+')  # open file bvec in write mode
    os.chmod(O1+'.bvec',0o777)
    fbval = open(O1+'.bval', 'w+')  # open file bvec in write mode
    os.chmod(O1+'.bval',0o777)
    for file in os.listdir(Input):
        f = os.path.join(Input,file)
        print(f)
        fd = open(os.path.join(f,'acquisition_method.txt'),'r')
        line=fd.readline()
        print(line)
        if(line=="DtiEpi"):
            for filename in os.listdir(f):
                if 'b0.nii' in filename:
                    continue
                elif '.nii' in filename:
                    list_Dti.append(os.path.join(f,filename))
                elif 'DwEffBval.txt' in filename:
                    current=open(os.path.join(f,filename),'r')
                    line=current.readline()
                    while line:
                        fbval.write(line)
                        line=current.readline()
                    #fbval.write('\n')
                    current.close()
                elif 'DwDir.txt' in filename:
                    current=open(os.path.join(f,filename),'r')
                    line=current.readline()
                    while line:
                        fbvec.write(line)
                        line=current.readline()
                    #fbval.write('\n')
                    current.close()
                elif 'DwGradVec.txt' in filename:
                    current=open(os.path.join(f,filename),'r')
                    line=current.readline()
                    while line:
                        fbvec2.write(line)
                        line=current.readline()
                    #fbval.write('\n')
                    current.close()
        fd.close()
 
    fbvec.close()
    fbvec2.close()
    fbval.close()
    fbval=open(O1+'.bval','r+')
    indx = open(OutDti+'/index.txt','w+')
    line=fbval.readline()
    while line:
        indx.write('1 ')
        line=fbval.readline()
    fbval.close()
    indx.close()
    
    print(list_Dti)
    
    if not os.path.isfile(O1+'.nii.gz'):
        u.ni_creator(12,O1+'.nii.gz')
    if os.stat(O1+'.nii.gz').st_size <= 111:
        img1 =u.merge(list_Dti,O1+'.nii.gz',error_file)
        if img1 == -1:
            error_file.write("merge liste vide "+O1+"\n")
        if img1 == 1:
            error_file.write("merge file "+O1+'.nii.gz'+" inexistant \n")
            
    """
    if not os.path.isfile(OutDti) or os.stat(OutDti).st_size <= 111:
        imj1 = u.json_merge(list_Dti,OutDti,error_file)
        if imj1 == -1:
            error_file.write("merge liste vide "+OutDti+"\n")
        if imj1 == 1:
            error_file.write("merge file "+OutDti+" inexistant \n")

    if not os.path.isfile(O2+'.nii.gz'):
        u.ni_creator(12,O2+'.nii.gz')
    if os.stat(O2+'.nii.gz').st_size <= 111:
       img2 = u.merge(list_multiT2,O2+'.nii.gz',error_file)
       if img2 == -1:
            error_file.write("merge liste vide"+O2+"\n")
       if img2 == 1:
            error_file.write("merge file "+O2+'.nii.gz'+" inexistant \n")
    if not os.path.isfile(OutT2) or os.stat(OutT2).st_size <= 111:
        imj2 = u.json_merge(list_multiT2, OutT2,error_file)
        if imj2 == -1:
            error_file.write("merge liste vide"+OutT2+"\n")
        if imj2 == 1:
            error_file.write("merge file "+OutT2+" inexistant \n")
    """



#======================================================================
#  Main
#======================================================================

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-n','--name',dest = 'name',
                      help='path of data')
    parser.add_option('-r','--replace',dest = 'replace',
                      help='replace data True/False')
    parser.add_option('-e','--elikopy',dest = 'elikopy',
                      help='pre-processing data with elikopy: True/False')
    parser.add_option('-s','--Stop',dest = 'Stop',
                      help='nomber of file convert')
    
    
    (options,args) = parser.parse_args()
    
    Stop = vars(options)['Stop']
    if Stop!=None and int(Stop)>1:
        Stop = int(Stop)
    else:
        Stop = 100

    replace = vars(options)['replace']
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No']:
        replace = False
    else:
        replace = True
        
    Base = vars(options)['name']
    if Base == None or (not os.path.isdir(Base)):
        Base = '/Users/eliotthenaut/Desktop/Mémoire'
    
    error_file = open(Base+'/error_file.txt','w+')
    os.chmod(Base+'/error_file.txt',0o777)
    
    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'Convert')
    u.create_folder(BaseOUT,replace)
    BaseMerge = os.path.join(Base,'Merge')
    u.create_folder(BaseOUT,replace)
    u.create_folder(BaseMerge,replace)
    Dti = os.path.join(BaseMerge,'Dti')
    u.create_folder(Dti,replace)
    T2 = os.path.join(BaseMerge,'T2')
    u.create_folder(T2,replace)
    
    d=0
    for file in os.listdir(BaseIN):
        if(os.path.isdir(os.path.join(BaseIN,file)) and file != 'cleaning_data'):
            Input  = os.path.join(BaseIN,file)
            if os.path.isfile(os.path.join(Input,'subject')):
                name = u.get_name(os.path.join(Input,'subject'),file,'SUBJECT_study_name')
            else:
                error_file.write("get_name "+os.path.join(Input,'subject')+"\n")
                continue
            Output = BaseOUT
            convert(Input,Output,name,error_file)
            d=d+1
            if d>=Stop:
                break

            
    for name in os.listdir(BaseOUT):
        Output = os.path.join(BaseOUT,name)

        f1 = os.path.join(Dti,name)
        f2 = os.path.join(T2,name)
        u.create_folder(f1,replace)
        u.create_folder(f2,replace)
        link(Output,f1,f2,name,error_file)
        
        # A changer
        acqp = os.path.join(f1,'acqparams.txt')
        a = open(acqp,'w')
        os.chmod(acqp,0o777)
        a.write("0 -1 0 0.095")
        a.close()
            
            
            
    error_file.close()
    if os.path.getsize(Base+'/error_file.txt') == 0:
        error_file = open(Base+'/error_file.txt','w+')
        error_file.write('During the execution, no errors were encountered. good job.')
        error_file.close()
    
    preproc = vars(options)['elikopy']
    if preproc in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux','Non','non','no','No']:
        print("no pré-processing")
        print("temps de convertion",tConv)
        print("temps de concatenation",tMerg)
    else:

        Ep.Process(Dti)
        t4 = time.time()
        
        tProc = t4-t3
                
        print("temps de convertion",tConv)
        print("temps de concatenation",tMerg)
        print("temps de pré-processing",tProc)
    
    
    
    
    
    
    