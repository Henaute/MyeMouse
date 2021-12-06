#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:57:42 2021

@author: Eliott Henaut & Nicolas Bioul
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
from termcolor import colored
from optparse import OptionParser



     
def create_folder(path,replace=True):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path): 
            os.mkdir(path)
  
def convert(Input,Out):
    g = '\u0022'       

    for file_name in os.listdir(Input):    
        if file_name.isnumeric():  # lit que les nom de chiffre
            In = os.path.join(Input,str(file_name))
            Option = get_name(In+'/visu_pars',str(file_name),'VisuAcquisitionProtocol')# va chercher le nom
            path_out = os.path.join(Out, Option)

            if(os.path.isfile(In)==False and os.path.isfile(path_out+'.nii.gz')==False):  # convert only directory not file 
                if file_name==str(1):
                    B=0
                    for fn in os.listdir(Out):
                        #print(fn)
                        if '01_Localizer' in fn:
                            B=1
                            
                    if(B==0):
                        os.system('/Applications/MRIcron.app/Contents/Resources/dcm2niix -f ' + g + Option + g + ' -p n -z y -o ' + g + Out + g +' '+ g + In + g)
                    
                        if os.path.isfile(path_out+'.nii.gz')==False:
                                print(colored(str(file_name),'magenta'),colored('was not converted','red'))
                                #time.sleep(2)
                        else:
                            print(colored(str(file_name),'cyan'),'was converted to',colored(Option,'green'))
                            #time.sleep(1)
                    else:
                        print(colored(str(file_name),'cyan'),'already converted to',colored(Option,'green'))
                        #time.sleep(1)
                elif file_name==str(5):
                    B=0
                    for fn in os.listdir(Out):
                        #print(fn)
                        if '05_T2map_MSME' in fn:
                            B=1
                            
                    if(B==0):
                        os.system('/Applications/MRIcron.app/Contents/Resources/dcm2niix -f ' + g + Option + g + ' -p n -z y -o ' + g + Out + g +' '+ g + In + g)
                    
                        if os.path.isfile(path_out+'.nii.gz')==False:
                                print(colored(str(file_name),'magenta'),colored('was not converted','red'))
                                #time.sleep(2)
                        else:
                            print(colored(str(file_name),'cyan'),'was converted to',colored(Option,'green'))
                            #time.sleep(1)
                    else:
                        print(colored(str(file_name),'cyan'),'already converted to',colored(Option,'green'))
                        #time.sleep(1)
                    
                else:

                    os.system('/Applications/MRIcron.app/Contents/Resources/dcm2niix -f ' + g + Option + g + ' -p n -z y -o ' + g + Out + g +' '+ g + In + g)
                    
                    if os.path.isfile(path_out+'.nii.gz')==False:
                            print(colored(str(file_name),'magenta'),colored('was not converted','red'))
                            #time.sleep(2)
                    else:
                        print(colored(str(file_name),'cyan'),'was converted to',colored(Option,'green'))
                        #time.sleep(1)
                file=open(In+'/method','r')
                a=[]
                b=[]
                bval = path_out+'.bval'
                bvec = path_out+'.bvec'
                fbvec = open(bvec, 'w+')  # open file bvec in write mode
                fbval = open(bval, 'w+')  # open file bvec in write mode
                line=file.readline()
                while line:
                    if 'PVM_DwEffBval' in line:
                        k=file.readline()
                        while ('#' not in k and '$' not in k):
                            b.append(list(re.findall("[+-]?\d+\.\d+", k))) 

                            k=file.readline()
                    if '##$PVM_DwDir=' in line:

                        j=file.readline()
                        while ('#' not in j and '$' not in j):
                            a.append(list(re.findall("[+-]?\d+\.\d+", j)))

                            j=file.readline()
                    line=file.readline()
                    
                print("num a , num b ",sum(len(x) for x in a)/3,sum(len(x) for x in b))
                if(sum(len(x) for x in a)/3 == 32.0 and sum(len(x) for x in b) == 35):
                    print("Incoherences between bval (35) and bvec (32), inserting bvec for calibration")
                    fbvec.write('1 0 0\n')
                    fbvec.write('0 1 0\n')
                    fbvec.write('0 0 1\n')
                elif(sum(len(x) for x in a)/3 == 1.0 and sum(len(x) for x in b) == 4):
                    print("Incoherences between bval (4) and bvec (1), inserting bvec for calibration")
                    fbvec.write('1 0 0\n')
                    fbvec.write('0 1 0\n')
                    fbvec.write('0 0 1\n')
                if len(a)==0:
                    os.remove(bval)
                else:
                    inc = 0
                    for i in range(len(a)):
                        for j in range(len(a[i])):
                            #print(a[i][j])
                            if(inc == 3):
                                inc = 0
                                fbvec.write('\n')
                                
                            fbvec.write(a[i][j])
                            inc+=1
                            fbvec.write(' ') 
                if len(b)==0:
                    os.remove(bvec)
                else:
                    for i in range(len(b)):
                        for j in range(len(b[i])):
                            #print(b[i][j])
                            fbval.write(b[i][j])
                            fbval.write('\n')                             
                file.close()
                fbvec.close()
                fbval.close()
            else:
                print(colored(str(file_name),'cyan'),'already converted to',colored(Option,'green'))

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

def link(Input,OutDti,OutT2):
    reverse = os.path.join(OutDti,'reverse_encoding')
    create_folder(reverse)
    start='/Groupe'
    end='/'
    name='Groupe'+(Input.split(start))[1].split(end)[0]
    O1 = os.path.join(OutDti,name)
    O2 = os.path.join(OutT2,name)
    fbvec = open(O1+'.bvec', 'w+')  # open file bvec in write mode
    fbval = open(O1+'.bval', 'w+')  # open file bvec in write mode
    indx = open(OutDti+'/index.txt','w+')
    ni_creator(12,O1+'.nii.gz')
    ni_creator(12,O2+'.nii.gz')
    list_Dti=[]
    list_multiT2=[]
    for i in range(1,20):
        #print(i)
        if i<10:
            c ='0'+str(i)
        else:
            c = str(i)
        for filename in os.listdir(Input):
            file = os.path.join(Input,filename)
            if c+'_DtiEpi' in filename :
                if 'reverse' in filename:
                    rev = os.path.join(reverse,name)
                    if 'nii.gz' in filename:
                        shutil.copyfile(file,rev+'.nii.gz')
                    elif 'nii' in filename:
                        shutil.copyfile(file,rev+'.nii')
                    elif 'bval' in filename:
                        shutil.copyfile(file,rev+'.bval')
                    elif 'bvec' in filename:
                        shutil.copyfile(file,rev+'.bvec')
                    elif 'json' in filename:
                        shutil.copyfile(file,rev+'.json')
                elif 'nii' in filename:
                    list_Dti.append(file)                
                elif 'bval' in filename:
                    current=open(file,'r')
                    line=current.readline()
                    while line:
                        fbval.write(line)
                        line=current.readline()
                    #fbval.write('\n')
                    current.close()
                elif 'bvec' in filename:
                    current=open(file,'r')
                    line=current.readline()
                    while line:
                        fbvec.write(line)
                        line=current.readline()
                    fbvec.write('\n')
                    current.close()
                else: 
                    pass
            elif 'MSME_e1' in filename and i==1:
                if 'nii' in filename:
                    list_multiT2.append(file)
            else:
                pass
    fbvec.close()
    fbval.close()
    fbval=open(O1+'.bval','r+')
    line=fbval.readline()
    while line:
        indx.write('1 ')
        line=fbval.readline()
    fbval.close()
    indx.close()
    merge(list_Dti,O1+'.nii.gz')
    json_merge(list_Dti,OutDti)
    merge(list_multiT2,O2+'.nii.gz')
    json_merge(list_multiT2, OutT2)
    
def merge(inlist, out):
    if 'T2map_MSME' in inlist[0]:
        img = nib.concat_images(inlist,axis=0)
        nib.save(img, out)
    else:
        img = nib.concat_images(inlist,axis=3)
        nib.save(img, out)
    return img
    
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


#======================================================================
#  Main
#======================================================================

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-n','--name',dest = 'name',
                      help='path of data')
    (options,args) = parser.parse_args()
    #makeFile(options)

    Base = vars(options)['name']
    
    if Base == None or (not os.path.isdir(Base)):
        Base = os.getcwd()

    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'ConvertEliott')
    BaseMerge = os.path.join(Base,'Merge')
    create_folder(BaseOUT)
    create_folder(BaseMerge)
    Dti = os.path.join(BaseMerge,'Dti')
    create_folder(Dti)
    T2 = os.path.join(BaseMerge,'T2')
    create_folder(T2)
    
    d=1
    for file in os.listdir(BaseIN):
        if(os.path.isdir(os.path.join(BaseIN,file)) and file != 'cleaning_data'):
            Input  = os.path.join(BaseIN,file)
            name = get_name(os.path.join(Input,'subject'),file,'SUBJECT_study_name')
            Output =os.path.join(BaseOUT, name)
            
            create_folder(Output)
            convert(Input,Output)
            
            f1 = os.path.join(Dti,'data_'+str(d))
            f2 = os.path.join(T2,'data_'+str(d))
            create_folder(f1)
            create_folder(f2)
    
            link(Output,f1,f2)
            acqp = os.path.join(f1,'acqparams.txt')
            a = open(acqp,'w')
            a.write("0 -1 0 0.095")
            a.close()
            
            d=d+1
            
    f_path = Dti
    study = elikopy.core.Elikopy(f_path)
    
    patient_list=None
    
    study.patient_list()
    
    study.preproc()
    
    """
    study.preproc(eddy=True,topup=True,denoising=True,reslice=False,gibbs=False,biasfield=False,patient_list_m=patient_list,starting_state=None)
    
    study.white_mask()
    
    study.dti(patient_list_m=patient_list)
    
    study.noddi()
    """
    
    study.export(raw=True, preprocessing=True, dti=False,noddi=False, 
                 diamond=False, mf=False, wm_mask=False, report=True)
    
