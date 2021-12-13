#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:57:42 2021

@author: Eliott Henaut & Nicolas Bioul
"""
import utils as u
import os
import re
import shutil
import time 
from termcolor import colored
from optparse import OptionParser

  

def convert(Input,Out,error_file):
    g = '\u0022'       

    for file_name in os.listdir(Input):    
        if file_name.isnumeric():  # lit que les nom de chiffre
            In = os.path.join(Input,str(file_name))
            # va chercher le nom
            if os.path.isfile(In+'/visu_pars'):
                Option = u.get_name(In+'/visu_pars',str(file_name),'VisuAcquisitionProtocol')
            elif os.path.isfile(In+'/acqp'):
                Option = u.get_name(In+'/acqp',str(file_name),'ACQ_protocol_name')
            else:
                error_file.write("get_name "+In+'/visu_pars')
                continue
            
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
                        
                        for ln in os.listdir(Out):

                            if '01_Localizer' in ln:
                                
                                B=1
                                
                        if (B==0):
                            print(colored(str(file_name),'magenta'),colored('was not converted','red'))
                            error_file.write("convert "+path_out+" was not converted \n")
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
                        
                        for fn in os.listdir(Out):
                            #print(fn)
                            if '05_T2map_MSME' in fn:
                                B=1
                        if (B==0):
                            print(colored(str(file_name),'magenta'),colored('was not converted','red'))
                            error_file.write("convert "+path_out+" was not converted \n")
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
                        error_file.write("convert "+path_out+" was not converted \n")
                        #time.sleep(2)
                    else:
                        print(colored(str(file_name),'cyan'),'was converted to',colored(Option,'green'))
                        #time.sleep(1)
                        os.chmod(path_out+'.nii.gz',0o777)
                file=open(In+'/method','r')
                a=[]
                b=[]
                bval = path_out+'.bval'
                bvec = path_out+'.bvec'
                fbvec = open(bvec, 'w+')  # open file bvec in write mode
                os.chmod(bvec,0o777)
                fbval = open(bval, 'w+')  # open file bvec in write mode
                os.chmod(bval,0o777)
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

def link(Input,OutDti,OutT2,error_file):
    reverse = os.path.join(OutDti,'reverse_encoding')
    u.create_folder(reverse)
    start='/Groupe'
    end='/'
    name='Groupe'+(Input.split(start))[1].split(end)[0]
    O1 = os.path.join(OutDti,name)
    O2 = os.path.join(OutT2,name)
    fbvec = open(O1+'.bvec', 'w+')  # open file bvec in write mode
    os.chmod(O1+'.bvec',0o777)
    fbval = open(O1+'.bval', 'w+')  # open file bvec in write mode
    os.chmod(O1+'.bval',0o777)
    indx = open(OutDti+'/index.txt','w+')
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
    if not os.path.isfile(O1+'.nii.gz'):
        u.ni_creator(12,O1+'.nii.gz')
    if os.stat(O1+'.nii.gz').st_size == 0:
        img1 =u.merge(list_Dti,O1+'.nii.gz')
        if img1 == -1:
            error_file.write("merge liste vide "+O1+"\n")
        if img1 == 1:
            error_file.write("merge file "+O1+'.nii.gz'+" inexistan \nt")
    if not os.path.isfile(OutDti) or os.stat(OutDti).st_size == 0:
        imj1 = u.json_merge(list_Dti,OutDti)
        if imj1 == -1:
            error_file.write("merge liste vide "+OutDti+"\n")
        if imj1 == 1:
            error_file.write("merge file "+OutDti+" inexistant \n")
    if not os.path.isfile(O2+'.nii.gz'):
        u.ni_creator(12,O2+'.nii.gz')
    if os.stat(O2+'.nii.gz').st_size == 0:
       img2 = u.merge(list_multiT2,O2+'.nii.gz')
       if img2 == -1:
            error_file.write("merge liste vide"+O2+"\n")
       if img2 == 1:
            error_file.write("merge file "+O2+'.nii.gz'+" inexistant \n")
    if not os.path.isfile(OutT2) or os.stat(OutT2).st_size == 0:
        imj2 = u.json_merge(list_multiT2, OutT2)
        if imj2 == -1:
            error_file.write("merge liste vide"+OutT2+"\n")
        if imj2 == 1:
            error_file.write("merge file "+OutT2+" inexistant \n")
    



#======================================================================
#  Main
#======================================================================

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-n','--name',dest = 'name',
                      help='path of data')
    parser.add_option('-r','--replace',dest = 'replace',
                      help='replace data True/False')
    (options,args) = parser.parse_args()
    #makeFile(options)
    replace = vars(options)['replace']
    if replace in ['False','false','F','f','Flase','flase','Fasle','fasle','Faux','faux']:
        replace = False
    else:
        replace = True
        
    Base = vars(options)['name']
    if Base == None or (not os.path.isdir(Base)):
        Base = os.getcwd()
    
    error_file = open(Base+'/error_file.txt','w')
    os.chmod(Base+'/error_file.txt',0o777)
    
    BaseIN = os.path.join(Base,'raw')
    BaseOUT = os.path.join(Base,'Convert')
    BaseMerge = os.path.join(Base,'Merge')
    u.create_folder(BaseOUT,replace)
    u.create_folder(BaseMerge,replace)
    Dti = os.path.join(BaseMerge,'Dti')
    u.create_folder(Dti,replace)
    T2 = os.path.join(BaseMerge,'T2')
    u.create_folder(T2,replace)
    
    tConv = 0
    tMerg = 0
    d=1
    for file in os.listdir(BaseIN):
        if(os.path.isdir(os.path.join(BaseIN,file)) and file != 'cleaning_data'):
            Input  = os.path.join(BaseIN,file)
            if os.path.isfile(os.path.join(Input,'subject')):
                name = u.get_name(os.path.join(Input,'subject'),file,'SUBJECT_study_name')
            else:
                error_file.write("get_name "+os.path.join(Input,'subject')+"\n")
                continue
            Output = os.path.join(BaseOUT, name)
            t1 = time.time()
            u.create_folder(Output,replace)
            convert(Input,Output,error_file)
            t2 = time.time()
            tConv += t2-t1
            f1 = os.path.join(Dti,'data_'+str(d))
            f2 = os.path.join(T2,'data_'+str(d))
            u.create_folder(f1,replace)
            u.create_folder(f2,replace)
            link(Output,f1,f2,error_file)
            t3 = time.time()
            tMerg += t3-t2 
            
            # A changer
            acqp = os.path.join(f1,'acqparams.txt')
            a = open(acqp,'w')
            os.chmod(acqp,0o777)
            a.write("0 -1 0 0.095")
            a.close()
            
            d=d+1
    error_file.close()
            
    print("temps de convertion",tConv)
    print("temps de concatenation",tMerg)
    
    
    
    