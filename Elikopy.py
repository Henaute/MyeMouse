#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:54:10 2021

@author: eliotthenaut
"""
import utils as u
import elikopy
import elikopy.utils

def Process(f_path):
    
    study = elikopy.core.Elikopy(f_path)
    
    study.patient_list()
    
    study.preproc()
    
    study.white_mask()
    
    study.dti()
    
    study.noddi()
    
    study.diamond()
    
    study.fingerprinting()
    
    study.export(raw=False, preprocessing=True, dti=False,noddi=False, 
                 diamond=False, mf=False, wm_mask=False, report=True)
