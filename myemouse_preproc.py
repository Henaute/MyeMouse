#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:55:09 2022

@author: nicolasbioul
"""

import os
import math
import time
import shutil
import elikopy
import numpy as np
import datetime as dt
import nibabel as nib

from dipy.denoise.localpca import mppca
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.viz import regtools
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


from skimage.morphology import binary_dilation,binary_erosion
from threading import Thread
from elikopy.utils import makedir

denoised = None

def preprocessing(folder_path, patient_path, Denoising=True, Motion_corr=True, Mask_off=True,logs=None):
    #variable to skip prossess
    global denoised

    log_prefix = "[Myemouse Preproc]"
    #logs.write('[Myemouse Preproc] started. Launching preprocessing on data  '+str(dt.datetime.now())+'\n')

    fdwi = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + '_raw_dmri.nii.gz'
    fbval = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bval"
    fbvec = folder_path + '/subjects/' + patient_path + '/dMRI/raw/' + patient_path + "_raw_dmri.bvec"

    fcorr_dwi = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + '_dmri_preproc.nii.gz'
    fcorr_bval = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bval"
    fcorr_bvec = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/' + patient_path + "_dmri_preproc.bvec"
    
    #logs.write('[Myemouse Preproc] Motion correction sequence launched '+str(dt.datetime.now())+'\n') 
    #if starting_state is None or starting_state=="motionCorr":
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=65)
    
    
    
    import json
    with open(os.path.join(folder_path + '/subjects/',"subj_type.json")) as json_file:
        subj_type = json.load(json_file)

    subj_type[patient_path]
    
    ############################
    ### MPPCA Denoising step ###
    ############################
    if Denoising:
        
        print("Start of denoising step", patient_path)
        denoisingMPPCA_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/denoisingMPPCA/'
        makedir(denoisingMPPCA_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt",log_prefix)

        #logs.write('[Myemouse Preproc] Denoising sequence launched '+str(dt.datetime.now())+'\n')
        shell_index = []
        with open(os.path.join(folder_path, "data_" + str(subj_type[patient_path]),"shell_index.txt"), "r") as f:
            for line in f:
                shell_index.append(int(line.strip()))

        denoised = data.copy()

        print(shell_index, data.shape)
        
        threads = []
        for i in range(len(shell_index)-1):
            print("Start of mppca for shell", i, " (index:", shell_index[i],",", shell_index[i+1],")")
            #logs.write('Marcenko-Pastur PCA algorithm launched for shell '+ str(i)+ '(index:'+ str(shell_index[i])+","+ str(shell_index[i+1])+')'+dt.datetime.now()+'\n')
            a = shell_index[i]
            b = shell_index[i+1]
            chunk = data[:,:,:,a:b].copy()
            threads.append(Thread(target=threaded_mppca, args=(a,b,chunk)))
            threads[-1].start()

        print("All threads have been launched")
        #logs.write('All threads have been launched at'+str(dt.datetime.now())+'\n')
        for i in range(len(threads)):
            threads[i].join()
        #logs.write('All threads finished at '+str(dt.datetime.now())+'\n')

        save_nifti(denoisingMPPCA_path + '/' + patient_path + '_mppca.nii.gz', denoised.astype(np.float32), affine)
        data = denoised
 
    ##############################
    ### Motion correction step ###
    ##############################
    
    
    if Motion_corr:
    
        print("Motion correction step for subject ", patient_path)
        motionCorr_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/motionCorrection/'
        makedir(motionCorr_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)

        #logs.write('Motion correction step for subject '+patient_path+dt.datetime.now()+'\n')
        reg_affines_precorrection = []
        static_precorrection = data[..., 0]
        static_grid2world_precorrection = affine
        moving_grid2world_precorrection = affine
        for i in range(data.shape[-1]):
            if gtab.b0s_mask[i]:
                print("Motion correction: Premoving b0 number ", i)
                #logs.write('Motion correction step for subject '+ patient_path+dt.datetime.now()+'\n')
                moving = data[...,i]
                moved, trans_affine = affine_reg(static_precorrection, static_grid2world_precorrection,
                                                 moving, moving_grid2world_precorrection)
                data[..., i] = moved
            else:
                moving = data[..., i]
                data[..., i] = trans_affine.transform(moving)
                reg_affines_precorrection.append(trans_affine.affine)

        gtab_precorrection = reorient_bvecs(gtab, reg_affines_precorrection)
        data_b0s = data[..., gtab.b0s_mask]

        data_avg_b0 = data_b0s.mean(axis=-1)
        save_nifti(motionCorr_path + patient_path + '_avgB0.nii.gz', data_avg_b0, affine)
        save_nifti(motionCorr_path + patient_path + '_B0S.nii.gz', data_b0s, affine)
        bvec = gtab.bvecs
        zerobvec = np.where((bvec[:, 0] == 0) & (bvec[:, 1] == 0) & (bvec[:, 2] == 0))
        bvec[zerobvec] = [1, 0, 0]
        save_nifti(motionCorr_path + patient_path + '_motionCorrected.nii.gz', data, affine)
        np.savetxt(motionCorr_path + patient_path + '_motionCorrected.bval', bvals)
        np.savetxt(motionCorr_path + patient_path + '_motionCorrected.bvec', bvec)

        gtab = gtab_precorrection

    elif os.path.exists(motionCorr_path + patient_path + '_motionCorrected.nii.gz'):
        data, affine = load_nifti(motionCorr_path + patient_path + '_motionCorrected.nii.gz')
        fbval = motionCorr_path + patient_path + '_motionCorrected.bval'
        fbvec = motionCorr_path + patient_path + '_motionCorrected.bvec'
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=65)
    
    #############################
    ### Brain extraction step ###
    #############################
    if Mask_off:
        print('Brain extraction step')
        brainExtraction_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/brainExtraction/'
        mask_path=folder_path + '/subjects/' + patient_path + 'masks/'

    
        makedir(brainExtraction_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt", log_prefix)
        makedir(mask_path, folder_path + '/subjects/' + patient_path + "/dMRI/preproc/preproc_logs.txt",log_prefix)
        
        # created a brain for designing a mask (sum of all shells)
        b0final=np.zeros(data.shape[:-1])
        for i in range(data.shape[-1]):
                b0final+=data[:, :, :, i]
    
        save_nifti(brainExtraction_path + patient_path + 'Brain_extraction_ref.nii.gz',b0final, affine)
        
        # function to find a mask
        final_mask=mask_Wizard(b0final,4,7,work='2D')
        
        # Saving
        out = nib.Nifti1Image(final_mask, affine)
        out.to_filename(mask_path + patient_path + 'brain_mask.nii.gz')
        
        data[final_mask==0] = 0
        
        save_nifti(brainExtraction_path + patient_path + 'Extracted_brain.nii.gz',data, affine)
        
    elif os.path.exists(brainExtraction_path + patient_path + 'Extracted_brain.nii.gz'):
        data, affine = load_nifti(brainExtraction_path + patient_path + 'Extracted_brain.nii.gz')
        

        
    ################################
    ### Final preprocessing step ###
    ################################
    
    final_path = folder_path + '/subjects/' + patient_path + '/dMRI/preproc/'
    save_nifti(final_path+'/'+ patient_path +'dmri_preproc.nii.gz',data,affine)
    np.savetxt(final_path+'/'+ patient_path +'dmri_preproc.bval', bvals)
    np.savetxt(final_path+'/'+ patient_path +'dmri_preproc.bvec', bvec)
    
    

"""
==========================================
Denoising of DWI data using MPPCA
@author: DESSIN Quentin
==========================================
"""

def threaded_mppca(a, b, chunk):
    global denoised
    pr = math.ceil((np.shape(chunk)[3] ** (1 / 3) - 1) / 2)
    denoised_chunk, sigma = mppca(chunk, patch_radius=pr, return_sigma=True)
    denoised[:,:,:,a:b] = denoised_chunk


"""
==========================================
Motion correction of DWI data
@author: DELINTE Nicolas 
==========================================
"""

def affine_reg(static, static_grid2world,
               moving, moving_grid2world):

    c_of_mass = transform_centers_of_mass(static,
                                          static_grid2world,
                                          moving,
                                          moving_grid2world)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [500, 100, 10]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    transformed = rigid.transform(moving)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    transformed = affine.transform(moving)

    return transformed, affine


"""
=======================================================
Brain extraction of DWI data
@author: HENAUT Eliott BIOUL Nicolas
=======================================================
"""
def mask_Wizard(data,r_fill,r_shape,scal=1,geo_shape='ball',work='2D'):
    """

    Parameters
    ----------
    data : TYPE : ArrayList
        The brain data that we want to find a mask.
    r_fill : TYPE : Interger
        The rayon of the circle,cylinder,or ball that we use for growing the surface.
    r_shape : TYPE : Interger
        The rayon of the circle,cylinder,or ball that we use for the opening/closing.
    scal : TYPE : Interger, optional
        The param λ in the formul y=µ+λ*σ. y was the threshold if we take the voxel or not.
        The default is 1.
    geo_shape : TYPE : String, optional
        The shape of the convolution in the opening/closing. ('ball','cylinder')
        The default is 'ball'.
    work : TYPE : String, optional
        The dimension. ('2D','3D')
        The default is '2D'.
        
    Returns
    -------
    mask : TYPE : Arraylist
        The matrice of the mask, 1 if we take it, O otherwise.

    """
    if work=='3D':
    
        (x,y,z)=data.shape
        seed = binsearch(data,x,y,z)
        brainish = fill(seed, data, 1,r_fill,scal)
        geo_shape=shape_matrix(r_shape,geo_shape)
        closing=binary_erosion(binary_dilation(brainish,selem=geo_shape))
        opening=binary_dilation(binary_erosion(closing,selem=geo_shape))
        mask=np.zeros(opening.shape)
        mask[opening]=1
    else:   
        (x,y,z)=data.shape
        b0final=data
        for i in range(z):
            if np.std(b0final[:,:,i]/np.mean(b0final[:,:,i]))>0.5:
                seed = binsearch(b0final[:,:,i],x,y,work='2D')
                brainish=fill(seed,b0final[:,:,i],1,r_fill)
                mat=shape_matrix(r_shape)
                closing = binary_erosion(binary_dilation(brainish,selem=mat),selem=mat)
                opening = binary_dilation(binary_erosion(closing,selem=mat),selem=mat)
                final= binary_dilation(opening,selem=shape_matrix(2,work='2D'))
                inter=np.zeros(final.shape)
                inter[final]=1
                b0final[:,:,i]=inter
            else:
                b0final[:,:,i]=np.zeros(b0final[:,:,i].shape)
        mask=b0final
        
    return mask
    

def fill(position, data, new_val,rad,scal=1,work='2D'):
    data_new = np.zeros(data.shape)
    init_val = int(np.mean(data)+scal*np.sqrt(np.var(data))) # can be touched for precision
    voxelList = set()
    voxelList.add(position)

    while len(voxelList) > 0:
        voxel = voxelList.pop()
        voxelList = getVoxels(voxel, data, init_val, voxelList,rad, data_new, new_val,work)
        data_new[voxel] = int(new_val)
    return data_new


def getVoxels(voxel, data, init_val, voxelList, rad, data_new,new_val,work='2D'):
    if work=='3D':
        (x, y, z) = voxel
        adjacentVoxelList = set()
        for i in range(rad):
            h=int(np.sqrt(rad**2-i**2))
            for j in range(-h,h+1):
                for k in range(-1,2):
                    adjacentVoxelList.add((x+i,y+j,z+k))
                    adjacentVoxelList.add((x-i,y+j,z+k))
        for adja in adjacentVoxelList:
            if isInbound(adja, data,work):
                if data[adja] >= init_val and data_new[adja] != new_val:
                    voxelList.add(adja)
    else:
        pixel=voxel
        (x, y) = pixel
        adjacentPixelList = set()
        for i in range(rad):
            h=int(np.sqrt(rad**2-i**2))
            for j in range(-h,h+1):
                adjacentPixelList.add((x+i,y+j))
                adjacentPixelList.add((x-i,y+j))
        for adja in adjacentPixelList:
            if isInbound(adja, data,work):
                if data[adja] >= init_val and data_new[adja] != new_val:
                    voxelList.add(adja)
    return voxelList


def isInbound(voxel, data, work='2D'):
    if work=='3D':
        return voxel[0] < data.shape[0] and voxel[0] >= 0 and voxel[1] < data.shape[1] and voxel[1] >= 0 and voxel[2] < \
           data.shape[2] and voxel[2] >= 0
    else:
        return voxel[0] < data.shape[0] and voxel[0] >= 0 and voxel[1] < data.shape[1] and voxel[1] >= 0


def binsearch(img,x2,y2,z=0,x1=0,y1=0,work='2D'):
    if work=='3D':
        if x2<=x1+1 or y2<=y1+1 :
            return (x1,y1,z//2)
        cand = [[[x1,(x1+x2)//2],[y1,(y1+y2)//2]],[[(x1+x2)//2,x2],[y1,(y1+y2)//2]],[[x1,(x1+x2)//2],[(y1+y2)//2,y2]],[[(x1+x2)//2,x2],[(y1+y2)//2,y2]]]
        
        Im1 = img[cand[0][0][0]:cand[0][0][1],cand[0][1][0]:cand[0][1][1],z//2]
        Im2 = img[cand[1][0][0]:cand[1][0][1],cand[1][1][0]:cand[1][1][1],z//2]
        Im3 = img[cand[2][0][0]:cand[2][0][1],cand[2][1][0]:cand[2][1][1],z//2]
        Im4 = img[cand[3][0][0]:cand[3][0][1],cand[3][1][0]:cand[3][1][1],z//2]
        
        idx = search([Im1,Im2,Im3,Im4],[0,1,2,3])
    
    else:
        if x2<=x1+1 or y2<=y1+1 :
            return (x1,y1)
        cand = [[[x1,(x1+x2)//2],[y1,(y2+y1)//2]],[[(x1+x2)//2,x2],[y1,(y2+y1)//2]],[[x1,(x1+x2)//2],[(y1+y2)//2,y2]],[[(x1+x2)//2,x2],[(y1+y2)//2,y2]]]
        Im1 = img[cand[0][0][0]:cand[0][0][1],cand[0][1][0]:cand[0][1][1]]
        Im2 = img[cand[1][0][0]:cand[1][0][1],cand[1][1][0]:cand[1][1][1]]
        Im3 = img[cand[2][0][0]:cand[2][0][1],cand[2][1][0]:cand[2][1][1]]
        Im4 = img[cand[3][0][0]:cand[3][0][1],cand[3][1][0]:cand[3][1][1]]
        
        idx = search([Im1,Im2,Im3,Im4],[0,1,2,3])
    return binsearch(img,cand[idx][0][1], cand[idx][1][1],z=z, x1=cand[idx][0][0], y1=cand[idx][1][0],work=work)
    
     
def search(l,idx):
    if len(idx)==1: return idx[0]
    newidx=[]
    for i in range(0,len(idx),2):
        if np.mean(l[idx[i]])<np.mean(l[idx[i+1]]):
            newidx.append(idx[i+1])
        else:
            newidx.append(idx[i])
    return search(l,newidx)
    
def shape_matrix(radius,shape='ball',height=5,work='2D'):
    if work=='3D':
        mat=0
        if shape=='cylinder':
            mat=np.zeros((2*radius+1,2*radius+1,height))
            for i in range(radius):
                h=int(np.sqrt(radius**2-i**2))
                for j in range(-h,h+1):
                    for k in range(3):
                        mat[radius+i,radius+j,k]=1
                        mat[radius-i,radius+j,k]=1
                    
        else:
            mat=np.zeros((2*radius+1,2*radius+1,2*radius+1))
            for i in range(radius):
                h=int(np.sqrt(radius**2-i**2))
                for j in range(-h,h+1):
                    h1=int(np.sqrt(radius**2-i**2-j**2))
                    for k in range(-h1,h1+1):
                        mat[radius+i,radius+j,radius+k]=1
                        mat[radius-i,radius+j,radius+k]=1
    else:
        mat=0
        mat=np.zeros((2*radius+1,2*radius+1))
        for i in range(radius):
            h=int(np.sqrt(radius**2-i**2))
            for j in range(-h,h+1):
                mat[radius+i,radius+j]=1
                mat[radius-i,radius+j]=1                        
    return mat
    