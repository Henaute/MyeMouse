"""
Created on Mon Jul 25 08:32:42 2022

@author: Eliott Henaut & Nicolas Bioul
"""
import sys
import cv2
import math
import numpy as np
from scipy import ndimage
import copy as cp
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PIL import ImageQt, Image
from dipy.io.image import load_nifti, save_nifti
import nibabel as nib

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

class Nifti():
    
    def __init__(self, data, aff):
        #super().__init__()
        self.data = data
        self.affine = aff

        self.work = len(self.data.shape)
        
        t = self.data.shape[:2]
        t += (3,)
        self.current = np.ones(t)  
        self.z1 = 0
        self.nvol = 0

        #self.Max = np.amax(self.data)
        
        if self.work ==3:
            self.Max = np.amax(self.data[:,:,0])
            self.z1 = self.data.shape[-1]
            self.current[:,:,0] = np.array(self.data[:,:,0]/self.Max *255, dtype = np.uint8)
            self.current[:,:,1] = np.array(self.data[:,:,0]/self.Max *255, dtype = np.uint8)
            self.current[:,:,2] = np.array(self.data[:,:,0]/self.Max *255, dtype = np.uint8)
            
        elif self.work == 4:
            self.Max = np.amax(self.data[:,:,0,0])
            self.z1 = self.data.shape[-2]
            self.nvol = self.data.shape[-1]
            self.current[:,:,0] = np.array(self.data[:,:,0,0]/self.Max *255, dtype = np.uint8)
            self.current[:,:,1] = np.array(self.data[:,:,0,0]/self.Max *255, dtype = np.uint8)
            self.current[:,:,2] = np.array(self.data[:,:,0,0]/self.Max *255, dtype = np.uint8)
            
    def update(self,v1,v2):
        
        v1 = int(v1)
        v2 = int(v2)
        
        if self.work ==3:
            self.Max = np.amax(self.data[:,:,v1])
            self.current[:,:,0] = np.array(self.data[:,:,v1]/self.Max *255, dtype = np.uint8)
            self.current[:,:,1] = np.array(self.data[:,:,v1]/self.Max *255, dtype = np.uint8)
            self.current[:,:,2] = np.array(self.data[:,:,v1]/self.Max *255, dtype = np.uint8)
            
        elif self.work == 4:
            self.Max = np.amax(self.data[:,:,v1,v2])
            self.current[:,:,0] = np.array(self.data[:,:,v1,v2]/self.Max *255, dtype = np.uint8)
            self.current[:,:,1] = np.array(self.data[:,:,v1,v2]/self.Max *255, dtype = np.uint8)
            self.current[:,:,2] = np.array(self.data[:,:,v1,v2]/self.Max *255, dtype = np.uint8)
        

class ComboBox(QWidget):

    def __init__(self, nom, parent):
        super().__init__()
        
        self.name = nom
        self.vbox = QVBoxLayout()
        self.cb = QComboBox()
        
        self.cb.addItems([" Nifti Tools", " TIFF Tools"])
        
        self.cb.currentIndexChanged.connect(parent.Combo)
        
        
        self.vbox.addWidget(self.cb)
        self.setLayout(self.vbox)
        
class CheckBox(QWidget):

    def __init__(self, nom, parent):
        super().__init__()
        self.name =nom
        self.vbox = QVBoxLayout()
        self.b = QCheckBox(self.name)
        self.b.setChecked(False)
        
        if "Display Nifti" in self.name :
            self.b.stateChanged.connect(parent.clickBox1)
            
        if "Display TIFF" in self.name :
            self.b.stateChanged.connect(parent.clickBox2)
            
        if "Flip" in self.name :
            self.b.stateChanged.connect(parent.clickFlip)
            
        self.vbox.addWidget(self.b)
        
        self.setLayout(self.vbox)
    
        
class Boutton(QWidget):
    
    def __init__(self,nom,parent):
        super().__init__()
        self.name =nom
        self.bou = QVBoxLayout()
        self.B = QPushButton(self.name)
        
        if "Clear Settings" in self.name :
            self.B.clicked.connect(parent.clickBoutton1)
            self.B.setShortcut("Maj+c")
            
        if "Save File" in self.name :
            self.B.clicked.connect(parent.Save)
            self.B.setShortcut("Ctrl+s")

        self.bou.addWidget(self.B)
        
        self.setLayout(self.bou)
        
class Volume(QWidget):
    
    def __init__(self, nom, x, y, parent):
        super().__init__()
        self.name =nom
        self.vbox = QVBoxLayout()
        self.spin = QDoubleSpinBox()
        self.spin.setRange(x,y)
        self.spin.setDecimals(0)
        #self.spin.setSuffix(" nVol")
        #self.spin.setDisplayIntegerBase(10)
        #self.spin.setValue(0)
        #self.spin.setTickInterval(1)
    
        self.spin.valueChanged.connect(parent.nVolume)
        #self.spin.valueChanged.connect(self.updateLabel)

    
        self.label = QLabel(nom.format(0), self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(120, 15)
        
        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.spin)
        
        self.setLayout(self.vbox)

class Curseur(QWidget):
    
    def __init__(self, nom, x, y, s, t, parent):
        super().__init__()
        

        self.name = nom
        self.box = QVBoxLayout()
        self.sld = QSlider(Qt.Orientation.Horizontal)
        self.sld.setRange(x, y)
        self.sld.setValue(s)
        self.sld.setTickInterval(t)
        
        if "zoom Nifti"in self.name:
            
            self.sld.valueChanged.connect(parent.updateZoomN)
            self.sld.valueChanged.connect(self.updateLabelZoom)

            
        elif "zoom TIFF"in self.name:
            
            self.sld.valueChanged.connect(parent.updateZoomT)
            self.sld.valueChanged.connect(self.updateLabelZoom)

        elif "x axis Nifti" in self.name:
            
            self.sld.valueChanged.connect(parent.updateNX)
            self.sld.valueChanged.connect(self.updateLabel)

            
        elif "y axis Nifti" in self.name:
            
            self.sld.valueChanged.connect(parent.updateNY)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "x axis TIFF" in self.name:
            
            self.sld.valueChanged.connect(parent.updateTX)
            self.sld.valueChanged.connect(self.updateLabel)

            
        elif "y axis TIFF" in self.name:
            
            self.sld.valueChanged.connect(parent.updateTY)
            self.sld.valueChanged.connect(self.updateLabel)
        
        
        elif "opacity"in self.name:
            
            self.sld.valueChanged.connect(parent.Opacity)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "x affine" in self.name:
            
            self.sld.valueChanged.connect(parent.affineX)
            self.sld.valueChanged.connect(self.updateLabelaff)

        elif "y affine" in self.name:
            
            self.sld.valueChanged.connect(parent.affineY)
            self.sld.valueChanged.connect(self.updateLabelaff)
            
        elif "z axis Nifti" in self.name:
            
            self.sld.valueChanged.connect(parent.Slice)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "zoom all" in self.name:
            
            self.sld.valueChanged.connect(parent.ZoomAll)
            self.sld.valueChanged.connect(self.updateLabelaff)

        
        self.label = QLabel(nom.format(s), self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setFixedSize(120, 15)
        
        self.box.addWidget(self.label)
        self.box.addWidget(self.sld)
        
        self.setLayout(self.box)

        
    def updateLabel(self,value):
        
        self.label.setText(self.name.format(value))
        
        
    def updateLabelZoom(self,value):
        
        self.label.setText(self.name.format(round(1.07**(value-25),2)))
        
    def updateLabelaff(self,value):
        
        self.label.setText(self.name.format(round(1.03**(value-25),2)))


class Rotor(QWidget):
    
    def __init__(self, nom, parent):
        super().__init__()
        
        self.name = nom
        self.box = QVBoxLayout()
        self.sld = QDial()
        self.sld.setRange(0, 359)
        
        if "rotation Nifti x" in self.name:
            self.sld.valueChanged.connect(parent.RotateNx)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "rotation Nifti y" in self.name:
            self.sld.valueChanged.connect(parent.RotateNy)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "rotation Nifti z" in self.name:
            self.sld.valueChanged.connect(parent.RotateNz)
            self.sld.valueChanged.connect(self.updateLabel)
            
        elif "rotation TIFF" in self.name:
            self.sld.valueChanged.connect(parent.RotateTx)
            self.sld.valueChanged.connect(self.updateLabel)
        
        self.label = QLabel(nom.format(0), self)
        #self.label.setAligment(Qt.AlignCenter | Qt.AlignVCenter)
        
        self.box.addWidget(self.label)
        self.box.addWidget(self.sld)
        
        self.setLayout(self.box)
        
    def updateLabel(self,value):
        
        self.label.setText(self.name.format(value))
        
    


class MainWindow(QMainWindow):

        
    def clickBox1(self,state):     
        
        if state != 0:
            self.bool1 = True
        else:
            self.bool1 = False
            
        self.visual()

    def clickBox2(self,state):
        
        if state != 0:
            self.bool2 = True
        else:
            self.bool2 = False
            
        self.visual()
        
    def clickFlip(self,state):    

        if state != 0:
            self.bool = True
        else:
            self.bool = False
            
        self.changed = True
        self.visual()
        
    def Save(self, state):
        
        name = QFileDialog.getSaveFileName(self,"save file")
        nom = name[0]
        
        if self.bool:
            saved= cv2.flip(cv2.resize(self.IMG2, None, fx = self.zoomTIFF*self.affix/self.zoomNifti ,fy = self.zoomTIFF*self.affiy/self.zoomNifti),1)
            #saved= cv2.flip(self.IMG2,1)
            
        else:
            saved= cv2.resize(self.IMG2, None, fx = self.zoomTIFF*self.affix/self.zoomNifti ,fy = self.zoomTIFF*self.affiy/self.zoomNifti)
            #saved = self.IMG2
            
        
        (L,C,Z) = np.shape(self.IMG1)
        
        saved_data = 0.2989*saved[...,0] + 0.5870*saved[...,1] + 0.1140*saved[...,2]
         
        (l,c) = np.shape(saved_data)
        
        array = np.zeros((l,c,1))
        array[:,:,0] = saved_data

        alpha = math.atan(l/c)
        
        beta = np.radians(self.rotTIFF-self.rotNiftiX)
        
        r = np.sqrt((l/2)**2 + (c/2)**2)
        
        affine = cp.deepcopy(self.aff)
        header = self.header
        spacing = self.header.get_zooms()
        

        # Translation
        
        vec = np.zeros(3)
        
        vec[0] = (-self.y2+self.y1)/self.zoomNifti + L//2-l//2
        vec[1] = (self.x2-self.x1)/self.zoomNifti+ C//2-c//2
        vec[2] = self.z1

        result = affine[:3,:3] @ vec
        
        affine[0,3] += result[0]
        affine[1,3] += result[1]
        affine[2,3] += result[2]
    

        # Coreection de la Rotation par Translation
        
        vec2 = np.zeros(3)
        
        vec2[0] = (l/2 + r * np.sin(beta-alpha))
        vec2[1] = (c/2 - r * np.cos(beta-alpha))
        vec2[2] = 0
        
        result = affine[:3,:3] @ vec2
        
        affine[0,3] += result[0]
        affine[1,3] += result[1]
        affine[2,3] += result[2]
        
        
        # Rotation 
        
        MatRot = np.zeros((3,3))
        MatRot[0,0] = np.cos(beta)
        MatRot[0,1] = np.sin(beta)*(-1)
        MatRot[1,0] = np.sin(beta)
        MatRot[1,1] = np.cos(beta)
        MatRot[2,2] = 1
        
        affine[:3,:3] =  affine[:3,:3] @ MatRot
        
        out = nib.Nifti1Image(array, affine)
        if not nom.endswith('.nii.gz'):
            nom+='.nii.gz'
        out.to_filename(nom)
        
  
        
    def clickBoutton1(self, state):
        
        self.im1 = self.IMG1
        self.im2 = self.IMG2
        
        self.bool = False
        self.Flip.deleteLater()
        self.Flip = CheckBox("Flip",self)
    
        self.zoomNifti = 1
        self.zoomTIFF = 1
        self.Zoomer1.deleteLater()
        self.Zoomer1 = Curseur("zoom Nifti: x{0}", 0, 50, 25, 1, self)
        self.Zoomer2.deleteLater()
        self.Zoomer2 = Curseur("zoom TIFF: x{0}", 0, 50, 25, 1, self)
        
        self.x1 = 0
        self.y1 = 0
        self.z1 = 0
        self.x2 = 0
        self.y2 = 0
        
        self.affix = 1
        self.affiy = 1
        self.affX.deleteLater()
        self.affX = Curseur("x affine :{0}", 0, 50, 25, 1, self)
        self.affY.deleteLater()
        self.affY = Curseur("y affine :{0}", 0, 50, 25, 1, self)

        self.rotNiftiX = 0
        self.rotTIFF = 0
        self.rotateNiftiX.deleteLater()
        self.rotateNiftiX = Rotor("rotation Nifti x :{0}°",self)
        self.rotateTIFF.deleteLater()
        self.rotateTIFF = Rotor("rotation TIFF :{0}°",self)
        
        # Nifti layout
        self.layout3.addWidget(self.Zoomer1,0,0)
        self.layout3.addWidget(self.volume,0,1)
        self.layout3.addWidget(self.SliderX1,1,0)
        self.layout3.addWidget(self.rotateNiftiX,1,1)
        self.layout3.addWidget(self.SliderY1,2,0)
        self.layout3.addWidget(self.rotateNiftiY,2,1)
        self.layout3.addWidget(self.SliderZ1,3,0)
        self.layout3.addWidget(self.rotateNiftiZ,3,1)
        
        #Tiff Layout
        self.layout4.addWidget(self.Zoomer2,0,0)
        self.layout4.addWidget(self.SliderX2,1,0)
        self.layout4.addWidget(self.affX,1,1)
        self.layout4.addWidget(self.SliderY2,2,0)
        self.layout4.addWidget(self.affY,2,1)
        self.layout4.addWidget(self.rotateTIFF,3,0)
        self.layout4.addWidget(self.Flip,3,1)
        
        self.changed = True
        self.visual()
            
    def Combo(self,value):
        
        if value == 0:
            self.stacked_layout.setCurrentIndex(0)
        else:
            self.stacked_layout.setCurrentIndex(1)
        
        #self.visual()
        
    def Opacity(self, value):
        
        self.opacity = value/100
        self.visual()
    
    def updateZoomN(self, value):
        
        self.zoomNifti = 1.07**(value-25)
        self.changed = True
        self.visual()
        
    def updateZoomT(self, value):
        
        self.zoomTIFF = 1.07**(value-25)
        self.changed = True
        self.visual()
        
    def updateNX(self, value):
        
        self.x1 = value
        self.visual()
        
        
    def updateNY(self, value):
        
        self.y1 = value
        self.visual()
    
            
    def updateTX(self, value):
        
        self.x2 = value
        self.visual()
        
    def updateTY(self, value):
        
        self.y2 = value
        self.visual()
        
    def affineX(self,value):
        
        self.affix = 1.03**(value-25)
        self.changed = True
        self.visual()
        
    def affineY(self,value):
        
        self.affiy = 1.03**(value-25)
        self.changed = True
        self.visual()
        
    def ZoomAll(self,value):
        
        self.zoomALL = 1.03**(value-25)

        self.changed = True
        self.visual()
        
    def RotateNx(self,value):
        
        self.rotNiftiX = value
        self.changed = True
        self.visual()
        
    def RotateNy(self,value):
        
        self.rotNiftiY = value
        self.changed = True
        self.visual()
        
    def RotateNz(self,value):
        
        self.rotNiftiZ = value
        self.changed = True
        self.visual()
        
    def RotateTx(self,value):
        
        self.rotTIFF = value
        self.changed = True
        self.visual()
    
    def nVolume(self,value):
        
        self.nVol = value
        self.Nifti.update(self.z1, self.nVol)
        self.IMG1 = self.Nifti.current
        self.changed = True
        self.visual()
        
    def Slice(self,value):
        
        self.z1 = value
        self.Nifti.update(self.z1, self.nVol)
        self.IMG1 = self.Nifti.current
        self.changed = True
        self.visual()
        
    def open1(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.tiff *.jpg *.png *.jpeg *.jpeg2000 *.nii*)')
        
        print(path)
        
        if ".nii" in path[0]:
            
            #self.nii = True
            self.bool1 = True
            self.Check1.b.setChecked(True)
            
            img = nib.load(path[0])
            
            data = img.get_fdata()
            self.header = img.header
            self.aff = img.affine
            
            self.Nifti = Nifti(data,self.aff)
            self.IMG1 = self.Nifti.current
            
            self.SliderZ1.deleteLater()
            self.SliderZ1 = Curseur("z axis Nifti :{0}", 0, self.Nifti.z1-1, 0, 1, self)
            
            self.volume.deleteLater()
            self.volume = Volume("Volume number", 0, self.Nifti.nvol-1, self)
            
            self.layout3.addWidget(self.volume,0,1)
            self.layout3.addWidget(self.SliderZ1,3,0)
            
        elif path !=('',''): 
            self.bool1 = True
            temp = cv2.imread(path[0])
            self.Check1.b.setChecked(True)
            
            if 'png' in path[0] or 'tiff' in path[0]:
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            
            
            (x,y,r) = np.shape(temp)
            
            threshold = 1440
            if x > threshold or y > threshold:
                if x >y:
                    self.IMG1 = cv2.resize(temp, None, fx = threshold/x,fy = threshold/x)
                    
                else :
                    self.IMG1 = cv2.resize(temp, None, fx = threshold/y,fy = threshold/y)
                   
            else:
                self.IMG1 = temp

            ones = np.ones(self.IMG1.shape)
            
            self.IMG1[self.IMG1[:,:]==[0,0,0]] = ones[self.IMG1[:,:]==[0,0,0]]
            
        self.changed = True   
        self.visual()
            
            
    def open2(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.tiff *.jpg *.png *.jpeg *.jpeg2000)')

        if path !=('',''):
            self.bool2 = True
            temp = cv2.imread(path[0])
            self.Check2.b.setChecked(True)
            
            if 'png' in path[0] or 'tiff' in path[0] or 'jpg' in path[0]:
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
             
            (x,y,r) = np.shape(temp)
            
            threshold = 1440
            if x > threshold or y > threshold:
                if x >y:
                    self.IMG2 = cv2.resize(temp, None, fx = threshold/x,fy = threshold/x)
                    
                else :
                    self.IMG2 = cv2.resize(temp, None, fx = threshold/y,fy = threshold/y)
                    
            else:
                self.IMG2 = temp
                
            ones = np.ones(self.IMG2.shape)
            
            self.IMG2[self.IMG2[:,:]==[0,0,0]] = ones[self.IMG2[:,:]==[0,0,0]]
            
            self.changed = True
            self.visual()

    def visual(self):
         
        if self.changed:
            
            (old_x1,old_y1,old_r1) = np.shape(self.im1)
            (old_x2,old_y2,old_r2) = np.shape(self.im2)
            
            if self.bool:
                self.im2 = cv2.flip(cv2.resize(self.IMG2, None, fx = self.zoomALL*self.zoomTIFF*self.affix ,fy = self.zoomALL*self.zoomTIFF*self.affiy),1)
            else:
                self.im2 = cv2.resize(self.IMG2, None, fx = self.zoomALL*self.zoomTIFF*self.affix ,fy = self.zoomALL*self.zoomTIFF*self.affiy)
            self.im1 = cv2.resize(self.IMG1, None, fx = self.zoomALL*self.zoomNifti ,fy = self.zoomALL*self.zoomNifti)
            
            if self.rotNiftiX !=0:
                self.im1 = ndimage.interpolation.rotate(self.im1, self.rotNiftiX, (1,2), reshape=True)
            if self.rotNiftiY !=0:
                self.im1 = ndimage.interpolation.rotate(self.im1, self.rotNiftiY, (1,2), reshape=True)
            if self.rotNiftiZ !=0:
                self.im1 = ndimage.interpolation.rotate(self.im1, self.rotNiftiZ, (0,1), reshape=True)
                
            if self.rotTIFF != 0:
                self.im2 = ndimage.interpolation.rotate(self.im2, self.rotTIFF, (0,1), reshape=True)
            
            (x1,y1,r1) = np.shape(self.im1)
            (x2,y2,r2) = np.shape(self.im2)
            
                    
            self.x1 = int(self.x1*(y2/old_y2))
            self.y1 = int(self.y1*(x2/old_x2))
            self.x2 = int(self.x2*(y1/old_y1))
            self.y2 = int(self.y2*(x1/old_x1))
            
            self.SliderX1.deleteLater()
            self.SliderY1.deleteLater()
            
            self.SliderX1 = Curseur("x axis Nifti: {0} px", -(y2//2) +1, y2//2 -1, self.x1, 1, self)
            self.SliderY1 = Curseur("y axis Nifti: {0} px", -(x2//2) +1, x2//2 -1, self.y1, 1, self)
            #self.SliderZ1 = Curseur("z axis Nifti :{0} px", 0, 20, 10, 1, self)
            
            self.layout3.addWidget(self.SliderX1,1,0)
            self.layout3.addWidget(self.SliderY1,2,0)
            
            self.SliderX2.deleteLater()
            self.SliderY2.deleteLater()
            
            self.SliderX2 = Curseur("x axis TIFF: {0} px", -(y1//2) +1, y1//2 -1, self.x2 , 1, self)
            self.SliderY2 = Curseur("y axis TIFF: {0} px", -(x1//2) +1, x1//2 -1, self.y2 , 1, self)
            
            self.layout4.addWidget(self.SliderX2,1,0)
            self.layout4.addWidget(self.SliderY2,2,0)
            
            self.changed = False
        
        ######
        (x1,y1,r1) = np.shape(self.im1)
        (x2,y2,r2) = np.shape(self.im2)
        
        size1 = x1+x2
        size2 = y1+y2
        
        self.final = np.ones((size1,size2,3))*236

        X1M1 = -x1//2 + size1//2 - self.y1
        X2M1 = x1//2 + size1//2 - self.y1
        Y1M1 = -y1//2 + size2//2 + self.x1
        Y2M1 = y1//2 + size2//2 + self.x1
        
        X1M2 = -x2//2 + size1//2 - self.y2
        X2M2 = x2//2 + size1//2 - self.y2
        Y1M2 = -y2//2 + size2//2 + self.x2
        Y2M2 = y2//2 + size2//2 + self.x2

        if self.bool1 == True and self.bool2 == False:
            
            tempo = self.final[X1M1:X2M1,Y1M1:Y2M1,:]
            
            tempo[self.im1[:,:]!=[0,0,0]] = self.im1[self.im1[:,:]!=[0,0,0]]
            
            self.final[X1M1:X2M1,Y1M1:Y2M1,:] = tempo

        
        elif self.bool1 == False and self.bool2 == True:
            
            tempo = self.final[X1M2:X2M2,Y1M2:Y2M2,:]
            
            tempo[self.im2[:,:]!=[0,0,0]] = self.im2[self.im2[:,:]!=[0,0,0]]
            
            self.final[X1M2:X2M2,Y1M2:Y2M2,:] = tempo

                    
        elif self.bool1 == True and self.bool2 == True:

            tempo1 = self.final[X1M1:X2M1,Y1M1:Y2M1,:]
            
            tempo1[self.im1[:,:]!=[0,0,0]] = self.im1[self.im1[:,:]!=[0,0,0]]
            
            self.final[X1M1:X2M1,Y1M1:Y2M1,:] = tempo1
 
            tempo2 = self.final[X1M2:X2M2,Y1M2:Y2M2,:]
            
            tempo2[self.im2[:,:]!=[0,0,0]] = tempo2[self.im2[:,:]!=[0,0,0]]*(1-self.opacity)
            
            tempo2[self.im2[:,:]!=[0,0,0]] += self.im2[self.im2[:,:]!=[0,0,0]]*self.opacity
            
            self.final[X1M2:X2M2,Y1M2:Y2M2,:] = tempo2
                 
            
            
        self.final = self.final.astype(np.uint8)
        #print(self.IMG1.shape)
        
        
        #self.final = cv2.imread('/Users/eliotthenaut/Desktop/test.jpg')
        #print(kake.all()==self.final.all())

        Img = Image.fromarray(self.final,mode='RGB')
        qt_image = ImageQt.ImageQt(Img)
        
        #label = QLabel(self)
        IMG = QPixmap.fromImage(qt_image)

        self.output.setPixmap(IMG)
        
    def LayoutBuilder(self):
        
        #Main upper layout
        self.layout2.addWidget(self.load1,0,0)
        self.layout2.addWidget(self.Check1,0,1)
        self.layout2.addWidget(self.load2,1,0)
        self.layout2.addWidget(self.Check2,1,1)
        self.layout2.addWidget(self.comboBox,2,0,1,2)

        
        
        # Nifti layout
        self.layout3.addWidget(self.Zoomer1,0,0)
        self.layout3.addWidget(self.volume,0,1)
        self.layout3.addWidget(self.SliderX1,1,0)
        self.layout3.addWidget(self.rotateNiftiX,1,1)
        self.layout3.addWidget(self.SliderY1,2,0)
        self.layout3.addWidget(self.rotateNiftiY,2,1)
        self.layout3.addWidget(self.SliderZ1,3,0)
        self.layout3.addWidget(self.rotateNiftiZ,3,1)
        
        #Tiff Layout
        self.layout4.addWidget(self.Zoomer2,0,0)
        self.layout4.addWidget(self.SliderX2,1,0)
        self.layout4.addWidget(self.affX,1,1)
        self.layout4.addWidget(self.SliderY2,2,0)
        self.layout4.addWidget(self.affY,2,1)
        self.layout4.addWidget(self.rotateTIFF,3,0)
        self.layout4.addWidget(self.Flip,3,1)
        
        #Main lower layout
        self.layout5.addWidget(self.ZoomerAll,0,0)
        self.layout5.addWidget(self.SliderOpacity,0,1)
        
        # Bottom save buttons 
        self.layout6.addWidget(self.boutton1,1,0)
        self.layout6.addWidget(self.boutton2,1,1)
        self.layout6.addWidget(self.boutton3,1,2)
        
        # Bottom layout
        self.layout7.addWidget(self.output,0,0)
        self.layout7.addLayout(self.layout6,1,0)
        
        #Assembling Left Layout
        widg1=QWidget()
        widg1.setLayout(self.layout3)
        widg2=QWidget()
        widg2.setLayout(self.layout4)
        self.stacked_layout.addWidget(widg1)
        self.stacked_layout.addWidget(widg2)
        
        self.layout9.addLayout(self.layout2,0,0)
        self.layout9.addLayout(self.stacked_layout,1,0)
        self.layout9.addLayout(self.layout5,2,0)

        self.Base.addLayout(self.layout9,0,0)
        self.Base.addLayout(self.layout7,0,1)
        self.Base.setColumnStretch(0, 1)
        self.Base.setColumnStretch(1, 3)
        
        self.widget = QWidget()
        self.widget.setLayout(self.Base)
        self.setCentralWidget(self.widget)
        
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.setWindowTitle("My App")
        self.resize(1440,700)
        
        self.changed = True
        self.header = None
        self.aff = np.eye(4)
        
        self.IMG1 = np.ones((500,500,3))*236
        self.IMG2 = np.ones((500,500,3))*236
        
        self.im1 = np.ones((1,1,3))*236
        self.im2 = np.ones((1,1,3))*236
        self.final = np.ones((1,1,3))*236
          
        
        self.bool1 = False
        self.bool2 = False
        self.Check1 = CheckBox("Display Nifti",self)
        self.Check2 = CheckBox("Display TIFF",self)
        
        self.bool = False
        self.Flip = CheckBox("Flip",self)
        
        self.opacity = 1        
        self.SliderOpacity = Curseur("opacity: {0} %", 0, 100, 100, 1, self)
    
        self.zoomNifti = 1
        self.zoomTIFF = 1
        self.zoomALL = 1
        self.Zoomer1 = Curseur("zoom Nifti: x{0}", 0, 50, 25, 1, self)
        self.Zoomer2 = Curseur("zoom TIFF: x{0}", 0, 50, 25, 1, self)
        self.ZoomerAll = Curseur("zoom all: x{0}", 0, 50, 25, 1, self)
        
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.z1 = 0
        
        self.SliderX1 = Curseur("x axis Nifti :{0}", 0, 20, 10, 1, self)
        self.SliderY1 = Curseur("y axis Nifti :{0}", 0, 20, 10, 1, self)
        self.SliderX2 = Curseur("x axis TIFF :{0}", 0, 20, 10, 1, self)
        self.SliderY2 = Curseur("y axis TIFF :{0}", 0, 20, 10, 1, self)
        self.SliderZ1 = Curseur("z axis Nifti :{0}", 0, 20, 0, 1, self)
        
        self.affix = 1
        self.affiy = 1
        
        self.affX = Curseur("x affine :{0}", 0, 50, 25, 1, self)
        self.affY = Curseur("y affine :{0}", 0, 50, 25, 1, self)

        self.rotNiftiX = 0
        self.rotNiftiY = 0
        self.rotNiftiZ = 0
        self.rotTIFF = 0

        
        self.rotateNiftiX = Rotor("rotation Nifti x :{0}°",self)
        self.rotateNiftiY = Rotor("rotation Nifti y :{0}°",self)
        self.rotateNiftiZ = Rotor("rotation Nifti z :{0}°",self)
        self.rotateTIFF = Rotor("rotation TIFF :{0}°",self)

        
        self.nVol = 0

        self.volume = Volume("Volume number", 0, 10, self)
        
        self.combo = True
        self.comboBox = ComboBox("ComboBox",self)
        
        self.load1 = QPushButton('&Load NIFTI', clicked = self.open1)
        self.load2 = QPushButton('&Load TIFF', clicked = self.open2)
        

        
        self.boutton1 = Boutton('&Clear Settings', self)
        self.boutton2 = QPushButton('&boutton2')
        self.boutton3 = Boutton('&Save File', self)

        
        self.output = QLabel()
        self.output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output.setFixedSize(1000, 700)
        
        Img = Image.fromarray(self.final,mode='RGB')
        qt_image = ImageQt.ImageQt(Img)
        
        #label = QLabel(self)
        IMG = QPixmap.fromImage(qt_image)

        self.output.setPixmap(IMG)
        
        """
        
        layout1 = QGridLayout()
        self.layout2 = QGridLayout()
        layout3 = QGridLayout()
        layout4 = QGridLayout()

        layout4.addWidget(boutton1,0,0)
        layout4.addWidget(boutton2,0,1)
        layout4.addWidget(boutton3,0,2)
        
        self.layout2.addWidget(self.load1,0,0)
        self.layout2.addWidget(self.Check1,0,1)
        
        #self.layout2.addWidget(self.label,1,0)
        self.layout2.addWidget(self.Zoomer1,1,0)
        self.layout2.addWidget(self.volume,1,1)
        
        self.layout2.addWidget(self.SliderX1,2,0)
        self.layout2.addWidget(self.rotateNiftiX,2,1)
        self.layout2.addWidget(self.SliderY1,3,0)
        self.layout2.addWidget(self.rotateNiftiY,3,1)
        self.layout2.addWidget(self.SliderZ1,4,0)
        self.layout2.addWidget(self.rotateNiftiZ,4,1)
        
        self.layout2.addWidget(self.load2,5,0)
        self.layout2.addWidget(self.Check2,5,1)
        self.layout2.addWidget(self.Zoomer2,6,0)
        self.layout2.addWidget(self.SliderOpacity,6,1)
        self.layout2.addWidget(self.SliderX2,7,0)
        self.layout2.addWidget(self.affX,7,1)
        self.layout2.addWidget(self.SliderY2,8,0)
        self.layout2.addWidget(self.affY,8,1)
        self.layout2.addWidget(self.rotateTIFFX,9,0)
        self.layout2.addWidget(self.Flip,9,1)
        
        self.layout2.addWidget(self.ZoomerAll,10,0)
        
        layout3.addWidget(self.output,0,0)
        layout3.addLayout(layout4,1,0)
        
        layout1.addLayout(self.layout2,0,0)
        layout1.addLayout(layout3,0,1)
        layout1.setColumnStretch(0,1)
        layout1.setColumnStretch(1,3)

        layout1 = QGridLayout()
        self.layout2 = QGridLayout()
        self.layout3 = QGridLayout()
        layout4 = QGridLayout()
        layout5 = QGridLayout()

        layout5.addWidget(boutton1,0,0,1,2)

        layout5.addWidget(boutton3,0,3,1,2)
        
        self.layout2.addWidget(self.load1,0,0)
        self.layout2.addWidget(self.Check1,1,0)
        self.layout2.addWidget(self.Zoomer1,2,0)
        self.layout2.addWidget(self.volume,3,0)
        self.layout2.addWidget(self.SliderX1,4,0)
        self.layout2.addWidget(self.rotateNiftiX,5,0)
        self.layout2.addWidget(self.SliderY1,6,0)
        self.layout2.addWidget(self.rotateNiftiY,7,0)
        self.layout2.addWidget(self.SliderZ1,8,0)
        self.layout2.addWidget(self.rotateNiftiZ,9,0)
        
        self.layout3.addWidget(self.load2,0,0)
        self.layout3.addWidget(self.Check2,1,0)
        self.layout3.addWidget(self.Zoomer2,2,0)
        self.layout3.addWidget(self.SliderOpacity,3,0)
        self.layout3.addWidget(self.SliderX2,4,0)
        self.layout3.addWidget(self.affX,5,0)
        self.layout3.addWidget(self.SliderY2,6,0)
        self.layout3.addWidget(self.affY,7,0)
        self.layout3.addWidget(self.rotateTIFFX,8,0)
        self.layout3.addWidget(self.Flip,10,0)
        
        layout5.addWidget(self.ZoomerAll,0,2)
        
        layout4.addWidget(self.output,0,0)
        layout4.addLayout(layout5,1,0)
        
        layout1.addLayout(self.layout2,0,0)
        layout1.addLayout(layout4,0,1)
        layout1.addLayout(self.layout3,0,2)
        layout1.setColumnStretch(0,1)
        layout1.setColumnStretch(1,3)
        layout1.setColumnStretch(2,1)
               
        
        self.widget = QWidget()
        self.widget.setLayout(layout1)
        self.setCentralWidget(self.widget)
        """
        
        self.Base = QGridLayout()
        self.layout2 = QGridLayout()
        self.layout3 = QGridLayout()
        self.layout4 = QGridLayout()
        self.layout5 = QGridLayout()
        self.layout6 = QGridLayout()
        self.layout7 = QGridLayout()
        self.layout9 = QGridLayout()
        self.stacked_layout = QStackedLayout()
        self.LayoutBuilder()

app = QApplication(sys.argv)
window = MainWindow()

window.show()

app.exec()
