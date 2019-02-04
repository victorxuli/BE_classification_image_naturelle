# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:22:46 2018

@author: victo
"""

from skimage import feature as ft
from skimage import util as ut
from PIL import Image
import skimage.measure as skm
import numpy as np
import scipy.fftpack as sc
import scipy as sp
import skimage.io as skio
from skimage import exposure as ske
from skimage import color
import matplotlib.pyplot as plt
from skimage import filters as fl
from skimage import img_as_float as img_as_float
from skimage.segmentation import clear_border
import skimage.measure as skim
from skimage.morphology import disk
import skimage.restoration as re
import skimage 
import skimage.morphology as sm
import cv2
from skimage import data_dir
import re
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
#from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import pickle
# classe 0 : deserts沙漠
# classe 1 : forets森林
# classe 2 : indoors室内
# classe 3 : plages沙滩
# classe 4 : villes城市

def convert_gray(f):
    rgb=skio.imread(f)
    return color.rgb2gray(rgb)

def convert_lab(f):
    rgb=skio.imread(f)
    return color.rgb2lab(rgb)

def performance(mat):
    perf = np.zeros((5,1))
    for j in range(5):
        perf[j,0] = mat[j,j]/sum(mat[j,:])
    return perf   
    
    
def matConf(predict,grps):
    mat = np.zeros((5,5))
    predict = np.int16(predict)
    grps = np.int16(grps)
    for i in range(len(grps)):
        mat[grps[i],predict[i]]+=1
    return mat

plt.close("all")
str='D:\data/*.jpg'
coll = skio.ImageCollection(str)
coll_gray = skio.ImageCollection(str,load_func=convert_gray)
coll_lab = skio.ImageCollection(str,load_func=convert_lab)
filenames = coll.files
labels = np.zeros((319,1))
lenth = len(coll)
#图像标签
for i in range(319):
    if(filenames[i][8] == 'd'):
        labels[i+1,0] = 0
    if(filenames[i][8] == 'f'):
        labels[i,0] = 1
    if(filenames[i][8] == 'i'):
        labels[i,0] = 2
    if(filenames[i][8] == 'p'):
        labels[i,0] = 3
    if(filenames[i][8] == 'v'):
        labels[i,0] = 4
        
#分类标签索引        
deserts = np.where(labels==0)
forets = np.where(labels==1)
indoors = np.where(labels==2)
plages = np.where(labels==3)
villes = np.where(labels==4)

##lab图像各通道均值
m_deserts = np.zeros((65,3))
m_forets = np.zeros((75,3))
m_plages = np.zeros((19,3))
m_indoors = np.zeros((42,3))
m_villes = np.zeros((118,3))
##lab图像各通道平均能量
e_deserts = np.zeros((65,3))
e_forets = np.zeros((75,3))
e_plages = np.zeros((19,3))
e_indoors = np.zeros((42,3))
e_villes = np.zeros((118,3))
a_b_pourc = np.zeros((lenth,2))
e_RGB = np.zeros((lenth,3))
for i in range(lenth):
    cmp1 = coll_lab[i][:,:,1][coll_lab[i][:,:,1]>0]
    cmp2 = coll_lab[i][:,:,2][coll_lab[i][:,:,2]>0]
    a_b_pourc[i,0] = cmp1.shape[0]/(coll_lab[i][:,:,:].shape[0]*coll_lab[i][:,:,:].shape[1])
    a_b_pourc[i,1] = cmp2.shape[0]/(coll_lab[i][:,:,:].shape[0]*coll_lab[i][:,:,:].shape[1])
    e_RGB[i][0] = np.mean(coll[i][:,:,0] * coll[i][:,:,0])
    e_RGB[i][1] = np.mean(coll[i][:,:,1] * coll[i][:,:,1])
    e_RGB[i][2] = np.mean(coll[i][:,:,2] * coll[i][:,:,2])
    

#########################################################################         LAB moyenne et energies 
###### 沙漠学习
for i in range(65):
    moyenne_deserts_l = np.mean(coll_lab[deserts[0][i]][:,:,0])
    m_deserts[i][0] = moyenne_deserts_l
    e_deserts[i][0] = np.mean(coll_lab[deserts[0][i]][:,:,0] * coll_lab[deserts[0][i]][:,:,0])
    moyenne_deserts_a = np.mean(coll_lab[deserts[0][i]][:,:,1])
    m_deserts[i][1] = moyenne_deserts_a
    e_deserts[i][1] = np.mean(coll_lab[deserts[0][i]][:,:,1] * coll_lab[deserts[0][i]][:,:,1])
    moyenne_deserts_b = np.mean(coll_lab[deserts[0][i]][:,:,2])
    m_deserts[i][2] = moyenne_deserts_b
    e_deserts[i][2] = np.mean(coll_lab[deserts[0][i]][:,:,2] * coll_lab[deserts[0][i]][:,:,2])
    
####### 森林
for i in range(75):
    moyenne_forets_l = np.mean(coll_lab[forets[0][i]][:,:,0])
    m_forets[i][0] = moyenne_forets_l
    e_forets[i][0] = np.mean(coll_lab[forets[0][i]][:,:,0] * coll_lab[forets[0][i]][:,:,0])
    moyenne_forets_a = np.mean(coll_lab[forets[0][i]][:,:,1])
    m_forets[i][1] = moyenne_forets_a
    e_forets[i][1] = np.mean(coll_lab[forets[0][i]][:,:,1] * coll_lab[forets[0][i]][:,:,1])
    moyenne_forets_b = np.mean(coll_lab[forets[0][i]][:,:,2])
    m_forets[i][2] = moyenne_forets_b
    e_forets[i][0] = np.mean(coll_lab[forets[0][i]][:,:,2] * coll_lab[forets[0][i]][:,:,2])

####### 沙滩
for i in range(19):
    moyenne_plage_l = np.mean(coll_lab[plages[0][i]][:,:,0])
    m_plages[i][0] = moyenne_plage_l
    e_plages[i][0] = np.mean(coll_lab[plages[0][i]][:,:,0] * coll_lab[plages[0][i]][:,:,0])
    moyenne_plage_a = np.mean(coll_lab[plages[0][i]][:,:,1])
    m_plages[i][1] = moyenne_plage_a
    e_plages[i][1] = np.mean(coll_lab[plages[0][i]][:,:,1] * coll_lab[plages[0][i]][:,:,1])
    moyenne_plage_b = np.mean(coll_lab[plages[0][i]][:,:,2])
    m_plages[i][2] = moyenne_plage_b
    e_plages[i][2] = np.mean(coll_lab[plages[0][i]][:,:,2] * coll_lab[plages[0][i]][:,:,2])
    
####### 室内
for i in range(42):
    moyenne_indoor_l = np.mean(coll_lab[indoors[0][i]][:,:,0])
    m_indoors[i][0] = moyenne_indoor_l
    e_indoors[i][0] = np.mean(coll_lab[indoors[0][i]][:,:,0] * coll_lab[indoors[0][i]][:,:,0])
    moyenne_indoor_a = np.mean(coll_lab[indoors[0][i]][:,:,1])
    m_indoors[i][1] = moyenne_indoor_a
    e_indoors[i][1] = np.mean(coll_lab[indoors[0][i]][:,:,1] * coll_lab[indoors[0][i]][:,:,1])
    moyenne_indoor_b = np.mean(coll_lab[indoors[0][i]][:,:,2])
    m_indoors[i][2] = moyenne_indoor_b
    e_indoors[i][2] = np.mean(coll_lab[indoors[0][i]][:,:,2] * coll_lab[indoors[0][i]][:,:,2])    
    
####### 城市
for i in range(118):
    moyenne_ville_l = np.mean(coll_lab[villes[0][i]][:,:,0])
    m_villes[i][0] = moyenne_ville_l
    e_villes[i][0] = np.mean(coll_lab[villes[0][i]][:,:,0] * coll_lab[villes[0][i]][:,:,0])
    moyenne_ville_a = np.mean(coll_lab[villes[0][i]][:,:,1])
    m_villes[i][1] = moyenne_ville_a
    e_villes[i][1] = np.mean(coll_lab[villes[0][i]][:,:,1] * coll_lab[villes[0][i]][:,:,1])
    moyenne_ville_b = np.mean(coll_lab[villes[0][i]][:,:,2])
    m_villes[i][2] = moyenne_ville_b
    e_villes[i][2] = np.mean(coll_lab[villes[0][i]][:,:,2] * coll_lab[villes[0][i]][:,:,2])
###轮廓平均能量
#e_lab = np.zeros((lenth,3))
#for i in range(lenth):
#    e_lab[i,0] = np.mean(coll_lab[i][:,:,0] * coll_lab[i][:,:,0])
#    e_lab[i,1] = np.mean(coll_lab[i][:,:,1] * coll_lab[i][:,:,1])
#    e_lab[i,2] = np.mean(coll_lab[i][:,:,2] * coll_lab[i][:,:,2])
    
e_contour = np.zeros((lenth,1))
for i in range(lenth):
    coll_contour = coll_gray[i][:,:] - fl.gaussian(coll_gray[i][:,:])
    e_contour[i,0] = np.mean(coll_contour * coll_contour)/np.mean(coll_gray[i][:,:] * coll_gray[i][:,:])
####水平/竖直轮廓平均能量
e_contour_h = np.zeros((lenth,1))
e_contour_v = np.zeros((lenth,1))
for i in range(lenth):
    e_contour_h[i,0] = np.mean(skimage.img_as_ubyte(fl.sobel_v(coll_gray[i][:,:])) * skimage.img_as_ubyte(fl.sobel_v(coll_gray[i][:,:])))
    e_contour_v[i,0] = np.mean(skimage.img_as_ubyte(fl.sobel_v(coll_gray[i][:,:])) * skimage.img_as_ubyte(fl.sobel_v(coll_gray[i][:,:])))

####描述因子X 12维
X = np.zeros((lenth,12))
X[0:65,0:3] = m_deserts
X[0:65,3:6] = e_deserts

X[65:140,0:3] = m_forets
X[65:140,3:6] = e_forets

X[140:182,0:3] = m_indoors
X[140:182,3:6] = e_indoors

X[182:201,0:3] = m_plages
X[182:201,3:6] = e_plages

X[201:319,0:3] = m_villes
X[201:319,3:6] = e_villes

X[0:319,6:7] = e_contour
X[0:319,7:9] = a_b_pourc

X[0:319,9:12] = e_RGB


##hsv图像各通道平均能量 / H通道百分比
e_hsv = np.zeros((lenth,3))
num_H = np.zeros((lenth,6))

for i in range(lenth):
    coll_hsv  = cv2.cvtColor(coll[i][:,:,:],cv2.COLOR_BGR2HSV)
    H = ske.histogram(coll_hsv[:,:,0])
    x = H[1][0:len(H[1])]
    y = H[0][0:len(H[0])]
    y_sum = sum(y)
    b1 = (x<=31)&(x>=0)
    b2 = (x<=63)&(x>=32)
    b3 = (x<=95)&(x>=64)
    b4 = (x<=127)&(x>=96)
    b5 = (x<=159)&(x>=128)
    b6 = (x<=191)&(x>=160)
#    b7 = (x<=255)&(x>=191)
    num_H[i][0] = sum(y[b1]) / y_sum
    num_H[i][1] =sum(y[b2])/ y_sum
    num_H[i][2] =sum(y[b3])/ y_sum
    num_H[i][3] =sum(y[b4])/ y_sum
    num_H[i][4] =sum(y[b5])/ y_sum
    num_H[i][5] =sum(y[b6])/ y_sum
#    num_H[i][6] =sum(y[b7])/ y_sum   


X1 = np.zeros((lenth,6))
X1[:,0:6] = num_H

X2 = np.zeros((lenth,11))
X2[:,0:6] = X1[:,0:6]
X2[:,6] = e_contour_h[:,0]
X2[:,7] = e_contour_v[:,0]
X2[:,8:11] = X[:,3:6]
pca = PCA(n_components=3)

newX1 = pca.fit_transform(X1)
print(pca.explained_variance_ratio_) 

newX2 = pca.fit_transform(X2)
print(pca.explained_variance_ratio_)

plt.figure(2)
plt.scatter(newX1[0:65, 0], newX1[0:65, 1],s=30,c='red',marker='o',alpha=0.5,label="deserts")
plt.scatter(newX1[65:140, 0], newX1[65:140, 1],s=30,c='blue',marker='x',alpha=0.5,label="forets")
plt.scatter(newX1[140:181, 0], newX1[140:181, 1],s=30,c='green',marker='v',alpha=0.5,label="indoor")
plt.scatter(newX1[181:200, 0], newX1[181:200, 1],s=30,c='black',marker='<',alpha=0.5,label="plages")
plt.scatter(newX1[200:319, 0], newX1[200:319, 1],s=30,c='yellow',marker='>',alpha=0.5,label="ville")
plt.legend(loc='upper left')

plt.figure(1)
plt.scatter(newX2[0:65, 0], newX2[0:65, 1],s=30,c='red',marker='o',alpha=0.5,label="deserts")
plt.scatter(newX2[65:140, 0], newX2[65:140, 1],s=30,c='blue',marker='x',alpha=0.5,label="forets")
plt.scatter(newX2[140:181, 0], newX2[140:181, 1],s=30,c='green',marker='v',alpha=0.5,label="indoor")
plt.scatter(newX2[181:200, 0], newX2[181:200, 1],s=30,c='black',marker='<',alpha=0.5,label="plages")
plt.scatter(newX2[200:319, 0], newX2[200:319, 1],s=30,c='yellow',marker='>',alpha=0.5,label="ville")
plt.legend(loc='upper left')


filename = 'gabor.data'
f = open(filename,'rb')
X_lire = pickle.load(f)
print(X_lire)

X3 = np.zeros((lenth,6+3+2))
#X3[:,0:12] = X_lire[:,0:12]

X3[:,0:6] = X2[:,0:6]
X3[:,6:9] = X2[:,8:11]
X3[:,9:11] = X2[:,6:7]

X5 = np.zeros((lenth,3))
X5[:,0:3] = X2[:,8:11]
labels = np.array(labels)

kf = KFold(4 , True , np.random)
mat = np.zeros((5,5))
for train_index,test_index in kf.split(X3):
    train_datas = X3[train_index];
    train_grps = labels[train_index];
    test_datas = X3[test_index];
    test_grps = labels[test_index];
    print(test_index)
    print(train_index)
    print('====================================')
    phb = GaussianNB()
    phb.fit(train_datas,train_grps)
        
#        Faire la Prediction
    predict = phb.predict(test_datas)
#        Matrice de confusion
    mat += matConf(predict, test_grps)

print(mat)
print(performance(mat))   
    

##加gabor
[[0.78461538]
 [0.96      ]
 [0.88095238]
 [0.73684211]
 [0.87288136]]
##sans gabor
[[0.86153846]
 [1.        ]
 [0.5952381 ]
 [0.78947368]
 [0.8220339 ]]



#newX_lire = pca.fit_transform(X_lire)
#print(pca.explained_variance_ratio_)
#plt.figure(1)
#plt.scatter(newX_lire[0:65, 0], newX_lire[0:65, 1],s=30,c='red',marker='o',alpha=0.5,label="deserts")
#plt.scatter(newX_lire[65:140, 0], newX_lire[65:140, 1],s=30,c='blue',marker='x',alpha=0.5,label="forets")
#plt.scatter(newX_lire[140:181, 0], newX_lire[140:181, 1],s=30,c='green',marker='v',alpha=0.5,label="indoor")
#plt.scatter(newX_lire[181:200, 0], newX_lire[181:200, 1],s=30,c='black',marker='<',alpha=0.5,label="plages")
#plt.scatter(newX_lire[200:319, 0], newX_lire[200:319, 1],s=30,c='yellow',marker='>',alpha=0.5,label="ville")
#plt.legend(loc='upper left')