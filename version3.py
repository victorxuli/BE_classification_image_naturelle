# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:21:19 2018

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


def convert_gray(f):
    rgb=skio.imread(f)
    return color.rgb2gray(rgb)

def convert_lab(f):
    rgb=skio.imread(f)
    return color.rgb2lab(rgb)

#### 读取训练集图片
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

##lab图像各通道平均能量
e_lab = np.zeros((lenth,3))
e_contour = np.zeros((lenth,1))
a_b_pourc = np.zeros((lenth,2))
for i in range(lenth):
    e_lab[i][0] = np.mean(coll_lab[i][:,:,0] * coll_lab[i][:,:,0])
    e_lab[i][1] = np.mean(coll_lab[i][:,:,1] * coll_lab[i][:,:,2])
    e_lab[i][2] = np.mean(coll_lab[i][:,:,1] * coll_lab[i][:,:,2])
for i in range(lenth):
    coll_contour = coll_gray[i][:,:] - fl.gaussian(coll_gray[i][:,:])
    e_contour[i,0] = np.mean(coll_contour * coll_contour)/np.mean(coll_gray[i][:,:] * coll_gray[i][:,:])
    cmp1 = coll_lab[i][:,:,1][coll_lab[i][:,:,1]>0]
    cmp2 = coll_lab[i][:,:,2][coll_lab[i][:,:,2]>0]
    a_b_pourc[i,0] = cmp1.shape[0]/(coll_lab[i][:,:,:].shape[0]*coll_lab[i][:,:,:].shape[1])
    a_b_pourc[i,1] = cmp2.shape[0]/(coll_lab[i][:,:,:].shape[0]*coll_lab[i][:,:,:].shape[1])


##hsv图像各通道平均能量
e_hsv = np.zeros((lenth,3))

for i in range(lenth):
    coll_hsv  = cv2.cvtColor(coll[i][:,:,:],cv2.COLOR_BGR2HSV)  
    e_hsv[i][0] = np.mean(coll_hsv[:,:,0] * coll_hsv[:,:,0])
    e_hsv[i][1] = np.mean(coll_hsv[:,:,1] * coll_hsv[:,:,1])
    e_hsv[i][2] = np.mean(coll_hsv[:,:,2] * coll_hsv[:,:,2])

pca = PCA(n_components=3)
#
X1 = np.zeros((lenth,9))
X1[:,0:3] = e_lab
X1[:,3] = e_contour[:,0]
X1[:,4:6] = a_b_pourc
X1[:,6:9] = e_hsv


#newX1 = pca.fit_transform(X1)
#print(pca.explained_variance_ratio_)


img_test = cv2.imread(r'D:\image\test.jpg',1)
img_test_lab = color.rgb2lab(img_test)
img_test_gray = color.rgb2gray(img_test)
test_hsv  = cv2.cvtColor(img_test,cv2.COLOR_BGR2HSV)
img_test_contour = img_test_gray - fl.gaussian(img_test_gray)
taille_img_test = img_test_lab.shape

feature_test = np.zeros((1,9))
feature_test[0,0] = np.mean(img_test_lab[:,:,0] * img_test_lab[:,:,0])
feature_test[0,1] = np.mean(img_test_lab[:,:,1] * img_test_lab[:,:,1])
feature_test[0,2] = np.mean(img_test_lab[:,:,2] * img_test_lab[:,:,2])
feature_test[0,3] = np.mean(img_test_contour * img_test_contour)


cmp1 = img_test_lab[:,:,1][img_test_lab[:,:,1]>0]
cmp2 = img_test_lab[:,:,2][img_test_lab[:,:,2]>0]
a_pourc = cmp1.shape[0]/(taille_img_test[0]*taille_img_test[1])
b_pourc = cmp2.shape[0]/(taille_img_test[0]*taille_img_test[1])

feature_test[0,4] = a_pourc
feature_test[0,5] = b_pourc


e_test_hsv = np.zeros((1,3))
e_test_hsv[0][0] = np.mean(test_hsv[:,:,0] * test_hsv[:,:,0])
e_test_hsv[0][1] = np.mean(test_hsv[:,:,1] * test_hsv[:,:,1])
e_test_hsv[0][2] = np.mean(test_hsv[:,:,2] * test_hsv[:,:,2])

feature_test[0,6:9] = e_test_hsv
#newfeature_test = pca.fit_transform(feature_test)


clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')

clf.fit(X1,labels)

#for i in range(lenth):
#    a = X1[i,0:9].reshape(1,-1)
#    res = clf.predict(a)
#    if(res==0):
#        print('deserts')
#    if(res==1):
#        print('forets')
#    if(res==2):
#        print('indoors')
#    if(res==3):
#        print('plages')
#    if(res==4):
#        print('villes')



a = feature_test.reshape(1,-1)
res = clf.predict(a)
if(res==0):
   print('deserts')
if(res==1):
    print('forets')
if(res==2):
    print('indoors')
if(res==3):
    print('plages')
if(res==4):
    print('villes')



#res = [] #滤波结果
#
#plt.figure('image original')
#plt.imshow(coll[200][:,:,:])
#filters = build_filters()
#for i in range(len(filters)):
#    res1 = process(coll[200][:,:,:],filters[i])
#    res.append(np.asarray(res1))
#for i in range(len(res)) :
#    plt.figure('resultats')
#    plt.subplot(4,6,i+1)
#    plt.imshow(res[i], cmap='gray' )
#
#for i in range(len(res)):
#    plt.figure('gabor filters')
#    plt.subplot(4,6,i+1)
#    plt.imshow(filters[i], cmap='gray' )
#    
#a = color.rgb2gray(res[0][:,:,:]) 
#plt.figure()
#plt.imshow(a,cmap='gray')
#
#b = ft.canny(a)