# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:31:05 2018

@author: victo
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import scipy as sc
from skimage import data, io
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage import color
from scipy.signal import hamming
from skimage import feature as ft
from skimage import util as ut
from PIL import Image
import skimage.measure as skm
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
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import pickle







filename = 'gabor.data'
f = open(filename,'rb')
X_gabor = pickle.load(f)
print(X_gabor)
taille_gabor = X_gabor.shape

filename = 'X2.data'
f = open(filename,'rb')
X2 = pickle.load(f)
print(X2)
taille_X2 = X_gabor.shape

feature = np.zeros((taille_gabor[0],taille_gabor[1]+taille_X2[1]))
feature[:,0:taille_gabor[1]] = X_gabor
feature[:,taille_gabor[1]:taille_gabor[1]+taille_X2[1]] = X_gabor


