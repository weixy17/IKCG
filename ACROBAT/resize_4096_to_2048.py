import os
import csv
import skimage.io, skimage.filters
import numpy as np
from timeit import default_timer
import pdb
from skimage import io
import pandas as pd
from numpy import *
import skimage
import random
import cv2
import scipy.io as scio
import matplotlib.pyplot as plt
data_path="/data/wxy/association/Maskflownet_association_1024/dataset/"
prep_path1 = "/data/wxy/association/Maskflownet_association_1024/dataset/4096after_affine/"
prep_path3="/data/wxy/association/Maskflownet_association_1024/dataset/2048after_affine/"
if not os.path.exists(prep_path3):
    os.mkdir(prep_path3)
def appendimages(im1, im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    return np.concatenate((im1, im2), axis=1)
with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])

        print(num)
        fimg = str(num)+'_1.jpg'
        flmk = str(num)+'_1.csv'
        im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
        im_temp1 = cv2.resize(im_temp1, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        skimage.io.imsave(prep_path3+fimg, im_temp1.astype(np.uint8))
        try:
            lmk1 = pd.read_csv(os.path.join(prep_path1, flmk))
            lmk1 = np.array(lmk1)
            lmk1 = lmk1[:, [2, 1]]/2
        except:
            continue
        else:
            fimg = str(num)+'_2.jpg'
            flmk = str(num)+'_2.csv'
            im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
            im_temp2 = cv2.resize(im_temp2, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
            skimage.io.imsave(prep_path3+fimg, im_temp2.astype(np.uint8))
            lmk2 = pd.read_csv(os.path.join(prep_path1, flmk))
            lmk2 = np.array(lmk2)
            lmk2 = lmk2[:, [2, 1]]/2
            im1=appendimages(im_temp1,im_temp2)
            name = ['X', 'Y']
            lmk1=np.asarray(lmk1)
            lmk1=lmk1[:,[1,0]]
            lmk2=np.asarray(lmk2)
            lmk2=lmk2[:,[1,0]]
            outlmk1 = pd.DataFrame(columns=name, data=lmk1)
            outlmk1.to_csv(prep_path3+str(num)+'_1.csv')
            outlmk2 = pd.DataFrame(columns=name, data=lmk2)
            outlmk2.to_csv(prep_path3+str(num)+'_2.csv')
            