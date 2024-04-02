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
data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"
prep_path1 = os.path.join(data_path, '1024after_affine')
# prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
# prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
# prep_path2="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
prep_path2="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
prep_path2="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
prep_path2="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
prep_path2="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"

prep_path3="/data/wxy/association/Maskflownet_association_1024/training_visualization/unique_kps/lmk_ite4/"
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
        
        if row[5] == 'training':
            # if num <376 or (num>=403 and num<=480):
                # continue
            print(num)
            fimg = str(num)+'_1.jpg'
            flmk = str(num)+'_1.csv'
            im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
            # im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
            # im_temp2[0] = im_temp1
            # im_temp2[1] = im_temp1
            # im_temp2[2] = im_temp1
            try:
                lmk1 = pd.read_csv(os.path.join(prep_path2, flmk))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                continue
            else:
                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                
                lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                # im1=appendimages(im_temp1,im_temp2)
                # for i in range (lmk2.shape[0]):
                    # plt.figure()
                    # plt.imshow(im1)
                    # plt.plot([lmk1[i,1],lmk2[i,1]+1024],[lmk1[i,0],lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig(prep_path3+str(num)+'_'+str(i)+'.jpg',dpi=300)
                    # plt.close()
                im1=appendimages(im_temp1,im_temp2)
                plt.figure()
                plt.imshow(im1)
                for i in range (lmk2.shape[0]):
                    plt.plot([lmk1[i,1],lmk2[i,1]+1024],[lmk1[i,0],lmk2[i,0]], '#FF0033',linewidth=0.5)
                plt.savefig(prep_path3+str(num)+'.jpg',dpi=600)
                plt.close()
                