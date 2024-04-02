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
data_path = "/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
# prep_path2="/ssd1/wxy/TMI_rebuttle_SFG/data/kps_0.85zone0.04mutiscale_imgsize1024/"
# prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC_new/"
# prep_path2="/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.972_0.92_delete_near_ransac/"
# prep_path2="/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_kps/IPCG_ite1_kps_1024_0.97_0.9_delete_near_ransac/"
# prep_path2="/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_kps/IPCG_ite2_kps_1024_0.97_0.9_delete_near_ransac3/"
# prep_path2="/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_kps/IPCG_ite3_kps_1024_0.97_0.9/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_singlescale_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_singlescale_layer2_3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite2_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite3_multiscale_kps1024_0.995_0.75/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite3_multiscale_kps1024_0.995_0.75_delete_near_ransac3/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9/"
prep_path2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
count1s=[]
count2s=[]
files=os.listdir(data_path)
for file in files:
    if file.split('_')[1]=='HE':
        continue
    
    num = file.split('.')[0]
    flmk = str(num)+'_1.csv'
    try:
        lmk1 = pd.read_csv(os.path.join(prep_path2, flmk))
        lmk1 = np.array(lmk1)
        lmk1 = lmk1[:, [2, 1]]
        count1s.append(lmk1.shape[0])
    except:
        count1s.append(0)
        try:
            lmk2 = pd.read_csv(os.path.join(prep_path2, str(num)+'_2.csv'))
            lmk2 = np.array(lmk2)
            lmk2 = lmk2[:, [2, 1]]
            count2s.append(lmk2.shape[0])
        except:
            count2s.append(0)
    else:
        flmk = str(num)+'_2.csv'
        lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
        lmk2 = np.array(lmk2)
        lmk2 = lmk2[:, [2, 1]]
        count2s.append(lmk2.shape[0])
        if lmk2.shape[0]!=lmk1.shape[0]:
            print(num)
print(np.mean(count1s))
print(len(count1s))
print(np.mean(count2s))
print(len(count2s))
# pdb.set_trace()