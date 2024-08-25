import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pdb
import sys 
sys.path.append('..') 
import numpy as np
from torch.backends import cudnn
import sys
import pprint
import random
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import time
import cv2
import csv
import skimage.io, skimage.filters
from timeit import default_timer
from skimage import io
import pandas as pd
from numpy import *
import skimage
import scipy.io as scio
import matplotlib.pyplot as plt
from torchsummary import summary
select_num=4
search_range=64
step_range=32
orb = cv2.ORB_create(select_num, scaleFactor=1.5,nlevels=3,edgeThreshold=2,patchSize=4)
image_size=1024
publicpath='ORB_select_32_32patch_scaleFactor1.5_nlevels3_patchsize2_steprange16_selectnum2_1024'
savepath='./kps_ORB64s4_4096_to1024_images/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
savepath2='./kps_ORB64s4_4096_to1024/'
if not os.path.exists(savepath2):
    os.mkdir(savepath2)
data_path = "/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
def LoadANHIR():
    files=os.listdir(data_path)
    for file in files:
        filepath=os.path.join(data_path,file)
        im_temp1 = io.imread(filepath, as_gray=True)
        if im_temp1.max()<1:
            im_temp1=np.uint8(im_temp1*255)
        filtered_coords1s=[]
        for i in range(int(np.floor((image_size-search_range)/step_range))):
            for j in range(int(np.floor((image_size-search_range)/step_range))):
                select_roi=im_temp1[step_range*i:(step_range*i+search_range),step_range*j:(step_range*j+search_range)]
                if select_roi.mean()>10:
                    kp1, _ = orb.detectAndCompute(select_roi, None)
                    if kp1!=():
                        filtered_coords1= addsiftkp([], kp1,i,j)
                        filtered_coords1s.extend(filtered_coords1)
        lmk1=np.array(filtered_coords1s)
        print(lmk1.shape[0])
        if lmk1.shape[0]>0:
            name = ['X', 'Y']
            outlmk1 = pd.DataFrame(columns=name, data=lmk1[:,[1,0]])
            outlmk1.to_csv(savepath2+file.split('.')[0]+'.csv')
            plt.figure()
            plt.imshow(im_temp1)
            # plt.scatter(lmk1[:,1],lmk1[:,0], s=0.05,c=randomcolor())
            plt.scatter(lmk1[:,1],lmk1[:,0], s=0.05,c='#FF0033')
            plt.savefig(savepath+file.split('.')[0]+'_'+str(lmk1.shape[0])+'.jpg', dpi=600)
            plt.close()


def addsiftkp(coords, kp,row,col):
    for i in range(np.shape(kp)[0]):
        siftco = [int(round(kp[i].pt[1]))+step_range*row, int(round(kp[i].pt[0]))+step_range*col]
        if siftco not in coords:
            coords.append(siftco)
    num=np.shape(coords)[0]
    return coords

def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return concatenate((im1, im2), axis=1)
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

if __name__ == '__main__':
    cudnn.benchmark = True 
    LoadANHIR()
