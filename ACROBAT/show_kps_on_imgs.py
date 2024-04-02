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
prep_name='512'
data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"
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

    return np.concatenate((im1, im2), axis=1)


prep_name1 = prep_name + 'after_affine'
prep_path1 = os.path.join(data_path, prep_name1)
temppath='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.965_0.98_with_network_update/'
count=-1
with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'training':
            count=count+1
            fimg = str(num)+'_1.jpg'
            flmk = str(count*2)+'_1.csv'
            
            im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                
            ##############vgg 
            try:
                lmk = pd.read_csv(os.path.join(temppath, flmk))
                lmk = np.array(lmk)
                lmk1 = lmk[:, [2, 1]]
            except:
                continue

            fimg = str(num) + '_2.jpg'
            flmk = str(count*2) + '_2.csv'
            im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
            ##############vgg 
            lmk = pd.read_csv(os.path.join(temppath, flmk))
            lmk = np.array(lmk)
            lmk2 = lmk[:, [2, 1]]
            # pdb.set_trace()
            im1=appendimages(im_temp1,im_temp2)
            plt.figure()
            plt.imshow(im1)
            for i in range (lmk2.shape[0]):
                plt.plot([lmk1[i,1],lmk2[i,1]+512],[lmk1[i,0],lmk2[i,0]], '#FF0033',linewidth=0.5)
            plt.savefig('/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_0.965_0.98_with_network_updatev_sum/'+str(count*2)+'.jpg',dpi=600)
            plt.close()

    
    
    
