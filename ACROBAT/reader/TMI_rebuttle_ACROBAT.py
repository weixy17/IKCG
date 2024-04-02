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
import time







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


def LoadANHIR_supervised_LFS_SFG(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_name2 = prep_name + '_kp_after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2 = '/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_mask = str(num)+'_1_mask.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                    except:
                        lmk_mask = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_mask = np.pad(lmk_mask,((0, 1000 -len(lmk_mask)), (0, 0)), "constant")
                    dataset[flmk_mask] = lmk_mask
                    groups[group].append((fimg, flmk,flmk_mask))
                    train_groups[group].append((fimg, flmk,flmk_mask))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_mask = str(num)+'_2_mask.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                    except:
                        lmk_mask = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_mask = np.pad(lmk_mask,((0, 1000 -len(lmk_mask)), (0, 0)), "constant")
                    dataset[flmk_mask] = lmk_mask
                    groups[group].append((fimg, flmk,flmk_mask))
                    train_groups[group].append((fimg, flmk,flmk_mask))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
    return dataset, groups, train_groups, val_groups








def LoadANHIR_vgg_vgglarge_multiscale_with_orb(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = '512after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4=os.path.join(data_path,'256after_affine')
    prep_path5=os.path.join(data_path,'1024after_affine')
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*3+1)-3)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk_2 = str(num)+'_2.csv'
                flmk_same_2 = str(num)+'_2_same.csv'
                flmk_5_2_2 = str(num)+'_2_5_2.csv'
                flmk_2_5_2 = str(num)+'_2_2_5.csv'
                flmk_same = str(num)+'_1_same.csv'
                flmk_5_2 = str(num)+'_1_5_2.csv'
                flmk_2_5 = str(num)+'_1_2_5.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                forb=str(num)+'_orb_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path4, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path5, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    # ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = lmk3
                            # except:
                                # lmk = np.zeros((2000, 2), dtype=np.int64)
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")

                    # dataset[flmk] = lmk
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                        
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old5 = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_old5 = np.array(lmk_old5)
                        lmk_old5 = lmk_old5[:, [2, 1]]
                    except:
                        lmk_old5 = np.zeros((1000, 2), dtype=np.int64)
                        
                    else:
                        lmk_old5 = np.pad(lmk_old5,((0, 1000 -len(lmk_old5)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk*2
                    dataset[flmk_old2] = lmk_old2*2
                    dataset[flmk_old5] = lmk_old5*2
                    dataset[flmk_large] = lmk_large*2
                    
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    # list_lmk_obtained=lmk.tolist()
                    # list_lmk1_orb=lmk1_orb.tolist()
                    # delete=[]
                    # for i in range (len(list_lmk_obtained)):
                        # if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            # temp=(grid2+list_lmk_obtained[i]).tolist()
                            # for temp_pair in temp:
                                # if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    # delete.append(temp_pair)
                    # for i in range (len(delete)):
                        # list_lmk1_orb.remove(delete[i])
                    # lmk1_orb=np.asarray(list_lmk1_orb)
                    dataset[forb]=lmk1_orb
                    groups[group].append((fimg, fimg_256,fimg_1024, flmk,flmk_old2,flmk_old5,flmk_large,forb))
                    train_groups[group].append((fimg, fimg_256,fimg_1024, flmk,flmk_old2,flmk_old5,flmk_large,forb))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk_same = str(num)+'_2_same.csv'
                flmk_5_2 = str(num)+'_2_5_2.csv'
                flmk_2_5 = str(num)+'_2_2_5.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                forb=str(num)+'_orb_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path4, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path5, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    
                    # ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = lmk3
                            # except:
                                # lmk = np.zeros((2000, 2), dtype=np.int64)
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")

                    # dataset[flmk] = lmk
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                        
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old5 = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_old5 = np.array(lmk_old5)
                        lmk_old5 = lmk_old5[:, [2, 1]]
                    except:
                        lmk_old5 = np.zeros((1000, 2), dtype=np.int64)
                        
                    else:
                        lmk_old5 = np.pad(lmk_old5,((0, 1000 -len(lmk_old5)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk*2
                    dataset[flmk_old2] = lmk_old2*2
                    dataset[flmk_old5] = lmk_old5*2
                    dataset[flmk_large] = lmk_large*2
                    
                    
                    
                    
                    
                    
                    
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    # list_lmk_obtained=lmk.tolist()
                    # list_lmk1_orb=lmk1_orb.tolist()
                    # delete=[]
                    # for i in range (len(list_lmk_obtained)):
                        # if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            # temp=(grid2+list_lmk_obtained[i]).tolist()
                            # for temp_pair in temp:
                                # if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    # delete.append(temp_pair)
                    # for i in range (len(delete)):
                        # list_lmk1_orb.remove(delete[i])
                    # lmk1_orb=np.asarray(list_lmk1_orb)
                    dataset[forb]=lmk1_orb
                    groups[group].append((fimg, fimg_256,fimg_1024, flmk,flmk_old2,flmk_old5,flmk_large,forb))
                    train_groups[group].append((fimg, fimg_256,fimg_1024, flmk,flmk_old2,flmk_old5,flmk_large,forb))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups



def LoadANHIR_DLFS_SFG_multiscale(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name+'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"#"/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"#'/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*3+1)-3)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                forb=str(num)+'_orb_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                forb=str(num)+'_orb_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_DLFS_SFG_multiscale2(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name+'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*3+1)-3)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                forb=str(num)+'_orb_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path5, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = lmk3
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = lmk4
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = lmk5
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = lmk6
                                        except:
                                            lmk = np.zeros((10, 2), dtype=np.int64)
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                        else:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path5, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.concatenate((lmk, lmk2), 0)
                        except:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                        else:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                    lmk, returned_index=np.unique(lmk, return_index=True,axis=0)
                    lmk=lmk.tolist()
                    lmk=np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")

                    dataset[flmk] = lmk*2
                    
                    groups[group].append((fimg,flmk))
                    train_groups[group].append((fimg,flmk))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                forb=str(num)+'_orb_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path5, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = lmk3
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = lmk4
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = lmk5
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = lmk6
                                        except:
                                            lmk = np.zeros((10, 2), dtype=np.int64)
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                        else:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path5, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.concatenate((lmk, lmk2), 0)
                        except:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                        else:
                            try:
                                lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                lmk3 = np.array(lmk3)
                                lmk3 = lmk3[:, [2, 1]]
                                lmk = np.concatenate((lmk, lmk3), 0)
                            except:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                            else:
                                try:
                                    lmk4 = pd.read_csv(os.path.join(prep_path4, flmk))
                                    lmk4 = np.array(lmk4)
                                    lmk4 = lmk4[:, [2, 1]]
                                    lmk = np.concatenate((lmk, lmk4), 0)
                                except:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk5), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                else:
                                    try:
                                        lmk5 = pd.read_csv(os.path.join(prep_path6, flmk))
                                        lmk5 = np.array(lmk5)
                                        lmk5 = lmk5[:, [2, 1]]
                                        lmk = np.concatenate((lmk, lmk5), 0)
                                    except:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                                    else:
                                        try:
                                            lmk6 = pd.read_csv(os.path.join(prep_path7, flmk))
                                            lmk6 = np.array(lmk6)
                                            lmk6 = lmk6[:, [2, 1]]
                                            lmk = np.concatenate((lmk, lmk6), 0)
                                        except:
                                            pass
                    lmk=lmk[returned_index].tolist()
                    lmk=np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                    dataset[flmk] = lmk*2
                    groups[group].append((fimg,flmk))
                    train_groups[group].append((fimg,flmk))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups


def LoadANHIR_vgg_vgglarge_with_orb(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_rotate16_0.98_0.95_corrected/'
    prep_path3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    # prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    # prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*3+1)-3)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk_before=str(num)+'_1_before.csv'
                flmk_2 = str(num)+'_2.csv'
                flmk_same_2 = str(num)+'_2_same.csv'
                flmk_5_2_2 = str(num)+'_2_5_2.csv'
                flmk_2_5_2 = str(num)+'_2_2_5.csv'
                flmk_same = str(num)+'_1_same.csv'
                flmk_5_2 = str(num)+'_1_5_2.csv'
                flmk_2_5 = str(num)+'_1_2_5.csv'
                flmk_large = str(num)+'_1_large.csv'
                forb=str(num)+'_orb_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    # ##############maskflownet_before_corrected and maskflownet_after_corrected
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # lmk = np.zeros((1000, 2), dtype=np.int64)
                        # else:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    
                    
                    
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    
                    
                    try:
                        lmk_before = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_before = np.array(lmk_before)
                        lmk_before = lmk_before[:, [2, 1]]
                    except:
                        
                        lmk_before = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_before = np.pad(lmk_before,((0, 1000 -len(lmk_before)), (0, 0)), "constant")
                    
                    dataset[flmk] = lmk
                    dataset[flmk_before] = lmk_before
                    
                    
                    
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    # list_lmk_obtained=lmk.tolist()
                    # list_lmk1_orb=lmk1_orb.tolist()
                    # delete=[]
                    # for i in range (len(list_lmk_obtained)):
                        # if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            # temp=(grid2+list_lmk_obtained[i]).tolist()
                            # for temp_pair in temp:
                                # if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    # delete.append(temp_pair)
                    # for i in range (len(delete)):
                        # list_lmk1_orb.remove(delete[i])
                    # lmk1_orb=np.asarray(list_lmk1_orb)
                    dataset[forb]=lmk1_orb
                    groups[group].append((fimg,flmk,flmk_before,forb))
                    train_groups[group].append((fimg,flmk,flmk_before,forb))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk_before=str(num)+'_2_before.csv'
                flmk_same = str(num)+'_2_same.csv'
                flmk_5_2 = str(num)+'_2_5_2.csv'
                flmk_2_5 = str(num)+'_2_2_5.csv'
                flmk_large = str(num)+'_2_large.csv'
                forb=str(num)+'_orb_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    
                    # # ###############maskflownet_before_corrected and maskflownet_after_corrected
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # lmk = np.zeros((1000, 2), dtype=np.int64)
                        # else:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    
                    
                    
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    
                    
                    try:
                        lmk_before = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_before = np.array(lmk_before)
                        lmk_before = lmk_before[:, [2, 1]]
                    except:
                        
                        lmk_before = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_before = np.pad(lmk_before,((0, 1000 -len(lmk_before)), (0, 0)), "constant")
                    
                    dataset[flmk] = lmk
                    dataset[flmk_before] = lmk_before
                    
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    # list_lmk_obtained=lmk.tolist()
                    # list_lmk1_orb=lmk1_orb.tolist()
                    # delete=[]
                    # for i in range (len(list_lmk_obtained)):
                        # if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            # temp=(grid2+list_lmk_obtained[i]).tolist()
                            # for temp_pair in temp:
                                # if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    # delete.append(temp_pair)
                    # for i in range (len(delete)):
                        # list_lmk1_orb.remove(delete[i])
                    # lmk1_orb=np.asarray(list_lmk1_orb)
                    dataset[forb]=lmk1_orb
                    groups[group].append((fimg,flmk,flmk_before,forb))
                    train_groups[group].append((fimg,flmk,flmk_before,forb))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups


def LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_1iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    # prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    # prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    # prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    # prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    # prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] != 'evaluation':
                # if num not in [60,61,62,121,123,129,231,233]:
                if num <338 or num >369:
                    continue
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,forb))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,forb))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_2iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    # prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    # prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    # prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    # prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    # prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [60,61,62,121,123,129,231,233]:
                # if num not in [129,231,233]:
                    # continue
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 2000 -len(lmk_ite1)), (0, 0)), "constant")###############vgg with vgg large
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0),np.unique(lmk_ite1, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    dataset[flmk_ite1] = lmk_ite1
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 2000 -len(lmk_ite1)), (0, 0)), "constant")###############vgg with vgg large
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0),np.unique(lmk_ite1, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    dataset[flmk_ite1] = lmk_ite1
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_3iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    # prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    # prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    # prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    # prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    # prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [60,61,62,121,123,129,231,233]:
                # if num not in [129,231,233]:
                    # continue
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 2000 -len(lmk_ite1)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 2000 -len(lmk_ite2)), (0, 0)), "constant")###############vgg with vgg large
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0),np.unique(lmk_ite1, return_index=False,axis=0),np.unique(lmk_ite2, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    dataset[flmk_ite1] = lmk_ite1
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                    except:
                        lmk2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk2 = np.pad(lmk2,((0, 2000 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 2000 -len(lmk_ite1)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 2000 -len(lmk_ite2)), (0, 0)), "constant")###############vgg with vgg large
                    # pdb.set_trace()
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0),np.unique(lmk_ite1, return_index=False,axis=0),np.unique(lmk_ite2, return_index=False,axis=0)), 0)
                        
                    lmk_obtained_grid=np.reshape(np.expand_dims(int32(lmk_obtained),0)+np.expand_dims(grid2,1),(-1,2))
                    lmk_list=int32(lmk).tolist()
                    lmk_obtained_grid_list=lmk_obtained_grid.tolist()
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk1_orb=lmk1_orb.tolist()
                    lmk_orb_valid=[]
                    for i in range (len(list_lmk1_orb)):
                        if (list_lmk1_orb[i] not in lmk_obtained_grid_list):
                            lmk_orb_valid.append(list_lmk1_orb[i])
                    lmk_orb_valid=np.asarray(lmk_orb_valid)
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    
                    dataset[forb]=lmk_orb_valid
                    dataset[flmk] = lmk
                    dataset[flmk2] = lmk2
                    dataset[flmk_ite1] = lmk_ite1
                    groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                    train_groups[group].append((fimg,fimg_256,fimg_1024, flmk,flmk2,flmk_ite1,forb))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups

def LoadANHIR_recursive_DLFSFG_multiscale_training_1iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_large_1024 = str(num)+'_1_large_1024.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2

                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_large_1024 = str(num)+'_2_large_1024.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_training_2iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
    prep_path11="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_large_1024 = str(num)+'_1_large_1024.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2

                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_large_1024 = str(num)+'_2_large_1024.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
    prep_path11="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path12="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path13="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
            # if row[5] != 'evaluation' and row[5] != 'training':
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_large_1024 = str(num)+'_1_large_1024.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_ite4 = str(num)+'_1_ite4.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img1_imshow=im_temp1
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 200 -len(lmk_gt)), (0, 0)), "constant")
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    
                    
                    # lmk_obtained=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    # lmk_obtained,indexes=np.unique(lmk_obtained, return_index=True,axis=0)
                    # lmk_obtained1_imshow=lmk_obtained
                    # lmk_obtained = np.pad(lmk_obtained,((0, 4000 -len(lmk_obtained)), (0, 0)), "constant")
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk#############################lmk_obtained#####################
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    dataset[flmk_ite3] = lmk_ite3
                    dataset[flmk_ite4] = lmk_ite4
                    dataset[flmk_gt] = lmk_gt
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_large_1024 = str(num)+'_2_large_1024.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_ite3 = str(num)+'_2_ite3.csv'
                flmk_ite4 = str(num)+'_2_ite4.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img2_imshow=im_temp1
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 200 -len(lmk_gt)), (0, 0)), "constant")
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    
                    # lmk_obtained=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    # lmk_obtained=lmk_obtained[indexes,:]
                    
                    
                    # im1=appendimages(img1_imshow,img2_imshow)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (lmk_obtained.shape[0]):
                        # plt.plot([lmk_obtained1_imshow[i,1],lmk_obtained[i,1]+1024],[lmk_obtained1_imshow[i,0],lmk_obtained[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig("/data/wxy/association/Maskflownet_association_1024/training_visualization/unique_kps/unique_kps_for_train/"+str(num)+'.jpg',dpi=600)
                    # plt.close()
                    
                    # lmk_obtained = np.pad(lmk_obtained,((0, 4000 -len(lmk_obtained)), (0, 0)), "constant")
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk###################################lmk_obtained##########################
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    dataset[flmk_ite3] = lmk_ite3
                    dataset[flmk_ite4] = lmk_ite4
                    dataset[flmk_gt] = lmk_gt
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))

            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration_concatenate(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
    prep_path11="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path12="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path13="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [57,59,64,65,118,119,127,130,132]:
                # if num not in [60,61,62,121,123,129,231]:
                # if num not in [285]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_large_1024 = str(num)+'_1_large_1024.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_ite4 = str(num)+'_1_ite4.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img1_imshow=im_temp1

                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                        # temp_lmk1=lmk_old2
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                        # temp_lmk1=lmk_ite1
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                        # temp_lmk1=lmk_ite2
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                        # temp_lmk1=lmk_ite3
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                        # temp_lmk1=lmk_ite4
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    try:
                        lmk_hand = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_hand = np.array(lmk_hand)
                        lmk_hand = lmk_hand[:, [2, 1]]
                    except:
                        lmk_hand = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_hand = np.pad(lmk_hand,((0, 200 -len(lmk_hand)), (0, 0)), "constant")
                    lmk_LFS1=lmk_LFS1*2
                    lmk_LFS2=lmk_LFS2*2
                    lmk_512=lmk_512*2
                    lmk_512_2=lmk_512_2*2
                    lmk_large=lmk_large*2
                    # lmk_obtained1024=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    lmk_obtained1024=np.concatenate((lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    lmk_obtained1024,indexes=np.unique(lmk_obtained1024, return_index=True,axis=0)
                    
                    
                    lmk_hand_grid=np.reshape(np.expand_dims(int32(lmk_hand),0)+np.expand_dims(grid2,1),(-1,2)).tolist()
                    lmk_obtained1024_list=int32(lmk_obtained1024).tolist()
                    
                    lmk_obtained1024 = np.pad(lmk_obtained1024,((0, 3000 -len(lmk_obtained1024)), (0, 0)), "constant")
                    
                    lmk_valid=[]
                    valid_index=[]
                    for i in range (len(lmk_obtained1024_list)):
                        if (lmk_obtained1024_list[i] in lmk_hand_grid):
                            lmk_valid.append(lmk_obtained1024_list[i])
                            valid_index.append(i)
                    lmk_valid=np.asarray(lmk_valid)
                    lmk_obtained1024_valid = np.pad(lmk_valid,((0, 1000 -len(lmk_valid)), (0, 0)), "constant")
                    
                    
                    
                    
                    lmk_obtained512=np.concatenate((lmk_LFS1,lmk_LFS2,lmk_512,lmk_512_2,lmk_large),0)
                    # lmk_obtained512=np.concatenate((lmk_LFS1,lmk_LFS2,lmk_512,lmk_512_2),0)
                    lmk_obtained512,indexes2=np.unique(lmk_obtained512, return_index=True,axis=0)

                    lmk_obtained512_list=int32(lmk_obtained512).tolist()
                    lmk_valid512=[]
                    valid_index512=[]
                    for i in range (len(lmk_obtained512_list)):
                        if (lmk_obtained512_list[i] in lmk_hand_grid):
                            lmk_valid512.append(lmk_obtained512_list[i])
                            valid_index512.append(i)
                    lmk_valid512=np.asarray(lmk_valid512)
                    lmk_obtained512_valid = np.pad(lmk_valid512,((0, 3000 -len(lmk_valid512)), (0, 0)), "constant")
                    
                    
                    
                    
                    
                    lmk_obtained_large=np.concatenate((lmk_large,lmk_large_1024),0)
                    lmk_obtained_large,indexes_large=np.unique(lmk_obtained_large, return_index=True,axis=0)
                    lmk_obtained_large = np.pad(lmk_obtained_large,((0, 30 -len(lmk_obtained_large)), (0, 0)), "constant")

                    
                    dataset[flmk_LFS1] = lmk_obtained1024_valid#lmk_obtained1024
                    dataset[flmk_LFS2] = lmk_obtained512_valid#lmk_obtained512
                    dataset[flmk] = lmk_hand#############################lmk_obtained#####################
                    dataset[flmk_large] = lmk_obtained_large#############################lmk_obtained#####################
                    
                    #lmk_obtained1024
                    # temp2_lmk1=lmk_obtained512
                    
                    
                    # groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk,flmk_large))
                    # train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk,flmk_large))
                    # groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk_large))
                    # train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk_large))
                    groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk))
                    train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_large_1024 = str(num)+'_2_large_1024.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_ite3 = str(num)+'_2_ite3.csv'
                flmk_ite4 = str(num)+'_2_ite4.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img2_imshow=im_temp1
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        # temp_lmk2=lmk
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                        # temp_lmk2=lmk_old2
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                        # temp_lmk2=lmk_ite1
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                        # temp_lmk2=lmk_ite2
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                        # temp_lmk2=lmk_ite3
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                        # temp_lmk2=lmk_ite4
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    try:
                        lmk_hand = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_hand = np.array(lmk_hand)
                        lmk_hand = lmk_hand[:, [2, 1]]
                    except:
                        lmk_hand = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_hand = np.pad(lmk_hand,((0, 200 -len(lmk_hand)), (0, 0)), "constant")
                    lmk_LFS1=lmk_LFS1*2
                    lmk_LFS2=lmk_LFS2*2
                    lmk_512=lmk_512*2
                    lmk_512_2=lmk_512_2*2
                    lmk_large=lmk_large*2
                    # lmk_obtained1024=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    lmk_obtained1024=np.concatenate((lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    lmk_obtained1024=lmk_obtained1024[indexes,:]
                    lmk_obtained1024valid=lmk_obtained1024[valid_index,:]
                    lmk_obtained1024 = np.pad(lmk_obtained1024,((0, 3000 -len(lmk_obtained1024)), (0, 0)), "constant")
                    lmk_obtained1024valid = np.pad(lmk_obtained1024valid,((0, 1000 -len(lmk_obtained1024valid)), (0, 0)), "constant")
                    lmk_obtained512=np.concatenate((lmk_LFS1,lmk_LFS2,lmk_512,lmk_512_2,lmk_large),0)
                    # lmk_obtained512=np.concatenate((lmk_LFS1,lmk_LFS2,lmk_512,lmk_512_2),0)
                    lmk_obtained512=lmk_obtained512[indexes2,:]
                    lmk_obtained512valid=lmk_obtained512[valid_index512,:]
                    # lmk_obtained512 = np.pad(lmk_obtained512,((0, 3000 -len(lmk_obtained512)), (0, 0)), "constant")
                    lmk_obtained512valid = np.pad(lmk_obtained512valid,((0, 3000 -len(lmk_obtained512valid)), (0, 0)), "constant")
                    lmk_obtained_large=np.concatenate((lmk_large,lmk_large_1024),0)
                    lmk_obtained_large=lmk_obtained_large[indexes_large,:]
                    lmk_obtained_large = np.pad(lmk_obtained_large,((0, 30 -len(lmk_obtained_large)), (0, 0)), "constant")
                    dataset[flmk_LFS1] = lmk_obtained1024valid#lmk_obtained1024#
                    dataset[flmk_LFS2] = lmk_obtained512valid#lmk_obtained512
                    dataset[flmk] = lmk_hand#############################lmk_obtained#####################
                    dataset[flmk_large] = lmk_obtained_large#############################lmk_obtained#####################
                    
                    # #lmk_obtained1024
                    # temp2_lmk2=lmk_obtained512
                    
                    
                    # groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk,flmk_large))
                    # train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk,flmk_large))
                    # groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk_large))
                    # train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk_large))
                    groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk))
                    train_groups[group].append((fimg,flmk_LFS1,flmk_LFS2,flmk))
                    
                    # if temp_lmk2.shape[0]!=temp_lmk1.shape[0]:
                        # print('{} 1024 errors,{}:{}'.format(num,temp_lmk1.shape[0],temp_lmk2.shape[0]))
                    # if temp2_lmk2.shape[0]!=temp2_lmk1.shape[0]:
                        # print('{} 512 errors,{}:{}'.format(num,temp2_lmk1.shape[0],temp2_lmk2.shape[0]))
                    # pdb.set_trace()
                    # im1=appendimages(img1_imshow,img2_imshow)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (temp_lmk1.shape[0]):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+1024],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Maskflownet_association_1024/training_visualization/all_kps/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (temp2_lmk1.shape[0]):
                        # plt.plot([temp2_lmk1[i,1],temp2_lmk2[i,1]+1024],[temp2_lmk1[i,0],temp2_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Maskflownet_association_1024/training_visualization/all_kps/'+str(num)+'_2.jpg',dpi=600)
                    # plt.close()
                    # # pdb.set_trace()
                    
                    
                    
                    
                    
                    
                    # im1=appendimages(img1_imshow,img2_imshow)
                    # try:
                        # for i in range (temp_lmk1.shape[0]):
                            # plt.figure()
                            # plt.imshow(im1)
                            # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+1024],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                            # plt.savefig('/data/wxy/association/Maskflownet_association_1024/training_visualization/unique_kps/lmk_ite3_single_pair/'+str(num)+'_'+str(i)+'.jpg',dpi=300)
                            # plt.close()
                        # plt.figure()
                        # plt.imshow(im1)
                        # for i in range (temp_lmk1.shape[0]):
                            # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+1024],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                        # plt.savefig('/data/wxy/association/Maskflownet_association_1024/training_visualization/unique_kps/lmk_ite3/'+str(num)+'.jpg',dpi=600)
                        # plt.close()
                    # except:
                        # pass
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    # pdb.set_trace()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration_refine_for_evaluation(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):
    # classes=['COAD_01','COAD_02','COAD_03','COAD_04','COAD_05','COAD_06','COAD_07','COAD_08','COAD_09','COAD_10',
              # 'COAD_11','COAD_12','COAD_13','COAD_14','COAD_15','COAD_16','COAD_17','COAD_18','COAD_19','COAD_20',
              # 'breast_1','breast_2','breast_3','breast_4','breast_5','gastric_1','gastric_2','gastric_3','gastric_4',
              # 'gastric_5','gastric_6','gastric_7','gastric_8','gastric_9','kidney_1','kidney_2','kidney_3','kidney_4'
              # ,'kidney_5','lung-lesion_2','lung-lesion_1','lung-lesion_3','lung-lobes_1','lung-lobes_2','lung-lobes_3'
              # ,'lung-lobes_4','mammary-gland_1','mammary-gland_2','mice-kidney_1']
    classes=['COAD_01','COAD_02','COAD_03','COAD_04','COAD_05','COAD_06','COAD_07','COAD_08','COAD_09','COAD_10',
              'COAD_11','COAD_12','COAD_13','COAD_14','COAD_15','COAD_16','COAD_17','COAD_18','COAD_19','COAD_20',
              'breast_1','breast_2','breast_3','breast_4','breast_5','gastric_1','gastric_2','gastric_3','gastric_4',
              'gastric_5','gastric_6','gastric_7','gastric_8','gastric_9','kidney_1','kidney_2','kidney_3','kidney_4'
              ,'kidney_5','mice-kidney_1']
    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2=os.path.join(data_path,'512after_affine')
    prep_path1_3=os.path.join(data_path,'2048after_affine')
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_20_0.25_30_0_40_0.5_50_0_60_0.25_rotate8_0.985_0.028_0.004_0.81_01norm_same_pixels_multiscale_1024/"
    prep_path4="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
    prep_path5="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path7="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
    prep_path8='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path9="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path10="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
    prep_path11="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path12="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
    prep_path13="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
            
            # if row[5] != 'evaluation' and row[5] != 'training':
                # if num >5:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk2 = str(num)+'_1_2.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_512 = str(num)+'_1_512.csv'
                flmk_512_2 = str(num)+'_1_512_2.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_large_1024 = str(num)+'_1_large_1024.csv'
                flmk_LFS1 = str(num)+'_1_LFS1.csv'
                flmk_LFS2 = str(num)+'_1_LFS2.csv'
                flmk_ite1 = str(num)+'_1_ite1.csv'
                flmk_ite2 = str(num)+'_1_ite2.csv'
                flmk_ite3 = str(num)+'_1_ite3.csv'
                flmk_ite4 = str(num)+'_1_ite4.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                forb = str(num)+'_1_orb.csv'
                # flmk_hand=str(num)+'_1_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img1_imshow=im_temp1
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 200 -len(lmk_gt)), (0, 0)), "constant")
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    
                    
                    # lmk_obtained=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    # lmk_obtained,indexes=np.unique(lmk_obtained, return_index=True,axis=0)
                    # lmk_obtained1_imshow=lmk_obtained
                    # lmk_obtained = np.pad(lmk_obtained,((0, 4000 -len(lmk_obtained)), (0, 0)), "constant")
                    
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk#############################lmk_obtained#####################
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    dataset[flmk_ite3] = lmk_ite3
                    dataset[flmk_ite4] = lmk_ite4
                    dataset[flmk_gt] = lmk_gt
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_512 = str(num)+'_2_512.csv'
                flmk_512_2 = str(num)+'_2_512_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_large_1024 = str(num)+'_2_large_1024.csv'
                flmk_LFS1 = str(num)+'_2_LFS1.csv'
                flmk_LFS2 = str(num)+'_2_LFS2.csv'
                flmk_ite1 = str(num)+'_2_ite1.csv'
                flmk_ite2 = str(num)+'_2_ite2.csv'
                flmk_ite3 = str(num)+'_2_ite3.csv'
                flmk_ite4 = str(num)+'_2_ite4.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                forb = str(num)+'_2_orb.csv'
                flmk_hand=str(num)+'_2_hand.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    img2_imshow=im_temp1
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 200 -len(lmk_gt)), (0, 0)), "constant")
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((2000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_old2 = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_old2 = np.array(lmk_old2)
                        lmk_old2 = lmk_old2[:, [2, 1]]
                    except:
                        lmk_old2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_old2 = np.pad(lmk_old2,((0, 1000 -len(lmk_old2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_512 = pd.read_csv(os.path.join(prep_path7, flmk))
                        lmk_512 = np.array(lmk_512)
                        lmk_512 = lmk_512[:, [2, 1]]
                    except:
                        lmk_512 = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_512 = np.pad(lmk_512,((0, 1200 -len(lmk_512)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_512_2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_512_2 = np.array(lmk_512_2)
                        lmk_512_2 = lmk_512_2[:, [2, 1]]
                    except:
                        lmk_512_2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_512_2 = np.pad(lmk_512_2,((0, 1000 -len(lmk_512_2)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFS1 = pd.read_csv(os.path.join(prep_path4, flmk))
                        lmk_LFS1 = np.array(lmk_LFS1)
                        lmk_LFS1 = lmk_LFS1[:, [2, 1]]
                    except:
                        lmk_LFS1 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS1 = np.pad(lmk_LFS1,((0, 1000 -len(lmk_LFS1)), (0, 0)), "constant")
                    try:
                        lmk_LFS2 = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_LFS2 = np.array(lmk_LFS2)
                        lmk_LFS2 = lmk_LFS2[:, [2, 1]]
                    except:
                        lmk_LFS2 = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_LFS2 = np.pad(lmk_LFS2,((0, 1000 -len(lmk_LFS2)), (0, 0)), "constant")
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path8, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 10 -len(lmk_large)), (0, 0)), "constant")
                    try:
                        lmk_large_1024 = pd.read_csv(os.path.join(prep_path10, flmk))
                        lmk_large_1024 = np.array(lmk_large_1024)
                        lmk_large_1024 = lmk_large_1024[:, [2, 1]]
                    except:
                        lmk_large_1024 = np.zeros((10, 2), dtype=np.int64)
                    else:
                        lmk_large_1024 = np.pad(lmk_large_1024,((0, 10 -len(lmk_large_1024)), (0, 0)), "constant")
                    try:
                        lmk_ite1 = pd.read_csv(os.path.join(prep_path9, flmk))
                        lmk_ite1 = np.array(lmk_ite1)
                        lmk_ite1 = lmk_ite1[:, [2, 1]]
                    except:
                        lmk_ite1 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite1 = np.pad(lmk_ite1,((0, 1500 -len(lmk_ite1)), (0, 0)), "constant")
                    try:
                        lmk_ite2 = pd.read_csv(os.path.join(prep_path11, flmk))
                        lmk_ite2 = np.array(lmk_ite2)
                        lmk_ite2 = lmk_ite2[:, [2, 1]]
                    except:
                        lmk_ite2 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite2 = np.pad(lmk_ite2,((0, 1500 -len(lmk_ite2)), (0, 0)), "constant")
                    try:
                        lmk_ite3 = pd.read_csv(os.path.join(prep_path12, flmk))
                        lmk_ite3 = np.array(lmk_ite3)
                        lmk_ite3 = lmk_ite3[:, [2, 1]]
                    except:
                        lmk_ite3 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite3 = np.pad(lmk_ite3,((0, 1500 -len(lmk_ite3)), (0, 0)), "constant")
                    try:
                        lmk_ite4 = pd.read_csv(os.path.join(prep_path13, flmk))
                        lmk_ite4 = np.array(lmk_ite4)
                        lmk_ite4 = lmk_ite4[:, [2, 1]]
                    except:
                        lmk_ite4 = np.zeros((1500, 2), dtype=np.int64)
                    else:
                        lmk_ite4 = np.pad(lmk_ite4,((0, 1500 -len(lmk_ite4)), (0, 0)), "constant")
                    
                    # lmk_obtained=np.concatenate((lmk,lmk_old2,lmk_large_1024,lmk_ite1,lmk_ite2,lmk_ite3,lmk_ite4),0)
                    # lmk_obtained=lmk_obtained[indexes,:]
                    
                    
                    # im1=appendimages(img1_imshow,img2_imshow)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (lmk_obtained.shape[0]):
                        # plt.plot([lmk_obtained1_imshow[i,1],lmk_obtained[i,1]+1024],[lmk_obtained1_imshow[i,0],lmk_obtained[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig("/data/wxy/association/Maskflownet_association_1024/training_visualization/unique_kps/unique_kps_for_train/"+str(num)+'.jpg',dpi=600)
                    # plt.close()
                    
                    # lmk_obtained = np.pad(lmk_obtained,((0, 4000 -len(lmk_obtained)), (0, 0)), "constant")
                    dataset[flmk_LFS1] = lmk_LFS1*2
                    dataset[flmk_LFS2] = lmk_LFS2*2
                    dataset[flmk] = lmk###################################lmk_obtained##########################
                    dataset[flmk_old2] = lmk_old2
                    dataset[flmk_512] = lmk_512*2
                    dataset[flmk_512_2] = lmk_512_2*2
                    dataset[flmk_large] = lmk_large*2
                    dataset[flmk_large_1024] = lmk_large_1024
                    dataset[flmk_ite1] = lmk_ite1
                    dataset[flmk_ite2] = lmk_ite2
                    dataset[flmk_ite3] = lmk_ite3
                    dataset[flmk_ite4] = lmk_ite4
                    dataset[flmk_gt] = lmk_gt
                    
                    groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))
                    train_groups[group].append((fimg,flmk,flmk_old2,flmk_512,flmk_512_2,flmk_LFS1,flmk_LFS2,flmk_large,flmk_large_1024,flmk_ite1,flmk_ite2,flmk_ite3,flmk_ite4,flmk_gt))

            elif row[5] == 'evaluation':
                # if (row[1].split('_')[0]+'_'+row[1].split('_')[1])!=classes[0]:
                    # continue
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
def LoadANHIR_recursive_vgg_vgglarge_maskmorethanvgg_masksamewithvgg_multisift(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    siftdir="/data/wxy/association/Maskflownet_association/dense/multi_siftflowmap_512/"
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*3+1)-3)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                forb=str(num)+'orb_1.csv'
                fmask_same_with_vgg = str(num)+'_1_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_1_mask_more_than_vgg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large

                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask_list=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask_list.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask_list.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    # pdb.set_trace()
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    
                    
                    if len(lmk_mask)>0:
                        lmk_obtained=np.concatenate((lmk,lmk_mask),0)
                    else:
                        lmk_obtained=lmk
                    
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk_obtained=lmk_obtained.tolist()
                    list_lmk1_orb=lmk1_orb.tolist()
                    delete=[]
                    for i in range (len(list_lmk_obtained)):
                        if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            temp=(grid2+list_lmk_obtained[i]).tolist()
                            # pdb.set_trace()
                            for temp_pair in temp:
                                if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    delete.append(temp_pair)
                    for i in range (len(delete)):
                        list_lmk1_orb.remove(delete[i])
                    lmk1_orb=np.asarray(list_lmk1_orb)
                    # lmk1_orb = np.pad(lmk1_orb,((0, 10000 -len(lmk1_orb)), (0, 0)), "constant")
                    dataset[forb]=lmk1_orb
                    
                    
                    
                    
                    
                    
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,forb))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,forb))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                fmask_same_with_vgg = str(num)+'_2_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_2_mask_more_than_vgg.csv'
                forb=str(num)+'orb_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask_list=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask_list.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask_list.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                            
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    
                    if len(lmk_mask)>0:
                        lmk_obtained=np.concatenate((lmk,lmk_mask),0)
                    else:
                        lmk_obtained=lmk
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    list_lmk_obtained=lmk_obtained.tolist()
                    list_lmk1_orb=lmk1_orb.tolist()
                    delete=[]
                    for i in range (len(list_lmk_obtained)):
                        if (list_lmk_obtained[i] in list_lmk1_orb) and (list_lmk_obtained[i] not in delete):
                            temp=(grid2+list_lmk_obtained[i]).tolist()
                            # pdb.set_trace()
                            for temp_pair in temp:
                                if temp_pair in list_lmk1_orb and temp_pair not in delete:
                                    delete.append(temp_pair)
                    for i in range (len(delete)):
                        list_lmk1_orb.remove(delete[i])
                    lmk1_orb=np.asarray(list_lmk1_orb)
                    # lmk1_orb = np.pad(lmk1_orb,((0, 10000 -len(lmk1_orb)), (0, 0)), "constant")
                    dataset[forb]=lmk1_orb
                    
                    
                    
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,forb))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,forb))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups














def LoadANHIR_supervised_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                fmask_same_with_vgg = str(num)+'_1_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_1_mask_more_than_vgg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 1000 -len(lmk_gt)), (0, 0)), "constant")
                    dataset[flmk_gt] = lmk_gt
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large

                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    # pdb.set_trace()
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,flmk_gt))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,flmk_gt))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                fmask_same_with_vgg = str(num)+'_2_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_2_mask_more_than_vgg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 1000 -len(lmk_gt)), (0, 0)), "constant")
                    dataset[flmk_gt] = lmk_gt
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                            
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,flmk_gt))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,flmk_gt))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups






















def LoadANHIR_supervised_vgg_vgglarge(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 1000 -len(lmk_gt)), (0, 0)), "constant")
                    dataset[flmk_gt] = lmk_gt
                    
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk,flmk_gt))
                    train_groups[group].append((fimg, flmk,flmk_gt))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk
                    try:
                        lmk_gt = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk_gt = np.array(lmk_gt)
                        lmk_gt = lmk_gt[:, [2, 1]]
                    except:
                        lmk_gt = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_gt = np.pad(lmk_gt,((0, 1000 -len(lmk_gt)), (0, 0)), "constant")
                    dataset[flmk_gt] = lmk_gt
                    groups[group].append((fimg, flmk,flmk_gt))
                    train_groups[group].append((fimg, flmk,flmk_gt))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups









def LoadANHIR_vgg_vgglarge(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_gt = str(num)+'_1_gt.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    train_groups[group].append((fimg, flmk))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_gt = str(num)+'_2_gt.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk

                    groups[group].append((fimg, flmk))
                    train_groups[group].append((fimg, flmk))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups



def LoadANHIR_LFS_kps_multiscale(prep_name, subsets = [""], data_path = r"/ssd1/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2 = os.path.join(data_path, '512after_affine')
    prep_path1_3 = os.path.join(data_path, '2048after_affine')
    prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC_new/"
    prep_path2_large="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_large_displacement/"
    orbpath='/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'evaluation':
                if num not in [118,127,132]:
                    continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg2 = str(num)+'_1_512.jpg'
                fimg3 = str(num)+'_1_2048.jpg'
                flmk = str(num)+'_1.csv'
                flmk_large = str(num)+'_large.csv'
                flmk_orb = str(num)+'_orb.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg2] = img2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg3] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk_large, return_index=False,axis=0)), 0)
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained.transpose((1,0))))
                    lmk_orb_valid=lmk1_orb[np.where(dis_temp.min(1)>5)[0],:]
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    dataset[flmk_orb] = lmk_orb_valid
                    groups[group].append((fimg, fimg2, fimg3, flmk,flmk_large,flmk_orb))
                    train_groups[group].append((fimg, fimg2, fimg3, flmk,flmk_large,flmk_orb))

                fimg = str(num)+'_2.jpg'
                fimg2 = str(num)+'_2_512.jpg'
                fimg3 = str(num)+'_2_2048.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_large = str(num)+'_large.csv'
                flmk_orb = str(num)+'_orb.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
                    dataset[fimg] = img2
                    im_temp1 = io.imread(os.path.join(prep_path1_2, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg2] = img2
                    im_temp1 = io.imread(os.path.join(prep_path1_3, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg3] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    lmk_obtained=np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk_large, return_index=False,axis=0)), 0)
                    lmk1_orb = pd.read_csv(orbpath+flmk)
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
                    dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained.transpose((1,0))))
                    lmk_orb_valid=lmk1_orb[np.where(dis_temp.min(1)>5)[0],:]
                    lmk_orb_valid, returned_index=np.unique(lmk_orb_valid, return_index=True,axis=0)
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    dataset[flmk_orb] = lmk_orb_valid
                    groups[group].append((fimg, fimg2, fimg3, flmk,flmk_large,flmk_orb))
                    train_groups[group].append((fimg, fimg2, fimg3, flmk,flmk_large,flmk_orb))
                    
                   
            else:
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img1
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[6:16]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[16:26]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
    # pdb.set_trace()
    return dataset, groups, train_groups, val_groups






def LoadANHIR_LFS_multiscale(prep_name, subsets = [""], data_path = r"/ssd1/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path1_2 = os.path.join(data_path, '512after_affine')
    prep_path1_3 = os.path.join(data_path, '2048after_affine')
    prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC_new/"
    prep_path2_large="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_large_displacement/"
    prep_path3="/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.972_0.92_delete_near_ransac/"
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*5+1)-5)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'evaluation':
                # if num not in [118,127,132]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_large = str(num)+'_1_large.csv'
                flmk_orb = str(num)+'_1_orb.csv'
                flmk_LFG = str(num)+'_1_LFG.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    
                    try:
                        lmk_LFG = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_LFG = np.array(lmk_LFG)
                        lmk_LFG = lmk_LFG[:, [2, 1]]
                    except:
                        lmk_LFG = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_LFG = np.pad(lmk_LFG,((0, 1200 -len(lmk_LFG)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    dataset[flmk_LFG] = lmk_LFG
                    groups[group].append((fimg, flmk,flmk_large,flmk_LFG))
                    train_groups[group].append((fimg, flmk,flmk_large,flmk_LFG))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk2 = str(num)+'_2_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                flmk_orb = str(num)+'_2_orb.csv'
                flmk_LFG = str(num)+'_2_LFG.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
                    dataset[fimg] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_LFG = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_LFG = np.array(lmk_LFG)
                        lmk_LFG = lmk_LFG[:, [2, 1]]
                    except:
                        lmk_LFG = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk_LFG = np.pad(lmk_LFG,((0, 1200 -len(lmk_LFG)), (0, 0)), "constant")###############vgg with vgg large
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    dataset[flmk_LFG] = lmk_LFG
                    groups[group].append((fimg, flmk,flmk_large,flmk_LFG))
                    train_groups[group].append((fimg, flmk,flmk_large,flmk_LFG))
                    
                   
            else:
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img1
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[6:16]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[16:26]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
    # pdb.set_trace()
    return dataset, groups, train_groups, val_groups








def LoadANHIR_vgg_vgglarge_multiscale_new(prep_name, subsets = [""], data_path = r"/ssd1/wxy/association/Maskflownet_association_1024/dataset/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    # prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC/"
    prep_path2_large="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_large_displacement/"
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    
    with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_large = str(num)+'_1_large.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    lmk1=lmk
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    groups[group].append((fimg, flmk,flmk_large))
                    train_groups[group].append((fimg, flmk,flmk_large))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_large = str(num)+'_2_large.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
                    dataset[fimg] = img2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1200, 2), dtype=np.int64)
                        
                    else:
                        lmk = np.pad(lmk,((0, 1200 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path2_large, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large = np.zeros((20, 2), dtype=np.int64)
                    else:
                        lmk_large = np.pad(lmk_large,((0, 20 -len(lmk_large)), (0, 0)), "constant")###############vgg with vgg large
                    lmk2=lmk
                    dataset[flmk] = lmk
                    dataset[flmk_large] = lmk_large
                    groups[group].append((fimg, flmk,flmk_large))
                    train_groups[group].append((fimg, flmk,flmk_large))
                # im1=appendimages(im_temp1,im_temp2)
                # plt.figure()
                # plt.imshow(im1)
                # for i in range (lmk2.shape[0]):
                    # plt.plot([lmk1[i,1],lmk2[i,1]+1024],[lmk1[i,0],lmk2[i,0]], '#FF0033',linewidth=0.5)
                # plt.savefig("/ssd1/wxy/association/Maskflownet_association_1024/test/"+str(num)+'.jpg',dpi=600)
                # plt.close()
            # if row[5] == 'training':
            else:
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img1
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[6:16]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[16:26]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
    # pdb.set_trace()
    return dataset, groups, train_groups, val_groups





def LoadANHIR_rebuttle_baseline(prep_name, subsets = [""], data_path = r"/ssd1/wxy/association/Maskflownet_association_1024/dataset/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
    prep_path3='/ssd1/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4="/ssd1/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path5="/ssd1/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    
    with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_old2 = str(num)+'_1_old2.csv'
                flmk_old5 = str(num)+'_1_old5.csv'
                flmk_large = str(num)+'_1_large.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    groups[group].append((fimg))
                    train_groups[group].append((fimg))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_old2 = str(num)+'_2_old2.csv'
                flmk_old5 = str(num)+'_2_old5.csv'
                flmk_large = str(num)+'_2_large.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    groups[group].append((fimg))
                    train_groups[group].append((fimg))
                    
            if row[5] == 'training':
            # else:
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img1
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[6:16]),0)),1)####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    # lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[16:26]),0)),1)####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    # lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
    return dataset, groups, train_groups, val_groups



def LoadANHIR_vgg_vgglarge_multiscale(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4=os.path.join(data_path,'256after_affine')
    prep_path5=os.path.join(data_path,'1024after_affine')
    prep_path6="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
    
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                fimg_256 = str(num)+'_1_256.jpg'
                fimg_1024 = str(num)+'_1_1024.jpg'
                flmk = str(num)+'_1.csv'
                flmk_2 = str(num)+'_2.csv'
                flmk_same_2 = str(num)+'_2_same.csv'
                flmk_5_2_2 = str(num)+'_2_5_2.csv'
                flmk_2_5_2 = str(num)+'_2_2_5.csv'
                flmk_same = str(num)+'_1_same.csv'
                flmk_5_2 = str(num)+'_1_5_2.csv'
                flmk_2_5 = str(num)+'_1_2_5.csv'
                flmk_large = str(num)+'_1_large.csv'
                
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path4, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path5, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    # ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = lmk3
                            # except:
                                # lmk = np.zeros((2000, 2), dtype=np.int64)
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")

                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large=np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_large=np.pad(lmk_large,((0, 1000 -len(lmk_large)), (0, 0)), "constant") 
                    lmk_same_index=[]
                    try:
                        lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk2 = np.array(lmk2)
                        lmk2 = lmk2[:, [2, 1]]
                        lmk2_2= pd.read_csv(os.path.join(prep_path6, flmk_2))
                        lmk2_2 = np.array(lmk2_2)
                        lmk2_2 = lmk2_2[:, [2, 1]]
                    except:
                        try:
                            lmk5 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk5 = np.array(lmk5)
                            lmk5 = lmk5[:, [2, 1]]
                            lmk5_2 = pd.read_csv(os.path.join(prep_path2, flmk_2))
                            lmk5_2 = np.array(lmk5_2)
                            lmk5_2 = lmk5_2[:, [2, 1]]
                        except:
                            lmk_same=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_2_5=np.zeros((1000, 2), dtype=np.int64)
                            lmk_same_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_2_5_2=np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk_same=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2=np.pad(lmk5,((0, 1000 -len(lmk5)), (0, 0)), "constant") 
                            lmk_2_5=np.zeros((1000, 2), dtype=np.int64)
                            lmk_same_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2_2=np.pad(lmk5_2,((0, 1000 -len(lmk5_2)), (0, 0)), "constant") 
                            lmk_2_5_2=np.zeros((1000, 2), dtype=np.int64)
                    else:
                        try:
                            lmk5 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk5 = np.array(lmk5)
                            lmk5 = lmk5[:, [2, 1]]
                            lmk5_2 = pd.read_csv(os.path.join(prep_path2, flmk_2))
                            lmk5_2 = np.array(lmk5_2)
                            lmk5_2 = lmk5_2[:, [2, 1]]
                        except:
                            lmk_same=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_2_5=np.pad(lmk2,((0, 1000 -len(lmk2)), (0, 0)), "constant") 
                            lmk_same_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_5_2_2=np.zeros((1000, 2), dtype=np.int64)
                            lmk_2_5_2=np.pad(lmk2_2,((0, 1000 -len(lmk2_2)), (0, 0)), "constant") 
                        else:
                            lmk5_list=lmk5.tolist()
                            lmk2_list=lmk2.tolist()
                            lmk5_2_list=lmk5_2.tolist()
                            lmk2_2_list=lmk2_2.tolist()
                            for i1 in range (len(lmk5_list)):
                                if lmk5_list[i1] in lmk2_list:
                                    lmk_same_index.append(i1)
                            lmk_2_5_index=[temp for temp in range(len(lmk2_list)) if lmk2_list[temp] not in lmk5_list]
                            lmk_5_2_index=[temp for temp in range(len(lmk5_list)) if lmk5_list[temp] not in lmk2_list]
                            # pdb.set_trace()
                            lmk_same=lmk5[lmk_same_index,:]
                            lmk_same=np.pad(lmk_same,((0, 1000 -len(lmk_same)), (0, 0)), "constant") 
                            lmk_2_5=lmk2[lmk_2_5_index,:]
                            lmk_2_5=np.pad(lmk_2_5,((0, 1000 -len(lmk_2_5)), (0, 0)), "constant") 
                            lmk_5_2=lmk5[lmk_5_2_index,:]
                            lmk_5_2=np.pad(lmk_5_2,((0, 1000 -len(lmk_5_2)), (0, 0)), "constant") 
                            lmk_same_2=lmk5_2[lmk_same_index,:]
                            lmk_same_2=np.pad(lmk_same_2,((0, 1000 -len(lmk_same_2)), (0, 0)), "constant") 
                            lmk_2_5_2=lmk2_2[lmk_2_5_index,:]
                            lmk_2_5_2=np.pad(lmk_2_5_2,((0, 1000 -len(lmk_2_5_2)), (0, 0)), "constant") 
                            lmk_5_2_2=lmk5_2[lmk_5_2_index,:]
                            lmk_5_2_2=np.pad(lmk_5_2_2,((0, 1000 -len(lmk_5_2_2)), (0, 0)), "constant") 

                    dataset[flmk_large] = lmk_large
                    dataset[flmk_same] = lmk_same
                    dataset[flmk_5_2] = lmk_5_2
                    dataset[flmk_2_5] = lmk_2_5
                    dataset[flmk_same_2] = lmk_same_2
                    dataset[flmk_5_2_2] = lmk_5_2_2
                    dataset[flmk_2_5_2] = lmk_2_5_2
                    groups[group].append((fimg, flmk_same,flmk_5_2,flmk_2_5,flmk_large))
                    train_groups[group].append((fimg, flmk_same,flmk_5_2,flmk_2_5,flmk_large))

                fimg = str(num)+'_2.jpg'
                fimg_256 = str(num)+'_2_256.jpg'
                fimg_1024 = str(num)+'_2_1024.jpg'
                flmk = str(num)+'_2.csv'
                flmk_same = str(num)+'_2_same.csv'
                flmk_5_2 = str(num)+'_2_5_2.csv'
                flmk_2_5 = str(num)+'_2_2_5.csv'
                flmk_large = str(num)+'_2_large.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path4, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_256] = im_temp2
                    im_temp1 = io.imread(os.path.join(prep_path5, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg_1024] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    
                    
                    
                    
                    # ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    # try:
                        # lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        # lmk = np.array(lmk)
                        # lmk = lmk[:, [2, 1]]
                    # except:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = lmk2
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = lmk3
                            # except:
                                # lmk = np.zeros((2000, 2), dtype=np.int64)
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            
                    # else:
                        # try:
                            # lmk2 = pd.read_csv(os.path.join(prep_path6, flmk))
                            # lmk2 = np.array(lmk2)
                            # lmk2 = lmk2[:, [2, 1]]
                            # lmk = np.pad(lmk, lmk2, "constant")
                        # except:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                        # else:
                            # try:
                                # lmk3 = pd.read_csv(os.path.join(prep_path2, flmk))
                                # lmk3 = np.array(lmk3)
                                # lmk3 = lmk3[:, [2, 1]]
                                # lmk = np.pad(lmk, lmk3, "constant")
                            # except:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")
                            # else:
                                # lmk = np.pad(lmk,((0, 2000 -len(lmk)), (0, 0)), "constant")



                    try:
                        lmk_large = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk_large = np.array(lmk_large)
                        lmk_large = lmk_large[:, [2, 1]]
                    except:
                        lmk_large=np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_large=np.pad(lmk_large,((0, 1000 -len(lmk_large)), (0, 0)), "constant") 

                    # dataset[flmk] = lmk
                    # groups[group].append((fimg, flmk))
                    # train_groups[group].append((fimg, flmk))
                    dataset[flmk_large] = lmk_large
                    groups[group].append((fimg, flmk_same,flmk_5_2,flmk_2_5,flmk_large))
                    train_groups[group].append((fimg, flmk_same,flmk_5_2,flmk_2_5,flmk_large))
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups





































def LoadANHIR_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                fmask_same_with_vgg = str(num)+'_1_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_1_mask_more_than_vgg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large

                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    # pdb.set_trace()
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                fmask_same_with_vgg = str(num)+'_2_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_2_mask_more_than_vgg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                            
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups
    
    
    





def LoadANHIR_vgg_vgglarge_maskmorethanvgg_masksamewithvgg_with_corrected_LFS(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    prep_path6='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_rotate16_0.98_0.95_corrected/'
    orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                fmask_same_with_vgg = str(num)+'_1_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_1_mask_more_than_vgg.csv'
                fmask_corrected=str(num)+'_1_mask_corrected.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    ###########dense
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    
                    
                    
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large

                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    # pdb.set_trace()
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    
                    try:
                        lmk_corrected = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_corrected = np.array(lmk_corrected)
                        lmk_corrected = lmk_corrected[:, [2, 1]]
                    except:
                        
                        lmk_corrected = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_corrected = np.pad(lmk_corrected,((0, 1000 -len(lmk_corrected)), (0, 0)), "constant")
                    
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    dataset[fmask_corrected]=lmk_corrected
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,fmask_corrected))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,fmask_corrected))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                fmask_same_with_vgg = str(num)+'_2_mask_same_with_vgg.csv'
                fmask_more_than_vgg= str(num)+'_2_mask_more_than_vgg.csv'
                fmask_corrected=str(num)+'_2_mask_corrected.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    # groups[group].append(fimg)
                    # train_groups[group].append(fimg)
                    ###############vgg with vgg large with maskflownet&vgg with (maskflownet_more_than_vgg)
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path3, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = lmk2
                        except:
                            lmk = np.zeros((1000, 2), dtype=np.int64)
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    else:
                        try:
                            lmk2 = pd.read_csv(os.path.join(prep_path2, flmk))
                            lmk2 = np.array(lmk2)
                            lmk2 = lmk2[:, [2, 1]]
                            lmk = np.pad(lmk, lmk2, "constant")
                        except:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                        else:
                            lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")###############vgg with vgg large
                    lmk_mask_more_than_vgg=[]
                    lmk_mask_same_with_vgg=[]
                    lmk_mask=[]
                    try:
                        lmk_mask = pd.read_csv(os.path.join(prep_path5, flmk))
                        lmk_mask = np.array(lmk_mask)
                        lmk_mask = lmk_mask[:, [2, 1]]
                        lmk_mask=lmk_mask.tolist()
                        try:
                            lmk_mask_more_than_vgg = pd.read_csv(os.path.join(prep_path4, flmk))
                            lmk_mask_more_than_vgg = np.array(lmk_mask_more_than_vgg)
                            lmk_mask_more_than_vgg = lmk_mask_more_than_vgg[:, [2, 1]]
                            lmk_mask_more_than_vgg=lmk_mask_more_than_vgg.tolist()
                        except:
                            lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                        else:
                            lmk_mask_same_with_vgg=lmk_mask.copy()
                            for temp in lmk_mask_more_than_vgg:
                                lmk_mask_same_with_vgg.remove(temp)
                            if len(lmk_mask_same_with_vgg)==0:
                                lmk_mask_same_with_vgg=np.zeros((1000, 2), dtype=np.int64)
                            
                    except:
                        lmk_mask_more_than_vgg = np.zeros((1000, 2), dtype=np.int64)
                        lmk_mask_same_with_vgg = np.zeros((1000, 2), dtype=np.int64)
                    lmk_mask_more_than_vgg=np.array(lmk_mask_more_than_vgg)
                    lmk_mask_same_with_vgg=np.array(lmk_mask_same_with_vgg)
                    lmk_mask_more_than_vgg = np.pad(lmk_mask_more_than_vgg,((0, 1000 -len(lmk_mask_more_than_vgg)), (0, 0)), "constant")
                    lmk_mask_same_with_vgg = np.pad(lmk_mask_same_with_vgg,((0, 1000 -len(lmk_mask_same_with_vgg)), (0, 0)), "constant")
                    try:
                        lmk_corrected = pd.read_csv(os.path.join(prep_path6, flmk))
                        lmk_corrected = np.array(lmk_corrected)
                        lmk_corrected = lmk_corrected[:, [2, 1]]
                    except:
                        
                        lmk_corrected = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_corrected = np.pad(lmk_corrected,((0, 1000 -len(lmk_corrected)), (0, 0)), "constant")
                    
                    dataset[flmk] = lmk
                    dataset[fmask_same_with_vgg] = lmk_mask_same_with_vgg
                    dataset[fmask_more_than_vgg]=lmk_mask_more_than_vgg
                    dataset[fmask_corrected]=lmk_corrected
                    groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,fmask_corrected))
                    train_groups[group].append((fimg, flmk,fmask_same_with_vgg,fmask_more_than_vgg,fmask_corrected))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups









def LoadANHIR_generate_maskflownetS_dense(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
    prep_path3='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
    prep_path4='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
    prep_path5='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
    prep_path6='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.975_0.975_with_network_update/'
    
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            # if num not in ([62]):
                # continue
            if row[5] == 'training':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                fmask = str(num)+'_1_mask.csv'
                fmask_more_than_gt = str(num)+'_1_mask_more_than_gt.csv'
                fkp=str(num)+'_1'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    temp_img1=im_temp1
                    dataset[fimg] = im_temp2
                    groups[group].append((fimg))
                    train_groups[group].append((fimg))

                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                fmask = str(num)+'_2_mask.csv'
                fmask_more_than_gt = str(num)+'_2_mask_more_than_gt.csv'
                fkp=str(num)+'_2'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    temp_img2=im_temp1
                    dataset[fimg] = im_temp2
                    groups[group].append((fimg))
                    train_groups[group].append((fimg))
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg_test'
                flmk = str(num)+'_1.csv_test'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg[:-5]), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk[:-5]))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg_test'
                flmk = str(num) + '_2.csv_test'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg[:-5]), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk[:-5]))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img2=im_temp1
                    # temp_lmk2=lmk
                    # im1=appendimages(temp_img1,temp_img2)
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (200):
                        # plt.plot([temp_lmk1[i,1],temp_lmk2[i,1]+512],[temp_lmk1[i,0],temp_lmk2[i,0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Association/images/evaluation/'+str(num)+'.jpg',dpi=600)
                    # plt.close()
    return dataset, groups, train_groups, val_groups













def LoadANHIR_supervised_SFG(prep_name, subsets = [""], data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):

    prep_name1 = prep_name + 'after_affine'
    prep_name2 = prep_name + '_kp_after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2 = os.path.join(data_path, prep_name2)
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'training':
                # if num not in [337]:
                    # continue
                # print(num)
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_sfg = str(num)+'_1_sfg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    try:
                        lmk_sfg = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_sfg = np.array(lmk_sfg)
                        lmk_sfg = lmk_sfg[:, [2, 1]]
                    except:
                        lmk_sfg = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_sfg = np.pad(lmk_sfg,((0, 1000 -len(lmk_sfg)), (0, 0)), "constant")
                    dataset[flmk_sfg] = lmk_sfg
                    groups[group].append((fimg, flmk,flmk_sfg))
                    train_groups[group].append((fimg, flmk,flmk_sfg))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_sfg = str(num)+'_2_sfg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                    except:
                        lmk = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk = np.pad(lmk,((0, 1000 -len(lmk)), (0, 0)), "constant")
                    dataset[flmk] = lmk
                    try:
                        lmk_sfg = pd.read_csv(os.path.join(prep_path2, flmk))
                        lmk_sfg = np.array(lmk_sfg)
                        lmk_sfg = lmk_sfg[:, [2, 1]]
                    except:
                        lmk_sfg = np.zeros((1000, 2), dtype=np.int64)
                    else:
                        lmk_sfg = np.pad(lmk_sfg,((0, 1000 -len(lmk_sfg)), (0, 0)), "constant")
                    dataset[flmk_sfg] = lmk_sfg
                    groups[group].append((fimg, flmk,flmk_sfg))
                    train_groups[group].append((fimg, flmk,flmk_sfg))
                    
                   
            elif row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    # temp_img1=im_temp1
                    # temp_lmk1=lmk
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp2 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    im_temp2[0] = im_temp1
                    im_temp2[1] = im_temp1
                    im_temp2[2] = im_temp1
                    dataset[fimg] = im_temp2
                    try:
                        lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                        lmk = np.array(lmk)
                        lmk = lmk[:, [2, 1]]
                        lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    except:
                        lmk = np.zeros((200, 2), dtype=np.int64)
                        # lmk = np.zeros((0,0), dtype=np.int64)
                        dataset[flmk] = lmk
                        print('lmk original length: 0')
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
                    else:
                        dataset[flmk] = lmk
                        groups[group].append((fimg, flmk))
                        val_groups[group].append((fimg, flmk))
    return dataset, groups, train_groups, val_groups







def LoadANHIR_SFG(prep_name, subsets = [""], data_path = r"/ssd1/wxy/association/Maskflownet_association_1024/dataset/"):

    prep_name1 = prep_name + 'after_affine'
    prep_name2 = prep_name + '_kp_after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path2 = os.path.join(data_path, prep_name2)
    kppath='/ssd1/wxy/TMI_rebuttle_SFG/data/kps_0.85zone0.04mutiscale_imgsize1024/'
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    grid=(np.arange(2*10+1)-10)
    grid_x,grid_y=np.meshgrid(grid,grid)
    grid2=np.concatenate((np.expand_dims(grid_x,2),np.expand_dims(grid_y,2)),2).reshape((-1,2))#(25,2)
    with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            
            if row[5] == 'evaluation':
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                flmk_sfg = str(num)+'_1_sfg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1 = np.zeros((3, np.shape(im_temp1)[0], np.shape(im_temp1)[1]))
                    img1[0] = im_temp1
                    img1[1] = im_temp1
                    img1[2] = im_temp1
                    dataset[fimg] = img1
                    try:
                        lmk_sfg = pd.read_csv(os.path.join(kppath, flmk))
                        lmk_sfg = np.array(lmk_sfg)
                        lmk_sfg = lmk_sfg[:, [1, 2]]#######################在得到kp时忘记翻转xy
                    except:
                        lmk_sfg = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_sfg = np.pad(lmk_sfg,((0, 200 -len(lmk_sfg)), (0, 0)), "constant")
                    lmk1=lmk_sfg
                    dataset[flmk_sfg] = lmk_sfg
                    groups[group].append((fimg, flmk_sfg))
                    train_groups[group].append((fimg, flmk_sfg))

                fimg = str(num)+'_2.jpg'
                flmk = str(num)+'_2.csv'
                flmk_sfg = str(num)+'_2_sfg.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in train_groups:
                        train_groups[group] = []
                    im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2 = np.zeros((3, np.shape(im_temp2)[0], np.shape(im_temp2)[1]))
                    img2[0] = im_temp2
                    img2[1] = im_temp2
                    img2[2] = im_temp2
                    dataset[fimg] = img2
                    try:
                        lmk_sfg = pd.read_csv(os.path.join(kppath, flmk))
                        lmk_sfg = np.array(lmk_sfg)
                        lmk_sfg = lmk_sfg[:, [1, 2]]#####################在得到kp时忘记翻转xy
                    except:
                        lmk_sfg = np.zeros((200, 2), dtype=np.int64)
                    else:
                        lmk_sfg = np.pad(lmk_sfg,((0, 200 -len(lmk_sfg)), (0, 0)), "constant")
                    dataset[flmk_sfg] = lmk_sfg
                    groups[group].append((fimg, flmk_sfg))
                    train_groups[group].append((fimg, flmk_sfg))
                    lmk2=lmk_sfg
                # pdb.set_trace()
                # im1=appendimages(im_temp1,im_temp2)
                # plt.figure()
                # plt.imshow(im1)
                # for i in range (lmk2.shape[0]):
                    # plt.plot([lmk1[i,1],lmk2[i,1]+1024],[lmk1[i,0],lmk2[i,0]], '#FF0033',linewidth=0.5)
                # plt.savefig("/ssd1/wxy/association/Maskflownet_association_1024/test/"+str(num)+'.jpg',dpi=600)
                # plt.close()
                   
            else:
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img1
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[6:16]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
                    dataset[fimg] = img2
                    para_preprocess=np.concatenate((np.zeros([1,1]),np.expand_dims(np.array(row[16:26]),0)),1).astype('double')####(1,11)
                    lmk = pd.read_csv(os.path.join(prep_path1, flmk))
                    # lmk = pd.read_csv(os.path.join("/ssd1/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/", flmk))
                    lmk = np.array(lmk)
                    lmk = lmk[:, [2, 1]]
                    lmk = np.pad(lmk, ((0, 200 - len(lmk)), (0, 0)), "constant")
                    lmk=np.concatenate((lmk,np.pad(para_preprocess,((0,200-1),(0,0)),'edge')),1)
                    dataset[flmk] = lmk
                    groups[group].append((fimg, flmk))
                    val_groups[group].append((fimg, flmk))
    return dataset, groups, train_groups, val_groups
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                # fileList.append(os.path.join(root, fileName))
                fileList.append(fileName)
    return fileList
def LoadACROBAT_baseline():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            train_pairs.append((filename1, filename2))
            # train_pairs.append((filename2, filename1))
            dataset[filename1] = img1
            dataset[filename2] = img2
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_DFS_SFG_multiscale():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    # trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/TMI_rebuttle_SFG/data/kps_0.8zone0.028mutiscale_imgsize1024_MAX_MATCHES2000_delete_near_ransac3/"
    trainkppath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    trainkppath4="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname1_3=filename1.split('.')[0]+'_1_3.csv'
            lmkname1_4=filename1.split('.')[0]+'_1_4.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            lmkname2_3=filename1.split('.')[0]+'_2_3.csv'
            lmkname2_4=filename1.split('.')[0]+'_2_4.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_3 = pd.read_csv(os.path.join(trainkppath3, lmkname1))
                lmk1_3 = np.array(lmk1_3)
                lmk1_3 = lmk1_3[:, [2, 1]]
            except:
                lmk1_3 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_3 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_3 = np.pad(lmk1_3,((0, 1200 -len(lmk1_3)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_3 = pd.read_csv(os.path.join(trainkppath3, lmkname2))
                lmk2_3 = np.array(lmk2_3)
                lmk2_3 = lmk2_3[:, [2, 1]]
                lmk2_3 = np.pad(lmk2_3,((0, 1200 -len(lmk2_3)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_4 = pd.read_csv(os.path.join(trainkppath4, lmkname1))
                lmk1_4 = np.array(lmk1_4)
                lmk1_4 = lmk1_4[:, [2, 1]]
            except:
                lmk1_4 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_4 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_4 = np.pad(lmk1_4,((0, 1200 -len(lmk1_4)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_4 = pd.read_csv(os.path.join(trainkppath4, lmkname2))
                lmk2_4 = np.array(lmk2_4)
                lmk2_4 = lmk2_4[:, [2, 1]]
                lmk2_4 = np.pad(lmk2_4,((0, 1200 -len(lmk2_4)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1
            dataset[lmkname2] = lmk2
            dataset[lmkname1_2] = lmk1_2
            dataset[lmkname2_2] = lmk2_2
            dataset[lmkname2_3] = lmk2_3
            dataset[lmkname1_3] = lmk1_3
            dataset[lmkname1_4] = lmk1_4
            dataset[lmkname2_4] = lmk2_4
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2,lmkname1_3,lmkname2_3,lmkname1_4,lmkname2_4))
            # train_pairs.append((filename2, filename1,lmkname2,lmkname1,lmkname2_2,lmkname1_2,lmkname2_3,lmkname1_3))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname1_training=filename1.split('.')[0]+'_training.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            lmkname2_training=filename2.split('.')[0]+'_training.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
                # if filename1== '83_KI67_val.jpg':
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_SFG_multiscale():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/TMI_rebuttle_SFG/data/kps_0.8zone0.028mutiscale_imgsize1024_MAX_MATCHES2000/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1
            dataset[lmkname2] = lmk2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # train_pairs.append((filename2, filename1,lmkname2,lmkname1))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs

def LoadACROBAT_baseline512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    trainkpspath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            try:
                lmk1 = pd.read_csv(os.path.join(trainkpspath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkpspath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            dataset[filename1] = img1
            dataset[filename2] = img2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
            # train_pairs.append((filename1, filename2))
            # train_pairs.append((filename2, filename1))
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_baseline512_PWC():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for num in range(750):
        trainfilenames_num=test_findfile(trainimgpath, '.jpg', str(num)+'_')
        for count,trainfilename_num in enumerate(trainfilenames_num):
            filename1=trainfilename_num
            if count>len(trainfilenames_num)-2:
                continue
            else:
                img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
                img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
                if img1.max()<2:
                    print(img1.max())
                    img1=np.uint8(img1*255)
                dataset[filename1] = img1
                for trainfilename_num2 in trainfilenames_num[count+1:]:
                    filename2=trainfilename_num2
                    img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
                    img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
                    if img2.max()<2:
                        print(img2.max())
                        img2=np.uint8(img2*255)
                    dataset[filename2] = img2
                    train_pairs.append((filename1, filename2))
                    train_pairs.append((filename2, filename1))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_baseline512_ite1():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainimgpath512="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainimgpath2048="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
    orbpath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_4096_to1024_delete_near3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename1_512=trainfilename.split('.')[0]+'_512.jpg'
            filename1_2048=trainfilename.split('.')[0]+'_2048.jpg'
            filename2=num+'_HE_train.jpg'
            filename2_512=num+'_HE_train_512.jpg'
            filename2_2048=num+'_HE_train_2048.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            flmk_orb1=filename1.split('.')[0]+'.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            flmk_orb2=filename2.split('.')[0]+'.csv'
            
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img1_512 = io.imread(os.path.join(trainimgpath512, filename1), as_gray=True)
            img1_512 = np.concatenate((np.expand_dims(img1_512,0),np.expand_dims(img1_512,0),np.expand_dims(img1_512,0)),0)
            if img1_512.max()<2:
                img1_512=np.uint8(img1_512*255)
            img1_2048 = io.imread(os.path.join(trainimgpath2048, filename1), as_gray=True)
            img1_2048 = np.concatenate((np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0)),0)
            if img1_2048.max()<2:
                img1_2048=np.uint8(img1_2048*255)
            dataset[filename1] = img1
            dataset[filename1_512] = img1_512
            dataset[filename1_2048] = img1_2048
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            img2_512 = io.imread(os.path.join(trainimgpath512, filename2), as_gray=True)
            img2_512 = np.concatenate((np.expand_dims(img2_512,0),np.expand_dims(img2_512,0),np.expand_dims(img2_512,0)),0)
            if img2_512.max()<2:
                img2_512=np.uint8(img2_512*255)
            img2_2048 = io.imread(os.path.join(trainimgpath2048, filename2), as_gray=True)
            img2_2048 = np.concatenate((np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0)),0)
            if img2_2048.max()<2:
                img2_2048=np.uint8(img2_2048*255)
            dataset[filename2] = img2
            dataset[filename2_512] = img2_512
            dataset[filename2_2048] = img2_2048
            dataset[filename1] = img1
            dataset[filename2] = img2
            
            # lmk_obtained1=np.concatenate((np.unique(lmk1, return_index=False,axis=0), np.unique(lmk1_2, return_index=False,axis=0)), 0)
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            # dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            # lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>2)[0],:]
            # lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk1_orb#lmk_orb_valid1
            
            # lmk_obtained2=np.concatenate((np.unique(lmk2, return_index=False,axis=0), np.unique(lmk2_2, return_index=False,axis=0)), 0)
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            # dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            # lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>2)[0],:]
            # lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk2_orb#lmk_orb_valid2
            # pdb.set_trace()
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_baseline512_ite2():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainimgpath512="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainimgpath2048="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
    trainkpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    orbpath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_4096_to1024_delete_near3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename1_512=trainfilename.split('.')[0]+'_512.jpg'
            filename1_2048=trainfilename.split('.')[0]+'_2048.jpg'
            filename2=num+'_HE_train.jpg'
            filename2_512=num+'_HE_train_512.jpg'
            filename2_2048=num+'_HE_train_2048.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            flmk_orb1=filename1.split('.')[0]+'.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            flmk_orb2=filename2.split('.')[0]+'.csv'
            
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img1_512 = io.imread(os.path.join(trainimgpath512, filename1), as_gray=True)
            img1_512 = np.concatenate((np.expand_dims(img1_512,0),np.expand_dims(img1_512,0),np.expand_dims(img1_512,0)),0)
            if img1_512.max()<2:
                img1_512=np.uint8(img1_512*255)
            img1_2048 = io.imread(os.path.join(trainimgpath2048, filename1), as_gray=True)
            img1_2048 = np.concatenate((np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0)),0)
            if img1_2048.max()<2:
                img1_2048=np.uint8(img1_2048*255)
            dataset[filename1] = img1
            dataset[filename1_512] = img1_512
            dataset[filename1_2048] = img1_2048
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            img2_512 = io.imread(os.path.join(trainimgpath512, filename2), as_gray=True)
            img2_512 = np.concatenate((np.expand_dims(img2_512,0),np.expand_dims(img2_512,0),np.expand_dims(img2_512,0)),0)
            if img2_512.max()<2:
                img2_512=np.uint8(img2_512*255)
            img2_2048 = io.imread(os.path.join(trainimgpath2048, filename2), as_gray=True)
            img2_2048 = np.concatenate((np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0)),0)
            if img2_2048.max()<2:
                img2_2048=np.uint8(img2_2048*255)
            
            try:
                lmk1 = pd.read_csv(os.path.join(trainkpspath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkpspath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            
            dataset[filename2] = img2
            dataset[filename2_512] = img2_512
            dataset[filename2_2048] = img2_2048
            dataset[filename1] = img1
            dataset[filename2] = img2
            
            lmk_obtained1=lmk1#np.concatenate((np.unique(lmk1, return_index=False,axis=0), np.unique(lmk1_2, return_index=False,axis=0)), 0)
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>3)[0],:]
            lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk1_orb#lmk_orb_valid1
            
            lmk_obtained2=lmk2#np.concatenate((np.unique(lmk2, return_index=False,axis=0), np.unique(lmk2_2, return_index=False,axis=0)), 0)
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>3)[0],:]
            lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk2_orb#lmk_orb_valid2
            # pdb.set_trace()
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_baseline512_1500():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    trainkpspath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    indices = np.arange(0, len(trainfilenames), dtype=np.int)
    np.random.shuffle(indices)
    trainfilenames=np.array(trainfilenames)[indices].tolist()
    count_nums={}
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            if num not in count_nums:
                count_nums[num]=1
            else:
                if count_nums[num]<2:
                    count_nums[num]=count_nums[num]+1
                else:
                    continue
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            try:
                lmk1 = pd.read_csv(os.path.join(trainkpspath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkpspath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            dataset[filename1] = img1
            dataset[filename2] = img2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_baseline512_1000():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    trainkpspath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    indices = np.arange(0, len(trainfilenames), dtype=np.int)
    np.random.shuffle(indices)
    trainfilenames=np.array(trainfilenames)[indices].tolist()
    count_nums={}
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            if num not in count_nums:
                count_nums[num]=1
            else:
                if int(num)%2==0:
                    if count_nums[num]<2:
                        count_nums[num]=count_nums[num]+1
                    else:
                        continue
                else:
                    if count_nums[num]<1:
                        count_nums[num]=count_nums[num]+1
                    else:
                        continue
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            try:
                lmk1 = pd.read_csv(os.path.join(trainkpspath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkpspath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkpspath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            dataset[filename1] = img1
            dataset[filename2] = img2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_DFS_SFG_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    trainkppath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            lmkname1_3=filename1.split('.')[0]+'_1_3.csv'
            lmkname2_3=filename1.split('.')[0]+'_2_3.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_3 = pd.read_csv(os.path.join(trainkppath3, lmkname1))
                lmk1_3 = np.array(lmk1_3)
                lmk1_3 = lmk1_3[:, [2, 1]]
            except:
                lmk1_3 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_3 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_3 = np.pad(lmk1_3,((0, 1200 -len(lmk1_3)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_3 = pd.read_csv(os.path.join(trainkppath3, lmkname2))
                lmk2_3 = np.array(lmk2_3)
                lmk2_3 = lmk2_3[:, [2, 1]]
                lmk2_3 = np.pad(lmk2_3,((0, 1200 -len(lmk2_3)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            dataset[lmkname1_3] = lmk1_3/2
            dataset[lmkname2_3] = lmk2_3/2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2,lmkname1_3,lmkname2_3))
            # train_pairs.append((filename2, filename1,lmkname2,lmkname1))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            lmkname1_training=filename1.split('.')[0]+'_training.xlsx'
            lmkname2_training=filename2.split('.')[0]+'_training.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
                
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
                
                # if filename1=='57_PGR_val.jpg':
                    # lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                    # lmk1=lmk1[1:,1:].astype('float')
                    # lmk1 = lmk1[:, [1, 0]]
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                    # lmk2=lmk2[1:,1:].astype('float')
                    # lmk2 = lmk2[:, [1, 0]]
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        # train_pairs.append((filename2, filename1,lmkname2_training,lmkname1_training,lmkname2_training,lmkname1_training))
                # if filename1== '83_KI67_val.jpg':
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        
                        
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_DFS_SFG_singlescale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_singlescale_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            lmkname1_training=filename1.split('.')[0]+'_training.xlsx'
            lmkname2_training=filename2.split('.')[0]+'_training.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
                
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
                
                # if filename1=='57_PGR_val.jpg':
                    # lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                    # lmk1=lmk1[1:,1:].astype('float')
                    # lmk1 = lmk1[:, [1, 0]]
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                    # lmk2=lmk2[1:,1:].astype('float')
                    # lmk2 = lmk2[:, [1, 0]]
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        # train_pairs.append((filename2, filename1,lmkname2_training,lmkname1_training,lmkname2_training,lmkname1_training))
                # if filename1== '83_KI67_val.jpg':
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        
                        
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_DFS_SFG_singlescale512_layer2_3():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_singlescale_layer2_3_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            lmkname1_training=filename1.split('.')[0]+'_training.xlsx'
            lmkname2_training=filename2.split('.')[0]+'_training.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
                
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
                
                # if filename1=='57_PGR_val.jpg':
                    # lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                    # lmk1=lmk1[1:,1:].astype('float')
                    # lmk1 = lmk1[:, [1, 0]]
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                    # lmk2=lmk2[1:,1:].astype('float')
                    # lmk2 = lmk2[:, [1, 0]]
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        # train_pairs.append((filename2, filename1,lmkname2_training,lmkname1_training,lmkname2_training,lmkname1_training))
                # if filename1== '83_KI67_val.jpg':
                    # dataset[lmkname1_training] = np.pad(lmk1[:,0:2],((0, 1200 -lmk1.shape[0]), (0, 0)), "constant")
                    # dataset[lmkname2_training] = np.pad(lmk2[:,0:2],((0, 1200 -lmk2.shape[0]), (0, 0)), "constant")
                    # for cycle in range(100):
                        # train_pairs.append((filename1, filename2,lmkname1_training,lmkname2_training,lmkname1_training,lmkname2_training))
                        
                        
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_LFS_SFG_kps_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_4096_to1024_delete_near3/"
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainimgpath512="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainimgpath2048="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename1_512=trainfilename.split('.')[0]+'_512.jpg'
            filename1_2048=trainfilename.split('.')[0]+'_2048.jpg'
            filename2=num+'_HE_train.jpg'
            filename2_512=num+'_HE_train_512.jpg'
            filename2_2048=num+'_HE_train_2048.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            flmk_orb1=filename1.split('.')[0]+'.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            flmk_orb2=filename2.split('.')[0]+'.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img1_512 = io.imread(os.path.join(trainimgpath512, filename1), as_gray=True)
            img1_512 = np.concatenate((np.expand_dims(img1_512,0),np.expand_dims(img1_512,0),np.expand_dims(img1_512,0)),0)
            if img1_512.max()<2:
                img1_512=np.uint8(img1_512*255)
            img1_2048 = io.imread(os.path.join(trainimgpath2048, filename1), as_gray=True)
            img1_2048 = np.concatenate((np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0)),0)
            if img1_2048.max()<2:
                img1_2048=np.uint8(img1_2048*255)
            dataset[filename1] = img1
            dataset[filename1_512] = img1_512
            dataset[filename1_2048] = img1_2048
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            img2_512 = io.imread(os.path.join(trainimgpath512, filename2), as_gray=True)
            img2_512 = np.concatenate((np.expand_dims(img2_512,0),np.expand_dims(img2_512,0),np.expand_dims(img2_512,0)),0)
            if img2_512.max()<2:
                img2_512=np.uint8(img2_512*255)
            img2_2048 = io.imread(os.path.join(trainimgpath2048, filename2), as_gray=True)
            img2_2048 = np.concatenate((np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0)),0)
            if img2_2048.max()<2:
                img2_2048=np.uint8(img2_2048*255)
            dataset[filename2] = img2
            dataset[filename2_512] = img2_512
            dataset[filename2_2048] = img2_2048
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large

            lmk_obtained1=lmk1#np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0)), 0)
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>5)[0],:]
            lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk_orb_valid1
            
            lmk_obtained2=lmk2#np.concatenate((np.unique(lmk, return_index=False,axis=0), np.unique(lmk2, return_index=False,axis=0)), 0)
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>5)[0],:]
            lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk_orb_valid2
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs


def LoadACROBAT_PCG_kps_ite1_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_4096_to1024_delete_near3/"
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainimgpath512="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainimgpath2048="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename1_512=trainfilename.split('.')[0]+'_512.jpg'
            filename1_2048=trainfilename.split('.')[0]+'_2048.jpg'
            filename2=num+'_HE_train.jpg'
            filename2_512=num+'_HE_train_512.jpg'
            filename2_2048=num+'_HE_train_2048.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            flmk_orb1=filename1.split('.')[0]+'.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            flmk_orb2=filename2.split('.')[0]+'.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img1_512 = io.imread(os.path.join(trainimgpath512, filename1), as_gray=True)
            img1_512 = np.concatenate((np.expand_dims(img1_512,0),np.expand_dims(img1_512,0),np.expand_dims(img1_512,0)),0)
            if img1_512.max()<2:
                img1_512=np.uint8(img1_512*255)
            img1_2048 = io.imread(os.path.join(trainimgpath2048, filename1), as_gray=True)
            img1_2048 = np.concatenate((np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0)),0)
            if img1_2048.max()<2:
                img1_2048=np.uint8(img1_2048*255)
            dataset[filename1] = img1
            dataset[filename1_512] = img1_512
            dataset[filename1_2048] = img1_2048
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            img2_512 = io.imread(os.path.join(trainimgpath512, filename2), as_gray=True)
            img2_512 = np.concatenate((np.expand_dims(img2_512,0),np.expand_dims(img2_512,0),np.expand_dims(img2_512,0)),0)
            if img2_512.max()<2:
                img2_512=np.uint8(img2_512*255)
            img2_2048 = io.imread(os.path.join(trainimgpath2048, filename2), as_gray=True)
            img2_2048 = np.concatenate((np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0)),0)
            if img2_2048.max()<2:
                img2_2048=np.uint8(img2_2048*255)
            dataset[filename2] = img2
            dataset[filename2_512] = img2_512
            dataset[filename2_2048] = img2_2048
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            lmk_obtained1=np.concatenate((np.unique(lmk1, return_index=False,axis=0), np.unique(lmk1_2, return_index=False,axis=0)), 0)
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>2)[0],:]
            lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk_orb_valid1
            
            lmk_obtained2=np.concatenate((np.unique(lmk2, return_index=False,axis=0), np.unique(lmk2_2, return_index=False,axis=0)), 0)
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>2)[0],:]
            lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk_orb_valid2
            # pdb.set_trace()
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs

def LoadACROBAT_PCG_kps_ite2_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_ORB16s8_4096_to1024_delete_near3/"
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
    trainimgpath512="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    trainimgpath2048="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
    trainkppath2="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
    trainkppath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename1_512=trainfilename.split('.')[0]+'_512.jpg'
            filename1_2048=trainfilename.split('.')[0]+'_2048.jpg'
            filename2=num+'_HE_train.jpg'
            filename2_512=num+'_HE_train_512.jpg'
            filename2_2048=num+'_HE_train_2048.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname1_2=filename1.split('.')[0]+'_1_2.csv'
            flmk_orb1=filename1.split('.')[0]+'.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            lmkname2_2=filename1.split('.')[0]+'_2_2.csv'
            flmk_orb2=filename2.split('.')[0]+'.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img1_512 = io.imread(os.path.join(trainimgpath512, filename1), as_gray=True)
            img1_512 = np.concatenate((np.expand_dims(img1_512,0),np.expand_dims(img1_512,0),np.expand_dims(img1_512,0)),0)
            if img1_512.max()<2:
                img1_512=np.uint8(img1_512*255)
            img1_2048 = io.imread(os.path.join(trainimgpath2048, filename1), as_gray=True)
            img1_2048 = np.concatenate((np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0),np.expand_dims(img1_2048,0)),0)
            if img1_2048.max()<2:
                img1_2048=np.uint8(img1_2048*255)
            dataset[filename1] = img1
            dataset[filename1_512] = img1_512
            dataset[filename1_2048] = img1_2048
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            img2_512 = io.imread(os.path.join(trainimgpath512, filename2), as_gray=True)
            img2_512 = np.concatenate((np.expand_dims(img2_512,0),np.expand_dims(img2_512,0),np.expand_dims(img2_512,0)),0)
            if img2_512.max()<2:
                img2_512=np.uint8(img2_512*255)
            img2_2048 = io.imread(os.path.join(trainimgpath2048, filename2), as_gray=True)
            img2_2048 = np.concatenate((np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0),np.expand_dims(img2_2048,0)),0)
            if img2_2048.max()<2:
                img2_2048=np.uint8(img2_2048*255)
            dataset[filename2] = img2
            dataset[filename2_512] = img2_512
            dataset[filename2_2048] = img2_2048
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_2 = pd.read_csv(os.path.join(trainkppath2, lmkname1))
                lmk1_2 = np.array(lmk1_2)
                lmk1_2 = lmk1_2[:, [2, 1]]
            except:
                lmk1_2 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_2 = np.pad(lmk1_2,((0, 1200 -len(lmk1_2)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_2 = pd.read_csv(os.path.join(trainkppath2, lmkname2))
                lmk2_2 = np.array(lmk2_2)
                lmk2_2 = lmk2_2[:, [2, 1]]
                lmk2_2 = np.pad(lmk2_2,((0, 1200 -len(lmk2_2)), (0, 0)), "constant")###############vgg with vgg large
            try:
                lmk1_3 = pd.read_csv(os.path.join(trainkppath3, lmkname1))
                lmk1_3 = np.array(lmk1_3)
                lmk1_3 = lmk1_3[:, [2, 1]]
            except:
                lmk1_3 = np.zeros((1200, 2), dtype=np.int64)
                lmk2_3 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1_3 = np.pad(lmk1_3,((0, 1200 -len(lmk1_3)), (0, 0)), "constant")###############vgg with vgg large
                lmk2_3 = pd.read_csv(os.path.join(trainkppath3, lmkname2))
                lmk2_3 = np.array(lmk2_3)
                lmk2_3 = lmk2_3[:, [2, 1]]
                lmk2_3 = np.pad(lmk2_3,((0, 1200 -len(lmk2_3)), (0, 0)), "constant")###############vgg with vgg large
            lmk_obtained1=np.concatenate((np.unique(lmk1, return_index=False,axis=0), np.unique(lmk1_2, return_index=False,axis=0),np.unique(lmk1_3, return_index=False,axis=0)), 0)
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>2)[0],:]
            lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk_orb_valid1
            
            lmk_obtained2=np.concatenate((np.unique(lmk2, return_index=False,axis=0), np.unique(lmk2_2, return_index=False,axis=0), np.unique(lmk2_3, return_index=False,axis=0)), 0)
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>2)[0],:]
            lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk_orb_valid2
            # pdb.set_trace()
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_SFG_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
    # trainkppath="/ssd2/wxy/IPCG_Acrobat/TMI_rebuttle_SFG/data/kps_0.8zone0.028mutiscale_imgsize1024_MAX_MATCHES2000/"
    trainkppath="/ssd2/wxy/IPCG_Acrobat/TMI_rebuttle_SFG/data/kps_0.8zone0.028mutiscale_imgsize1024_MAX_MATCHES2000_delete_near_ransac3/"
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    trainfilenames=test_findfile(trainimgpath,'.jpg','')
    validfilenames=test_findfile(validimgpath,'.jpg','')
    for trainfilename in trainfilenames:
        stain_type=trainfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            num=trainfilename.split('_')[0]
            filename1=trainfilename
            filename2=num+'_HE_train.jpg'
            lmkname1=filename1.split('.')[0]+'_1.csv'
            lmkname2=filename1.split('.')[0]+'_2.csv'
            img1 = io.imread(os.path.join(trainimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(trainimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 = pd.read_csv(os.path.join(trainkppath, lmkname1))
                lmk1 = np.array(lmk1)
                lmk1 = lmk1[:, [2, 1]]
            except:
                lmk1 = np.zeros((1200, 2), dtype=np.int64)
                lmk2 = np.zeros((1200, 2), dtype=np.int64)
            else:
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")###############vgg with vgg large
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")###############vgg with vgg large
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # train_pairs.append((filename2, filename1,lmkname2,lmkname1))
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                if filename1=='57_PGR_val.jpg':
                    lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_for_submit():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1/"#####first submission
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127/"#####second submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#####third submission
    validfilenames=test_findfile(validimgpath,'.jpg','')
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                print(img1.max())
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                print(img2.max())
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]##############first submission -1###################matlab转python要-1
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]##############first submission -1###################matlab转python要-1
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_image_visualization512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    print(validimgpath)
    validfilenames=test_findfile(validimgpath,'.jpg','')
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                # if filename1=='57_PGR_val.jpg':
                    # lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_image_visualization1024():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validkppath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
    validkppath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
    print(validkppath)
    validfilenames=test_findfile(validimgpath,'.jpg','')
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validkppath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]*2
                # if filename1=='57_PGR_val.jpg':
                    # lmk1=lmk1[0:-7,:]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validkppath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]*2
                # if filename1=='57_PGR_val.jpg':
                    # lmk2=lmk2[0:-7,:]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_image_visualization1024_submit():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    # validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python_add_large57/"#######third submission
    validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
    validfilenames=test_findfile(validimgpath,'.jpg','')
    
    csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
    csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
            # if int(validfilename.split('_')[0]) not in [25,26,27,29,31,36,50,65,81,93]:
                # continue
            num=validfilename.split('_')[0]
            filename1=validfilename
            filename2=num+'_HE_val.jpg'
            lmkname1=filename1.split('.')[0]+'.xlsx'
            lmkname2=filename2.split('.')[0]+'.xlsx'
            img1 = io.imread(os.path.join(validimgpath, filename1), as_gray=True)
            img1 = np.concatenate((np.expand_dims(img1,0),np.expand_dims(img1,0),np.expand_dims(img1,0)),0)
            if img1.max()<2:
                img1=np.uint8(img1*255)
            img2 = io.imread(os.path.join(validimgpath, filename2), as_gray=True)
            img2 = np.concatenate((np.expand_dims(img2,0),np.expand_dims(img2,0),np.expand_dims(img2,0)),0)
            if img2.max()<2:
                img2=np.uint8(img2*255)
            dataset[filename1] = img1
            dataset[filename2] = img2
            try:
                # pdb.set_trace()
                lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
                lmk1=lmk1[1:,1:].astype('float')
                lmk1 = lmk1[:, [1, 0]]
                lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],7][0]*np.ones([200,1])
                if isinstance(csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0],(int,float)):
                    rotation=csvdata2[np.where(csvdata2[:,1]==int(num))[0],13][0]*np.ones([200,1])
                else:
                    rotation=1000*np.ones([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename1)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
                
                lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
                lmk2=lmk2[1:,1:].astype('float')
                lmk2 = lmk2[:, [1, 0]]
                lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
                resolution=csvdata2[np.where(csvdata2[:,1]==int(num))[0],8][0]*np.ones([200,1])
                rotation=np.zeros([200,1])
                crop_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],1:5],((0,200-1),(0,0)),'edge')
                pad_para=np.pad(csvdata[np.where(csvdata[:,0]==filename2)[0],6:10],((0,200-1),(0,0)),'edge')
                lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
            except:
                valid_pairs.append((filename1, filename2,None,None))
                print(lmkname1)
            else:
                dataset[lmkname1] = lmk1
                dataset[lmkname2] = lmk2
                valid_pairs.append((filename1, filename2,lmkname1,lmkname2))
            # im=appendimages(img1[0,:,:],img2[0,:,:])
            # plt.figure()
            # plt.imshow(im)
            # plt.plot([lmk1[:,1],lmk2[:,1]+1024],[lmk1[:,0],lmk2[:,0]], '#FF0033',linewidth=0.5)
            # plt.savefig("/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/visualization/"+num+'.jpg')
            # plt.close()
    # train_pairs=valid_pairs
    # valid_pairs=train_pairs
    return dataset, train_pairs, valid_pairs

    
def addsiftkp(coords, kp,desc):
    len = np.shape(kp)[0]
    descnew=[]
    ## init , raw 0 is 0, 这样后面匹配到第一个点的匹配不会被当成不匹配而消去
    sizedesc = np.shape(desc[0])
    coords.append([0, 0])
    descnew.append(np.zeros(sizedesc))
    for i in range(len):
        siftco = [int(round(kp[i].pt[1])), int(round(kp[i].pt[0]))]
        if siftco not in coords:
            coords.append(siftco)
            descnew.append(desc[i])
    num=np.shape(coords)[0]
    for i in range (800 -num):
        coords.append([0, 0])
        # coords=np.pad(coords, ((0, 800 - len(coords)), (0, 0)), "constant")
    # pdb.set_trace()
    return coords,descnew
    