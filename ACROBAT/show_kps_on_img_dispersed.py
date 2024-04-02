import shutil
#coding:utf-8
import os
import csv
import pdb
from shutil import copyfile
import pandas as pd
import numpy as np
from skimage import io
from pylab import *
import random
import matplotlib.pyplot as plt
import cv2
import gc
from PIL import Image
import scipy.io as scio
dataset = {}
groups = {}
train_pairs = []
eval_pairs = []
all_pairs = []
out_pairs = []
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                # fileList.append(os.path.join(root, fileName))
                fileList.append(fileName)
    return fileList
###########################kps matched by ourself
savepath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/kps_on_imgs_dispersed/vgg+ite_same_pixel_multiscale_kps_on_imgs_sparse_HR2/"
# savepath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/kps_on_imgs_dispersed/valid_hand-annotations/"
if not os.path.exists(savepath):
    os.mkdir(savepath)


colorpath="/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/warp_results14aJan03-1102_88480_train_final_4096_kps_add1_0127-python-color/"
colorpath2="/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/warp_resultsc83Jan08-2004_168590_train_final_4096_kps_add1_0127-python-color/"
colorpath3="/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/warp_results390Jan02-2356_32444_valid_final_4096_kps_add1_0127-python-color/"
trainimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
# trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac/"
trainkppath="/ssd2/wxy/IPCG_Acrobat/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16_delete_near_ransac3/"
trainkppath2="/ssd2/wxy/IPCG_Acrobat/TMI_rebuttle_SFG/data/kps_0.8zone0.028mutiscale_imgsize1024_MAX_MATCHES2000_delete_near_ransac3/"
trainkppath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
trainkppath4="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85_delete_near_ransac3/"
# validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python_add_large57/"#######third submission
validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to1024_all_kps_add1_0127-python/"#######third submission
trainfilenames=test_findfile(trainkppath,'.csv','')
validfilenames=test_findfile(validimgpath,'.jpg','')
dis_mins=[]
dis_mins2=[]


for trainfilename in trainfilenames:
    stain_type=trainfilename.split('_')[-1]
    if stain_type[0]=='2':
        continue
    else:
        num=trainfilename.split('_')[0]
        if num!='95':
            continue
        lmkname1=trainfilename
        lmkname2=trainfilename.split('.')[0][0:-1]+'2.csv'
        filename1=trainfilename.split('.')[0][0:-2]+'.jpg'
        filename2=trainfilename.split('_')[0]+'_HE_train.jpg'

# trainkppath=validimgpath
# for trainfilename in validfilenames:
    # stain_type=trainfilename.split('_')[1]
    # num=trainfilename.split('_')[0]
    # if stain_type=='HE' or num!='82':
        # continue
    # else:
        # filename1=trainfilename
        # filename2=num+'_HE_val.jpg'
        # lmkname1=filename1.split('.')[0]+'.xlsx'
        # lmkname2=num+'_HE_val.xlsx'
        
        
        
        try:
            img1 = io.imread(os.path.join(colorpath, filename1), as_gray=False)
        except:
            try:
                img1 = io.imread(os.path.join(colorpath2, filename1), as_gray=False)
            except:
                img1 = io.imread(os.path.join(colorpath3, filename1), as_gray=False)
        try:
            img2 = io.imread(os.path.join(colorpath, filename2), as_gray=False)
        except:
            try:
                img2 = io.imread(os.path.join(colorpath2, filename2), as_gray=False)
            except:
                img2 = io.imread(os.path.join(colorpath3, filename2), as_gray=False)
        try:
            lmk1 =pd.read_csv(os.path.join(trainkppath, lmkname1))
            lmk1=np.array(lmk1)
            lmk1 = lmk1[:, [2, 1]]*4
            lmk2 =pd.read_csv(os.path.join(trainkppath, lmkname2))
            lmk2=np.array(lmk2)
            lmk2 = lmk2[:, [2, 1]]*4
            
            
            
            # lmk1 =np.array(pd.read_excel(os.path.join(trainkppath, lmkname1),header=None,index_col=None))
            # lmk1=lmk1[1:,1:].astype('float')
            # lmk1 = lmk1[:, [1, 0]]*4
            # lmk2 =np.array(pd.read_excel(os.path.join(trainkppath, lmkname2),header=None,index_col=None))
            # lmk2=lmk2[1:,1:].astype('float')
            # lmk2 = lmk2[:, [1, 0]]*4
        except:
            lmk1=np.zeros([10,2])
            lmk2=np.zeros([10,2])
        
        
        
        try:
            # lmk1_temp =np.array(pd.read_excel(os.path.join(trainkppath2, lmkname1),header=None,index_col=None))
            # lmk1_temp=lmk1_temp[1:,1:].astype('float')
            # lmk1_temp = lmk1_temp[:, [1, 0]]*4
            # lmk2_temp =np.array(pd.read_excel(os.path.join(trainkppath2, lmkname2),header=None,index_col=None))
            # lmk2_temp=lmk2_temp[1:,1:].astype('float')
            # lmk2_temp = lmk2_temp[:, [1, 0]]*4
            
            
            lmk1_temp =pd.read_csv(os.path.join(trainkppath2, lmkname1))
            lmk1_temp=np.array(lmk1_temp)
            lmk1_temp = lmk1_temp[:, [2, 1]]*4
            lmk2_temp =pd.read_csv(os.path.join(trainkppath2, lmkname2))
            lmk2_temp=np.array(lmk2_temp)
            lmk2_temp = lmk2_temp[:, [2, 1]]*4
            
        except:
            lmk1_temp=np.zeros([10,2])
            lmk2_temp=np.zeros([10,2])
        lmk1=np.concatenate((lmk1,lmk1_temp),0)
        lmk2=np.concatenate((lmk2,lmk2_temp),0)
        
        
        try:
            # lmk1_temp =np.array(pd.read_excel(os.path.join(trainkppath3, lmkname1),header=None,index_col=None))
            lmk1_temp =pd.read_csv(os.path.join(trainkppath3, lmkname1))
            lmk1_temp=np.array(lmk1_temp)
            lmk1_temp = lmk1_temp[:, [2, 1]]*4
            # lmk2_temp =np.array(pd.read_excel(os.path.join(trainkppath3, lmkname2),header=None,index_col=None))
            lmk2_temp =pd.read_csv(os.path.join(trainkppath3, lmkname2))
            lmk2_temp=np.array(lmk2_temp)
            lmk2_temp = lmk2_temp[:, [2, 1]]*4
        except:
            lmk1_temp=np.zeros([10,2])
            lmk2_temp=np.zeros([10,2])
        lmk1=np.concatenate((lmk1,lmk1_temp),0)
        lmk2=np.concatenate((lmk2,lmk2_temp),0)
        
        try:
            # lmk1_temp =np.array(pd.read_excel(os.path.join(trainkppath4, lmkname1),header=None,index_col=None))
            lmk1_temp =pd.read_csv(os.path.join(trainkppath4, lmkname1))
            lmk1_temp=np.array(lmk1_temp)
            lmk1_temp = lmk1_temp[:, [2, 1]]*4
            # lmk2_temp =np.array(pd.read_excel(os.path.join(trainkppath4, lmkname2),header=None,index_col=None))
            lmk2_temp =pd.read_csv(os.path.join(trainkppath4, lmkname2))
            lmk2_temp=np.array(lmk2_temp)
            lmk2_temp = lmk2_temp[:, [2, 1]]*4
        except:
            lmk1_temp=np.zeros([10,2])
            lmk2_temp=np.zeros([10,2])
        lmk1=np.concatenate((lmk1,lmk1_temp),0)
        lmk2=np.concatenate((lmk2,lmk2_temp),0)
        idx=np.where(lmk2[:,0]!=0)[0]
        # print(idx)
        if idx.shape[0]>0:
            lmk1=lmk1[idx,:]
            lmk2=lmk2[idx,:]
        
            dis=np.sqrt(np.abs(np.expand_dims((lmk1**2).sum(1),1)+np.expand_dims((lmk1**2).sum(1),0)-2*np.dot(lmk1,lmk1.transpose((1,0)))))
            dis[np.where(dis<0.05)[0],np.where(dis<0.05)[1]]=1000000000
            dis_min=np.mean(np.min(dis,1))
            dis_mins.append(dis_min)
            dis=np.sqrt(np.abs(np.expand_dims((lmk2**2).sum(1),1)+np.expand_dims((lmk2**2).sum(1),0)-2*np.dot(lmk2,lmk2.transpose((1,0)))))
            dis[np.where(dis<0.05)[0],np.where(dis<0.05)[1]]=1000000000
            dis_min=np.mean(np.min(dis,1))
            dis_mins2.append(dis_min)
            
        else:
            continue
        colors=[]
        for i in range (lmk1.shape[0]):
            colors.append((255,0,0))
            cv2.drawMarker(img1,position=[int(lmk1[i,1]),int(lmk1[i,0])],color=colors[i],markerSize = 45, markerType=4,thickness=45)
        io.imsave(savepath+trainfilename.split('.')[0][0:-2]+'_1.jpg',np.uint8(img1))
        scio.savemat(savepath+trainfilename.split('.')[0][0:-2]+'_1.mat',{'data':lmk1})
        for i in range (lmk2.shape[0]):
            cv2.drawMarker(img2,position=[int(lmk2[i,1]),int(lmk2[i,0])],color=(0, 0, 255),markerSize = 45, markerType=4,thickness=45)
        io.imsave(savepath+trainfilename.split('.')[0][0:-2]+'_2.jpg',np.uint8(img2))
        scio.savemat(savepath+trainfilename.split('.')[0][0:-2]+'_2.mat',{'data':lmk2})
        # pdb.set_trace()
# print(np.mean(dis_mins))
# print(np.mean(dis_mins2))
pdb.set_trace()