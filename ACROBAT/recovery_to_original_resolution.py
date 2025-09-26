import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import cv2
csvpath="ACROBAT/data/preprocessing_val.xlsx"
csvpath2="ACROBAT_validation_annotated_kps.csv"
csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
############## "ACROBAT_validation_annotated_kps.csv" provides locations of keypoints in WSIs (original image and original resolution)
############## WSIs from ACROBAT were cropped, padded and finally downsampled to (512,512). Examples of the processed data were provided in 'ACROBAT/data/'
############## Examples of crop paras and pad paras are provided in "ACROBAT/data/preprocessing_val.xlsx"

##############this code shows how to recovery keypoints coordinates (provided in 'ACROBAT/data/') to original resolution

################load lmk1 and img1
lmkpath='ACROBAT/data/valid_after_affine_4096_to512/0_KI67_val.xlsx'
imgpath='ACROBAT/data/valid_after_affine_4096_to512/0_KI67_val.jpg'
lmk1 =np.array(pd.read_excel(lmkpath,header=None,index_col=None))
lmk1=lmk1[1:,1:].astype('float')
lmk1 = lmk1[:, [1, 0]]
lmk1 = np.pad(lmk1, ((0, 200 - len(lmk1)), (0, 0)), "constant")
resolution=csvdata2[np.where(csvdata2[:,1]==int(lmkpath2.split('_')[0]))[0],7][0]*np.ones([200,1])
rotation=np.zeros([200,1])
crop_para=np.pad(csvdata[np.where(csvdata[:,0]==imgpath)[0],1:5],((0,200-1),(0,0)),'edge')
pad_para=np.pad(csvdata[np.where(csvdata[:,0]==imgpath)[0],6:10],((0,200-1),(0,0)),'edge')
lmk1=np.concatenate((lmk1,resolution,crop_para,rotation,pad_para),1)
img1=np.array(Image.open(imgpath))


################load lmk2 and img2
lmkpath2='ACROBAT/data/valid_after_affine_4096_to512/0_HE_val.xlsx'
imgpath2='ACROBAT/data/valid_after_affine_4096_to512/0_HE_val.jpg'   ####shape(512,512)
lmk2 =np.array(pd.read_excel(lmkpath2,header=None,index_col=None))
lmk2=lmk2[1:,1:].astype('float')
lmk2 = lmk2[:, [1, 0]]
lmk2 = np.pad(lmk2, ((0, 200 - len(lmk2)), (0, 0)), "constant")
resolution=csvdata2[np.where(csvdata2[:,1]==int(lmkpath2.split('_')[0]))[0],8][0]*np.ones([200,1])
rotation=np.zeros([200,1])
crop_para=np.pad(csvdata[np.where(csvdata[:,0]==imgpath2)[0],1:5],((0,200-1),(0,0)),'edge')
pad_para=np.pad(csvdata[np.where(csvdata[:,0]==imgpath2)[0],6:10],((0,200-1),(0,0)),'edge')
lmk2=np.concatenate((lmk2,resolution,crop_para,rotation,pad_para),1)
img2=np.array(Image.open(imgpath2))


# # # # #######################perform registration
# # # # lmk1---->warp----->warped_lmk
# # # # img2---->warp----->warped_img
# # # # #########################

######################resize keypoints to original resolution
warped_lmk[:,0]=(warped_lmk[:,0]/(512-1)*(lmk2[:,4]-lmk2[:,3]+lmk2[:,11]+lmk2[:,10])+lmk2[:,3]-1-lmk2[:,10])*lmk2[:,2]
warped_lmk[:,1]=(warped_lmk[:,1]/(512-1)*(lmk2[:,6]-lmk2[:,5]+lmk2[:,9]+lmk2[:,8])+lmk2[:,5]-1-lmk2[:,8])*lmk2[:,2]####### (to be submitted to the official challenge)
lmk2[:,0]=(lmk2[:,0]/(512-1)*(lmk2[:,4]-lmk2[:,3]+lmk2[:,11]+lmk2[:,10])+lmk2[:,3]-1-lmk2[:,10])*lmk2[:,2]
lmk2[:,1]=(lmk2[:,1]/(512-1)*(lmk2[:,6]-lmk2[:,5]+lmk2[:,9]+lmk2[:,8])+lmk2[:,5]-1-lmk2[:,8])*lmk2[:,2]
######################resize img to original resolution
warped_img=cv2.resize(warped_img,((lmk2[:,4]-lmk2[:,3]+lmk2[:,11]+lmk2[:,10]),(lmk2[:,6]-lmk2[:,5]+lmk2[:,9]+lmk2[:,8])))
######################calculate distance at original resolution
lmk2=lmk2[:,0:2]
warped_lmk=warped_lmk[:,0:2]
lmk_dist = np.sqrt(np.sum(np.square(warped_lmk - lmk2), axis=-1))