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


def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                # fileList.append(os.path.join(root, fileName))
                fileList.append(fileName)
    return fileList


def LoadACROBAT_baseline512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="./data/train_after_affine_4096_to512/"
    validimgpath="./data/valid_after_affine_4096_to512/"
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
            dataset[filename1] = img1
            dataset[filename2] = img2
            train_pairs.append((filename1, filename2))
            train_pairs.append((filename2, filename1))
    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs

def LoadACROBAT_F_KCG_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="./data/train_after_affine_4096_to512/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    validimgpath="./data/valid_after_affine_4096_to512/"
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
    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs

def LoadACROBAT_L_KCG_kps_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_ORB64s4_4096_to1024/"
    trainimgpath="./data/train_after_affine_4096_to1024/"
    trainimgpath512="./data/train_after_affine_4096_to512/"
    trainimgpath2048="./data/train_after_affine_4096_to2048/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    validimgpath="./data/valid_after_affine_4096_to512/"
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
                lmk1 = np.pad(lmk1,((0, 1200 -len(lmk1)), (0, 0)), "constant")
                lmk2 = pd.read_csv(os.path.join(trainkppath, lmkname2))
                lmk2 = np.array(lmk2)
                lmk2 = lmk2[:, [2, 1]]
                lmk2 = np.pad(lmk2,((0, 1200 -len(lmk2)), (0, 0)), "constant")

            lmk_obtained1=lmk1
            lmk1_orb = np.array(pd.read_csv(orbpath+flmk_orb1))
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk1_orb**2).sum(1),1)+np.expand_dims((lmk_obtained1**2).sum(1),0)-2*np.dot(lmk1_orb,lmk_obtained1.transpose((1,0))))
            lmk_orb_valid1=lmk1_orb[np.where(dis_temp.min(1)>5)[0],:]
            lmk_orb_valid1, returned_index=np.unique(lmk_orb_valid1, return_index=True,axis=0)
            dataset[flmk_orb1] = lmk_orb_valid1
            
            lmk_obtained2=lmk2
            lmk2_orb = np.array(pd.read_csv(orbpath+flmk_orb2))
            lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)################imgsize1024
            dis_temp=np.sqrt(np.expand_dims((lmk2_orb**2).sum(1),1)+np.expand_dims((lmk_obtained2**2).sum(1),0)-2*np.dot(lmk2_orb,lmk_obtained2.transpose((1,0))))
            lmk_orb_valid2=lmk2_orb[np.where(dis_temp.min(1)>5)[0],:]
            lmk_orb_valid2, returned_index=np.unique(lmk_orb_valid2, return_index=True,axis=0)
            dataset[flmk_orb2] = lmk_orb_valid2
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs


def LoadACROBAT_L_KCG_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="./data/train_after_affine_4096_to512/"
    trainkppath="..."#########keypoint pairs matched based on learnable deep features from converged F-KCG
    validimgpath="./data/valid_after_affine_4096_to512/"
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
    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs

def LoadACROBAT_LF_KCG_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="./data/train_after_affine_4096_to512/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    trainkppath2="..."#########keypoint pairs matched based on learnable deep features from converged F-KCG
    validimgpath="./data/valid_after_affine_4096_to512/"
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
            dataset[lmkname1] = lmk1/2
            dataset[lmkname2] = lmk2/2
            dataset[lmkname1_2] = lmk1_2/2
            dataset[lmkname2_2] = lmk2_2/2
            train_pairs.append((filename1, filename2,lmkname1,lmkname2,lmkname1_2,lmkname2_2))
    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_KCG_kps_ite1_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_ORB64s4_4096_to1024/"
    trainimgpath="./data/train_after_affine_4096_to1024/"
    trainimgpath512="./data/train_after_affine_4096_to512/"
    trainimgpath2048="./data/train_after_affine_4096_to2048/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    trainkppath2="..."#########keypoint pairs matched based on learnable deep features from converged F-KCG
    validimgpath="./data/valid_after_affine_4096_to512/"
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
            train_pairs.append((filename1, filename2,filename1_512,filename2_512,filename1_2048, filename2_2048,flmk_orb1,flmk_orb2))

    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_IKCG_1ite_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    
    trainimgpath="./data/train_after_affine_4096_to512/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    trainkppath2="..."#########keypoint pairs matched based on learnable deep features from converged F-KCG
    trainkppath3="..."#########keypoint pairs matched based on learnable deep features from converged KCG
    validimgpath="./data/valid_after_affine_4096_to512/"
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
    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs
def LoadACROBAT_KCG_kps_ite2_multiscale512():
    dataset = {}
    train_pairs = []
    valid_pairs = []
    orbpath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_ORB64s4_4096_to1024/"
    trainimgpath="./data/train_after_affine_4096_to1024/"
    trainimgpath512="./data/train_after_affine_4096_to512/"
    trainimgpath2048="./data/train_after_affine_4096_to2048/"
    trainkppath="IKCG/ACROBAT/Generate_kps_pairs_based_on_fixed_deep_features/kps_vgg16/"
    trainkppath2="..."#########keypoint pairs matched based on learnable deep features from converged F-KCG
    trainkppath3="..."#########keypoint pairs matched based on learnable deep features from converged IKCG-1 iteration
    validimgpath="./data/valid_after_affine_4096_to512/"
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

    csvpath="IKCG/Predicted_mask_for_val.xlsx"
    csvpath2="IKCG/ACROBAT_validation_annotated_kps.csv"
    csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
    csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
    for validfilename in validfilenames:
        stain_type=validfilename.split('_')[1]
        if stain_type=='HE':
            continue
        else:
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
    return dataset, train_pairs, valid_pairs
