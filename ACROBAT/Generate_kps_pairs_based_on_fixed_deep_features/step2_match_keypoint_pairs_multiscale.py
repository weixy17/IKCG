import torch
import argparse
import os
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pdb
import sys 
sys.path.append('..') 
sys.path.append('../..') 
import time
import numpy as np
from torch.backends import cudnn
import sys
import pprint
import random
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import cv2
import csv
import skimage.io, skimage.filters
from timeit import default_timer
import pdb
from skimage import io
import pandas as pd
from numpy import *
import skimage
import scipy.io as scio
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.transforms as transforms

img_size=1024
orbpath="./kps_ORB64s4_4096_to1024/"
csvfilepath='./kps_vgg16/'
precessed_files=os.listdir(csvfilepath)
savepath='./kps_vgg16_images/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
if not os.path.exists(csvfilepath):
    os.mkdir(csvfilepath)
data_path =  "/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
data_path_512 = "/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to512/"
data_path_2048 = "/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to2048/"
resnet101 = torchvision.models.vgg16(pretrained=True).to(device)
res_l5=nn.Sequential(*(list(resnet101.children())[:-1]))
res_l51=nn.Sequential(*(list(resnet101.children())[-1])[:-5])
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                fileList.append(fileName)
    return fileList
def LoadACROBAT():
    files=os.listdir(data_path)
    # pdb.set_trace()
    for file in files:
        if file.split('_')[1]!="HE":
            continue
        filepath2=os.path.join(data_path,file)
        im_temp2 = io.imread(filepath2, as_gray=True)
        im2=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
        if im2.max()<=1:
            im2=np.uint8(im2*255)
        filepath2=os.path.join(data_path_512,file.split('_')[0]+'_HE_train.jpg')
        im_temp2 = io.imread(filepath2, as_gray=True)
        im2_512=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
        if im2_512.max()<=1:
            im2_512=np.uint8(im2_512*255)
        filepath2=os.path.join(data_path_2048,file.split('_')[0]+'_HE_train.jpg')
        im_temp2 = io.imread(filepath2, as_gray=True)
        im2_2048=np.concatenate((np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0),np.expand_dims(im_temp2,0)),0)
        if im2_2048.max()<=1:
            im2_2048=np.uint8(im2_2048*255)
        lmk2_orb = pd.read_csv(orbpath+file.split('_')[0]+'_HE_train.csv')
        lmk2_orb = np.array(lmk2_orb)
        lmk2_orb = int32(lmk2_orb[:, [2, 1]]).reshape(-1,2)
        
        
        file1s=test_findfile(data_path,'.jpg',file.split('_')[0]+'_')
        file1s=[file1_temp for file1_temp in file1s if file1_temp!=file.split('_')[0]+'_HE_train.jpg']
        valid_count=-1
        for count,file1 in enumerate(file1s):
            if file1.split('.')[0]+'_1.csv' in precessed_files:
                continue
            valid_count=valid_count+1
            im_temp1 = io.imread(os.path.join(data_path,file1), as_gray=True)
            im1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
            if im1.max()<=1:
                im1=np.uint8(im1*255)
            filepath=os.path.join(data_path_512,file1)
            im_temp1 = io.imread(filepath, as_gray=True)
            im1_512=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
            if im1_512.max()<=1:
                im1_512=np.uint8(im1_512*255)
            filepath=os.path.join(data_path_2048,file1)
            im_temp1 = io.imread(filepath, as_gray=True)
            im1_2048=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
            if im1_2048.max()<=1:
                im1_2048=np.uint8(im1_2048*255)
            lmk1_orb = pd.read_csv(orbpath+file1[0:-4]+'.csv')
            lmk1_orb = np.array(lmk1_orb)
            lmk1_orb = int32(lmk1_orb[:, [2, 1]]).reshape(-1,2)
            if valid_count==0:
                desc2,kp2 = train(np.expand_dims(im1,axis=0),np.expand_dims(im1_512,axis=0),np.expand_dims(im1_2048,axis=0), np.expand_dims(im2,axis=0),np.expand_dims(im2_512,axis=0),np.expand_dims(im2_2048,axis=0),lmk1_orb,lmk2_orb,file1.split('.')[0])
            else:
                train2(np.expand_dims(im1,axis=0),np.expand_dims(im1_512,axis=0),np.expand_dims(im1_2048,axis=0), np.expand_dims(im2,axis=0),np.expand_dims(im2_512,axis=0),np.expand_dims(im2_2048,axis=0),lmk1_orb,lmk2_orb,file1.split('.')[0],desc2,kp2)
        if valid_count>=0:
            del desc2,kp2
    return 0

def addsiftkp(coords, kp,row,col):
    len = np.shape(kp)[0]
    for i in range(len):
        siftco = [int(round(kp[i].pt[1]))+step_range*row, int(round(kp[i].pt[0]))+step_range*col]
        if siftco not in coords:
            coords.append(siftco)
    num=np.shape(coords)[0]
    return coords


def train(img1,img1_256, img1_1024, img2, img2_256,img2_1024,lmk1,lmk2,steps):
    print(steps)
    img1s,img2s = (torch.from_numpy(img1).float()).to(device), (torch.from_numpy(img2).float()).to(device)
    img1s_256,img2s_256 = (torch.from_numpy(img1_256).float()).to(device), (torch.from_numpy(img2_256).float()).to(device)
    img1s_1024,img2s_1024 = (torch.from_numpy(img1_1024).float()).to(device), (torch.from_numpy(img2_1024).float()).to(device)
    im1=appendimages(img1[0,0,:,:],img2[0,0,:,:])
    desc2,kp2 = association(img1s,img2s,img1s_256,img2s_256,img1s_1024,img2s_1024,lmk1,lmk2,steps,im1)#N,H*W,1
    print('Finished!')
    return desc2,kp2
def train2(img1,img1_256, img1_1024, img2, img2_256,img2_1024,lmk1,lmk2,steps,desc2,kp2):
    print(steps)
    img1s,img2s = (torch.from_numpy(img1).float()).to(device), (torch.from_numpy(img2).float()).to(device)
    img1s_256,img2s_256 = (torch.from_numpy(img1_256).float()).to(device), (torch.from_numpy(img2_256).float()).to(device)
    img1s_1024,img2s_1024 = (torch.from_numpy(img1_1024).float()).to(device), (torch.from_numpy(img2_1024).float()).to(device)
    im1=appendimages(img1[0,0,:,:],img2[0,0,:,:])
    _=association2(img1s,img1s_256,img1s_1024,lmk1,steps,im1,desc2,kp2)
    print('Finished!')
    return 0
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
    

from loss_multiscale_new import AssociationLoss
FeatAssociationLoss = AssociationLoss(metric='cos', 
                img_size=img_size, print_info=False)
def association(img1,img2,img1_256,img2_256,img1_1024,img2_1024,lkp1,lkp2,step,im1):
    arange6=30*2
    arange5=25*2
    arange4=20*2
    arange3=15*2
    arange2=10*2
    img1,img2=img1/255,img2/255
    img1_256,img2_256=img1_256/255,img2_256/255
    img1_1024,img2_1024=img1_1024/255,img2_1024/255
    
    N,C,H,W=img1.shape[0],img1.shape[1],img1.shape[2],img1.shape[3]

    list_kp1=lkp1.tolist()
    list_kp2=lkp2.tolist()
    count_kps=-1
    coor1s,coor2s,masks,mask_accurates,mask_errors=[],[],[],[],[]
    
    print('features extracting')
    time0=time.time()
    patch_img1s=torch.zeros((5,3,224,224)).to(device)
    patch_img1s_256=torch.zeros((5,3,224,224)).to(device)
    patch_img1s_1024=torch.zeros((5,3,224,224)).to(device)
    patch_img2s=torch.zeros((5,3,224,224)).to(device)
    patch_img2s_256=torch.zeros((5,3,224,224)).to(device)
    patch_img2s_1024=torch.zeros((5,3,224,224)).to(device)
    for i in range(len(list_kp1)):
        kp1_x1,kp1_y1=list_kp1[i][1],list_kp1[i][0]
        patch_img1s[0,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),1024),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),1024)],(224,224),mode='bilinear')
        patch_img1s[1,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),1024),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),1024)],(224,224),mode='bilinear')
        patch_img1s[2,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),1024),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),1024)],(224,224),mode='bilinear')
        patch_img1s[3,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),1024),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),1024)],(224,224),mode='bilinear')
        patch_img1s[4,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),1024),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),1024)],(224,224),mode='bilinear')
        patch_img1s_256[0,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(1024/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[1,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(1024/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[2,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(1024/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[3,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(1024/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[4,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(1024/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_1024[0,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),1024*2),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[1,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),1024*2),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[2,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),1024*2),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[3,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),1024*2),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[4,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),1024*2),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),1024*2)],(224,224),mode='bilinear')
        patch_img1s=(patch_img1s-torch.mean(patch_img1s,dim=(2,3),keepdim=True))/(torch.std(patch_img1s,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_256=(patch_img1s_256-torch.mean(patch_img1s_256,dim=(2,3),keepdim=True))/(torch.std(patch_img1s_256,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_1024=(patch_img1s_1024-torch.mean(patch_img1s_1024,dim=(2,3),keepdim=True))/(torch.std(patch_img1s_1024,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_all=torch.cat((patch_img1s.unsqueeze(1),transforms.functional.rotate(patch_img1s,90).unsqueeze(1),transforms.functional.rotate(patch_img1s,180).unsqueeze(1),transforms.functional.rotate(patch_img1s,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s,45).unsqueeze(1),transforms.functional.rotate(patch_img1s,135).unsqueeze(1),transforms.functional.rotate(patch_img1s,225).unsqueeze(1),transforms.functional.rotate(patch_img1s,315).unsqueeze(1)),1)
        patch_img1s_256_all=torch.cat((patch_img1s_256.unsqueeze(1),transforms.functional.rotate(patch_img1s_256,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s_256,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,315).unsqueeze(1)),1)
        patch_img1s_1024_all=torch.cat((patch_img1s_1024.unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s_1024,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,315).unsqueeze(1)),1)###(5,8,3,224,224)
        patch_img1s_all_multi=torch.cat((patch_img1s_all.view(-1,3,224,224),patch_img1s_256_all.view(-1,3,224,224),patch_img1s_1024_all.view(-1,3,224,224)),0)####(120,3,224,224)#######8+8+8+8+8=40 40*3=120
        with torch.no_grad():
            count_kps=count_kps+1
            if count_kps==0:
                desc1=res_l51(res_l5(patch_img1s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)#(5,4096*8*3,1)
            else:
                desc1=torch.cat((desc1,res_l51(res_l5(patch_img1s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)),2)#(5,4096*8,i)
    count_kps=-1
    del patch_img1s,patch_img1s_256,patch_img1s_1024
    for i in range(len(list_kp2)):
        kp2_x2,kp2_y2=list_kp2[i][1],list_kp2[i][0]
        patch_img2s[0,:,:,:]=F.upsample(img2[:,:,max(int((kp2_y2-arange2)),0):min(int((kp2_y2+arange2)),1024),max(int((kp2_x2-arange2)),0):min(int((kp2_x2+arange2)),1024)],(224,224),mode='bilinear')
        patch_img2s[1,:,:,:]=F.upsample(img2[:,:,max(int((kp2_y2-arange3)),0):min(int((kp2_y2+arange3)),1024),max(int((kp2_x2-arange3)),0):min(int((kp2_x2+arange3)),1024)],(224,224),mode='bilinear')
        patch_img2s[2,:,:,:]=F.upsample(img2[:,:,max(int((kp2_y2-arange4)),0):min(int((kp2_y2+arange4)),1024),max(int((kp2_x2-arange4)),0):min(int((kp2_x2+arange4)),1024)],(224,224),mode='bilinear')
        patch_img2s[3,:,:,:]=F.upsample(img2[:,:,max(int((kp2_y2-arange5)),0):min(int((kp2_y2+arange5)),1024),max(int((kp2_x2-arange5)),0):min(int((kp2_x2+arange5)),1024)],(224,224),mode='bilinear')
        patch_img2s[4,:,:,:]=F.upsample(img2[:,:,max(int((kp2_y2-arange6)),0):min(int((kp2_y2+arange6)),1024),max(int((kp2_x2-arange6)),0):min(int((kp2_x2+arange6)),1024)],(224,224),mode='bilinear')
        patch_img2s_256[0,:,:,:]=F.upsample(img2_256[:,:,max(int((kp2_y2/2-arange2)),0):min(int((kp2_y2/2+arange2)),int(1024/2)),max(int((kp2_x2/2-arange2)),0):min(int((kp2_x2/2+arange2)),int(1024/2))],(224,224),mode='bilinear')
        patch_img2s_256[1,:,:,:]=F.upsample(img2_256[:,:,max(int((kp2_y2/2-arange3)),0):min(int((kp2_y2/2+arange3)),int(1024/2)),max(int((kp2_x2/2-arange3)),0):min(int((kp2_x2/2+arange3)),int(1024/2))],(224,224),mode='bilinear')
        patch_img2s_256[2,:,:,:]=F.upsample(img2_256[:,:,max(int((kp2_y2/2-arange4)),0):min(int((kp2_y2/2+arange4)),int(1024/2)),max(int((kp2_x2/2-arange4)),0):min(int((kp2_x2/2+arange4)),int(1024/2))],(224,224),mode='bilinear')
        patch_img2s_256[3,:,:,:]=F.upsample(img2_256[:,:,max(int((kp2_y2/2-arange5)),0):min(int((kp2_y2/2+arange5)),int(1024/2)),max(int((kp2_x2/2-arange5)),0):min(int((kp2_x2/2+arange5)),int(1024/2))],(224,224),mode='bilinear')
        patch_img2s_256[4,:,:,:]=F.upsample(img2_256[:,:,max(int((kp2_y2/2-arange6)),0):min(int((kp2_y2/2+arange6)),int(1024/2)),max(int((kp2_x2/2-arange6)),0):min(int((kp2_x2/2+arange6)),int(1024/2))],(224,224),mode='bilinear')
        patch_img2s_1024[0,:,:,:]=F.upsample(img2_1024[:,:,max(int((kp2_y2*2-arange2)),0):min(int((kp2_y2*2+arange2)),1024*2),max(int((kp2_x2*2-arange2)),0):min(int((kp2_x2*2+arange2)),1024*2)],(224,224),mode='bilinear')
        patch_img2s_1024[1,:,:,:]=F.upsample(img2_1024[:,:,max(int((kp2_y2*2-arange3)),0):min(int((kp2_y2*2+arange3)),1024*2),max(int((kp2_x2*2-arange3)),0):min(int((kp2_x2*2+arange3)),1024*2)],(224,224),mode='bilinear')
        patch_img2s_1024[2,:,:,:]=F.upsample(img2_1024[:,:,max(int((kp2_y2*2-arange4)),0):min(int((kp2_y2*2+arange4)),1024*2),max(int((kp2_x2*2-arange4)),0):min(int((kp2_x2*2+arange4)),1024*2)],(224,224),mode='bilinear')
        patch_img2s_1024[3,:,:,:]=F.upsample(img2_1024[:,:,max(int((kp2_y2*2-arange5)),0):min(int((kp2_y2*2+arange5)),1024*2),max(int((kp2_x2*2-arange5)),0):min(int((kp2_x2*2+arange5)),1024*2)],(224,224),mode='bilinear')
        patch_img2s_1024[4,:,:,:]=F.upsample(img2_1024[:,:,max(int((kp2_y2*2-arange6)),0):min(int((kp2_y2*2+arange6)),1024*2),max(int((kp2_x2*2-arange6)),0):min(int((kp2_x2*2+arange6)),1024*2)],(224,224),mode='bilinear')
        patch_img2s=(patch_img2s-torch.mean(patch_img2s,dim=(2,3),keepdim=True))/(torch.std(patch_img2s,dim=(2,3),keepdim=True)+1e-10)
        patch_img2s_256=(patch_img2s_256-torch.mean(patch_img2s_256,dim=(2,3),keepdim=True))/(torch.std(patch_img2s_256,dim=(2,3),keepdim=True)+1e-10)
        patch_img2s_1024=(patch_img2s_1024-torch.mean(patch_img2s_1024,dim=(2,3),keepdim=True))/(torch.std(patch_img2s_1024,dim=(2,3),keepdim=True)+1e-10)
        
        patch_img2s_all=torch.cat((patch_img2s.unsqueeze(1),transforms.functional.rotate(patch_img2s,90).unsqueeze(1),transforms.functional.rotate(patch_img2s,180).unsqueeze(1),transforms.functional.rotate(patch_img2s,270).unsqueeze(1),
            transforms.functional.rotate(patch_img2s,45).unsqueeze(1),transforms.functional.rotate(patch_img2s,135).unsqueeze(1),transforms.functional.rotate(patch_img2s,225).unsqueeze(1),transforms.functional.rotate(patch_img2s,315).unsqueeze(1)),1)
        patch_img2s_256_all=torch.cat((patch_img2s_256.unsqueeze(1),transforms.functional.rotate(patch_img2s_256,90).unsqueeze(1),transforms.functional.rotate(patch_img2s_256,180).unsqueeze(1),transforms.functional.rotate(patch_img2s_256,270).unsqueeze(1),
            transforms.functional.rotate(patch_img2s_256,45).unsqueeze(1),transforms.functional.rotate(patch_img2s_256,135).unsqueeze(1),transforms.functional.rotate(patch_img2s_256,225).unsqueeze(1),transforms.functional.rotate(patch_img2s_256,315).unsqueeze(1)),1)
        patch_img2s_1024_all=torch.cat((patch_img2s_1024.unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,90).unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,180).unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,270).unsqueeze(1),
            transforms.functional.rotate(patch_img2s_1024,45).unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,135).unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,225).unsqueeze(1),transforms.functional.rotate(patch_img2s_1024,315).unsqueeze(1)),1)###(5,8,3,224,224)
        patch_img2s_all_multi=torch.cat((patch_img2s_all.view(-1,3,224,224),patch_img2s_256_all.view(-1,3,224,224),patch_img2s_1024_all.view(-1,3,224,224)),0)####8+8+8+8+8=40 40*3=120
        with torch.no_grad():
            count_kps=count_kps+1
            if count_kps==0:
                desc2=res_l51(res_l5(patch_img2s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)#(5,4096*8*3,1)
            else:
                desc2=torch.cat((desc2,res_l51(res_l5(patch_img2s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)),2)#(5,4096*8,i)
    desc1=desc1.unsqueeze(2)#(5,4096*8,1,K)
    print(desc1.shape)
    desc2=desc2.unsqueeze(2)#(5,4096*8,1,K)
    kp1=torch.tensor(list_kp1).to(device).unsqueeze(0).unsqueeze(0).float()
    kp2=torch.tensor(list_kp2).to(device).unsqueeze(0).unsqueeze(0).float()
    time1=time.time()
    print('{}s'.format(time1-time0))
    print('pairing')
    del patch_img2s,patch_img2s_256,patch_img2s_1024
    indices, max_indices,mid_indices,mask,_,sim_mat_12,_,_, _, _,_,_,_,_,_= FeatAssociationLoss(desc1, desc2,kp1,kp2)#[:,1000:,:,:]
    del desc1

    if mask.size()[0]>0:
        coor1_12=torch.index_select(kp1.squeeze(),0,mask.squeeze())#.squeeze()#K,2
        mid_indices_valid=torch.index_select(mid_indices,1,mask.squeeze())#N,K,1
        coor2_12=torch.index_select(kp2.squeeze(),0,mid_indices_valid.squeeze())
        weights_pre=torch.index_select(sim_mat_12,1,mask.squeeze())###(N,K,200)
        weights=torch.gather(weights_pre,2,mid_indices_valid).squeeze(0)
        coor1=torch.cat((coor1_12,weights),1).detach().cpu().numpy().tolist()#####weighted
        coor2=torch.cat((coor2_12,weights),1).detach().cpu().numpy().tolist()#####weighted
        time2=time.time()
        print('{}s'.format(time2-time1))
        if len(coor1)>0:
            plt.figure()
            plt.imshow(im1)
            plt.title(str(step)+'_key_points='+str(len(coor1)))
            plt.plot([np.array(coor1)[:,1],np.array(coor2)[:,1]+H],[np.array(coor1)[:,0],np.array(coor2)[:,0]], '#FF0033',linewidth=0.5)
            plt.savefig(savepath+str(step)+'_key_points='+str(len(coor1))+'.jpg', dpi=600)
            plt.close()
            name = ['X', 'Y','W']
            lmk1=np.asarray(coor1)
            lmk1=lmk1[:,[1,0,2]]#.transpose(0,1)
            lmk2=np.asarray(coor2)#.transpose(0,1)
            lmk2=lmk2[:,[1,0,2]]
            outlmk1 = pd.DataFrame(columns=name, data=lmk1)
            outlmk1.to_csv(csvfilepath+str(step)+'_1.csv')
            outlmk2 = pd.DataFrame(columns=name, data=lmk2)
            outlmk2.to_csv(csvfilepath+str(step)+'_2.csv')
    return desc2,kp2
    
    


def association2(img1,img1_256,img1_1024,lkp1,step,im1,desc2,kp2):
    arange6=30*2
    arange5=25*2
    arange4=20*2
    arange3=15*2
    arange2=10*2
    img1=img1/255
    img1_256=img1_256/255
    img1_1024=img1_1024/255
    
    N,C,H,W=img1.shape[0],img1.shape[1],img1.shape[2],img1.shape[3]

    list_kp1=lkp1.tolist()
    count_kps=-1
    coor1s,coor2s,masks,mask_accurates,mask_errors=[],[],[],[],[]

    print('features extracting')
    time0=time.time()
    patch_img1s=torch.zeros((5,3,224,224)).to(device)
    patch_img1s_256=torch.zeros((5,3,224,224)).to(device)
    patch_img1s_1024=torch.zeros((5,3,224,224)).to(device)
    patch_img2s=torch.zeros((5,3,224,224)).to(device)
    patch_img2s_256=torch.zeros((5,3,224,224)).to(device)
    patch_img2s_1024=torch.zeros((5,3,224,224)).to(device)
    for i in range(len(list_kp1)):
        kp1_x1,kp1_y1=list_kp1[i][1],list_kp1[i][0]
        patch_img1s[0,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),1024),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),1024)],(224,224),mode='bilinear')
        patch_img1s[1,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),1024),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),1024)],(224,224),mode='bilinear')
        patch_img1s[2,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),1024),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),1024)],(224,224),mode='bilinear')
        patch_img1s[3,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),1024),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),1024)],(224,224),mode='bilinear')
        patch_img1s[4,:,:,:]=F.upsample(img1[:,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),1024),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),1024)],(224,224),mode='bilinear')
        patch_img1s_256[0,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(1024/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[1,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(1024/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[2,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(1024/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[3,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(1024/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_256[4,:,:,:]=F.upsample(img1_256[:,:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(1024/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(1024/2))],(224,224),mode='bilinear')
        patch_img1s_1024[0,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),1024*2),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[1,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),1024*2),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[2,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),1024*2),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[3,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),1024*2),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),1024*2)],(224,224),mode='bilinear')
        patch_img1s_1024[4,:,:,:]=F.upsample(img1_1024[:,:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),1024*2),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),1024*2)],(224,224),mode='bilinear')
        patch_img1s=(patch_img1s-torch.mean(patch_img1s,dim=(2,3),keepdim=True))/(torch.std(patch_img1s,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_256=(patch_img1s_256-torch.mean(patch_img1s_256,dim=(2,3),keepdim=True))/(torch.std(patch_img1s_256,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_1024=(patch_img1s_1024-torch.mean(patch_img1s_1024,dim=(2,3),keepdim=True))/(torch.std(patch_img1s_1024,dim=(2,3),keepdim=True)+1e-10)
        patch_img1s_all=torch.cat((patch_img1s.unsqueeze(1),transforms.functional.rotate(patch_img1s,90).unsqueeze(1),transforms.functional.rotate(patch_img1s,180).unsqueeze(1),transforms.functional.rotate(patch_img1s,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s,45).unsqueeze(1),transforms.functional.rotate(patch_img1s,135).unsqueeze(1),transforms.functional.rotate(patch_img1s,225).unsqueeze(1),transforms.functional.rotate(patch_img1s,315).unsqueeze(1)),1)
        patch_img1s_256_all=torch.cat((patch_img1s_256.unsqueeze(1),transforms.functional.rotate(patch_img1s_256,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s_256,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,315).unsqueeze(1)),1)
        patch_img1s_1024_all=torch.cat((patch_img1s_1024.unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,270).unsqueeze(1),
            transforms.functional.rotate(patch_img1s_1024,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,315).unsqueeze(1)),1)###(5,8,3,224,224)
        patch_img1s_all_multi=torch.cat((patch_img1s_all.view(-1,3,224,224),patch_img1s_256_all.view(-1,3,224,224),patch_img1s_1024_all.view(-1,3,224,224)),0)####(120,3,224,224)#######8+8+8+8+8=40 40*3=120
        with torch.no_grad():
            count_kps=count_kps+1
            if count_kps==0:
                desc1=res_l51(res_l5(patch_img1s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)#(5,4096*8*3,1)
            else:
                desc1=torch.cat((desc1,res_l51(res_l5(patch_img1s_all_multi.view(-1,3,224,224)).view(-1,25088)).view(3,5,8,4096).permute((1,0,2,3)).reshape(5,-1).unsqueeze(2)),2)#(5,4096*8,i)
    count_kps=-1
    del patch_img1s,patch_img1s_256,patch_img1s_1024
    desc1=desc1.unsqueeze(2)#(5,4096*8,1,K)
    print(desc1.shape)
    kp1=torch.tensor(list_kp1).to(device).unsqueeze(0).unsqueeze(0).float()
    time1=time.time()
    print('{}s'.format(time1-time0))
    print('pairing')
    indices, max_indices,mid_indices,mask,_,sim_mat_12,_,_, _, _,_,_,_,_,_= FeatAssociationLoss(desc1, desc2,kp1,kp2)#[:,1000:,:,:]
    del desc1
    
    if mask.size()[0]>0:
        coor1_12=torch.index_select(kp1.squeeze(),0,mask.squeeze())#.squeeze()#K,2
        mid_indices_valid=torch.index_select(mid_indices,1,mask.squeeze())#N,K,1
        coor2_12=torch.index_select(kp2.squeeze(),0,mid_indices_valid.squeeze())
        weights_pre=torch.index_select(sim_mat_12,1,mask.squeeze())###(N,K,200)
        weights=torch.gather(weights_pre,2,mid_indices_valid).squeeze(0)
        coor1=torch.cat((coor1_12,weights),1).detach().cpu().numpy().tolist()#####weighted
        coor2=torch.cat((coor2_12,weights),1).detach().cpu().numpy().tolist()#####weighted
        time2=time.time()
        print('{}s'.format(time2-time1))
        if len(coor1)>0:
            plt.figure()
            plt.imshow(im1)
            plt.title(str(step)+'_key_points='+str(len(coor1)))
            plt.plot([np.array(coor1)[:,1],np.array(coor2)[:,1]+H],[np.array(coor1)[:,0],np.array(coor2)[:,0]], '#FF0033',linewidth=0.5)
            plt.savefig(savepath+str(step)+'_key_points='+str(len(coor1))+'.jpg', dpi=600)
            plt.close()
            name = ['X', 'Y','W']
            lmk1=np.asarray(coor1)
            lmk1=lmk1[:,[1,0,2]]#.transpose(0,1)
            lmk2=np.asarray(coor2)#.transpose(0,1)
            lmk2=lmk2[:,[1,0,2]]
            outlmk1 = pd.DataFrame(columns=name, data=lmk1)
            outlmk1.to_csv(csvfilepath+str(step)+'_1.csv')
            outlmk2 = pd.DataFrame(columns=name, data=lmk2)
            outlmk2.to_csv(csvfilepath+str(step)+'_2.csv')
    return 0


if __name__ == '__main__':
    cudnn.benchmark = True 
    LoadACROBAT()
