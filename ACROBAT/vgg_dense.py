import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pdb
import sys 
sys.path.append('..') 
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
import hdf5storage
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
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
select_num=8
search_range=16
step_range=8
orb = cv2.ORB_create(select_num, scaleFactor=1.5,nlevels=3,edgeThreshold=2,patchSize=2)
image_size=512
# publicpath='101_vgg16_features_ORB16s8_fc30_1_0.04_0.004_0.95_non_zero_mean'
temp_name='_10_0.5_15_0_20_0.25_25_0_30_0.25_'
publicpath='vgg16_features_ORB16s8_fc6'+str(temp_name)+'rotate8_0.99_0.028_0.004_0.8_01norm'
publicpath_hand='vgg16_features_hand_fc6'+str(temp_name)+'rotate8_0.99_0.028_0.004_0.8_01norm'
publicpath_with_hand='vgg16_features_ORB16s8_fc6'+str(temp_name)+'rotate8_with_hand_0.99_0.028_0.004_0.8_01norm'
orbpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/ORB16s8/'
csvpath='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/'
csvfilepath=csvpath+publicpath+'/'
if not os.path.exists(csvfilepath):
    os.mkdir(csvfilepath)
pca = PCA(n_components=128)
def LoadANHIR(data_path = r"/home/wxy/Pixel-Level-Cycle-Association-main/Pixel-Level-Cycle-Association-main/data/"):##"/data3/gl/wxy_data/"

    prep_name1 = '512after_affine'
    prep_path1 = os.path.join(data_path, prep_name1)
    prep_path = prep_path1
    dataset = {}
    groups = {}
    train_groups = {}
    val_groups = {}
    train_pairs = []
    eval_pairs = []
    delete=144
    delete2=0-delete
    count=-1
    count_valid=-1
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            if row[5] == 'training':
                count=count+1
                # if num in [4,5,39]:
                    # continue
                if num <389 or num>=390:
                    continue
                # if count!=4:
                    # continue
                print('num={}'.format(num))
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
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
                    
                    
                    
                    dataset[fimg] = im_temp2
                    img1=im_temp2
                    
                    
                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
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
                    
                    
                    # pdb.set_trace()
                    dataset[fimg] = im_temp2
                    img2=im_temp2
                    
                    
                train_log = train(np.expand_dims(img1,axis=0), np.expand_dims(img2,axis=0),num)
            elif row[5] == 'evaluation':
                count_valid=count_valid+1
                if count_valid>6:
                    continue
                fimg = str(num)+'_1.jpg'
                flmk = str(num)+'_1.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp1=im_temp1[delete:delete2,delete:delete2]
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

                fimg = str(num) + '_2.jpg'
                flmk = str(num) + '_2.csv'
                if fimg not in dataset:
                    group = fimg.split("_")[0]
                    if group not in groups:
                        groups[group] = []
                    if group not in val_groups:
                        val_groups[group] = []
                    im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
                    im_temp1=im_temp1[delete:delete2,delete:delete2]
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
    return dataset, groups, train_groups, val_groups

def addsiftkp(coords, kp,row,col):
    len = np.shape(kp)[0]
    ## init , raw 0 is 0, 这样后面匹配到第一个点的匹配不会被当成不匹配而消去
    # coords.append([0, 0])
    for i in range(len):
        siftco = [int(round(kp[i].pt[1]))+step_range*row, int(round(kp[i].pt[0]))+step_range*col]
        if siftco not in coords:
            coords.append(siftco)
    num=np.shape(coords)[0]
    # for i in range (select_num -num):
        # coords.append([0, 0])
    return coords


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with specified model parameters',
                        default=None, type=str)
    parser.add_argument('--resume', dest='resume',
                        help='initialize with saved solver status',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--local_rank', dest='local_rank',
                        help='optional local rank',
                        default=0, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch size',
                        default=1, type=int)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def init_net_D(args, state_dict=None):
    net_D = FCDiscriminator(cfg.DATASET.NUM_CLASSES)

    if args.distributed:
        net_D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_D)

    if cfg.MODEL.DOMAIN_BN:
        net_D = DomainBN.convert_domain_batchnorm(net_D, num_domains=2)

    if state_dict is not None:
        try:
            net_D.load_state_dict(state_dict)
        except:
            net_D = DomainBN.convert_domain_batchnorm(net_D, num_domains=2)
            net_D.load_state_dict(state_dict)

    if cfg.TRAIN.FREEZE_BN:
        net_D.apply(freeze_BN)

    if torch.cuda.is_available():
        net_D.cuda()

    if args.distributed:
        net_D = DistributedDataParallel(net_D, device_ids=[args.gpu])
    else:
        net_D = torch.nn.DataParallel(net_D)

    return net_D

def train(img1, img2,steps):
    # resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
    resnet101 = torchvision.models.vgg16(pretrained=True).to(device)
    for k in range(1):
        shape=img1.shape
        img1s,img2s = (torch.from_numpy(img1).float()).to(device), (torch.from_numpy(img2).float()).to(device)
        # lmk1,lmk2=(torch.from_numpy(lmk1).float()).to(device),(torch.from_numpy(lmk2).float()).to(device)
        # lmk1_hand,lmk2_hand=(torch.from_numpy(lmk1_hand).float()).to(device),(torch.from_numpy(lmk2_hand).float()).to(device)
        # N,C,H,W=img1s.shape[0],img1s.shape[1],img1s.shape[2],img1s.shape[3]
        indices, max_indices,mid_indices,mask = association(img1s,img2s,resnet101,steps)#N,H*W,1
    print('Finished!')
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
    

def association(img1,img2,net,step):
    img1,img2=img1/255,img2/255
    res_l5=nn.Sequential(*(list(net.children())[:-1]))
    res_l51=nn.Sequential(*(list(net.children())[-1])[:-5])
    # pdb.set_trace()
    img1ss=torch.squeeze(img1[0,0,:,:]).detach().cpu().numpy()
    img2ss=torch.squeeze(img2[0,0,:,:]).detach().cpu().numpy()
    im1=appendimages(img1ss,img2ss)
    N,C,H,W=img1.shape[0],img1.shape[1],img1.shape[2],img1.shape[3]
    image_size=512
    
    arange6=30
    arange5=25
    arange4=20
    arange3=15
    arange2=10
    print('features extracting')
    time0=time.time()
    patch_img1s=torch.zeros((5,3,224,224)).to(device)
    patch_img2s=torch.zeros((5,3,224,224)).to(device)
    desc1s=torch.zeros((1,4096,512,512)).to(device)
    desc2s=torch.zeros((1,4096,512,512)).to(device)
    for i in range(512):
        for j in range (512):
            kp1_x1,kp1_y1=j,i
            patch_img1=F.upsample(img1[:,:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),512),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),512)],(224,224),mode='bilinear')
            patch_img1_arange3=F.upsample(img1[:,:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),512),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),512)],(224,224),mode='bilinear')
            patch_img1_arange4=F.upsample(img1[:,:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),512),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),512)],(224,224),mode='bilinear')
            patch_img1_arange5=F.upsample(img1[:,:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),512),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),512)],(224,224),mode='bilinear')
            patch_img1_arange6=F.upsample(img1[:,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),512),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),512)],(224,224),mode='bilinear')
            if patch_img1_arange3.sum()<=0:
                continue
            patch_img1s[0,:,:,:]=patch_img1
            patch_img1s[1,:,:,:]=patch_img1_arange3
            patch_img1s[2,:,:,:]=patch_img1_arange4
            patch_img1s[3,:,:,:]=patch_img1_arange5
            patch_img1s[4,:,:,:]=patch_img1_arange6
            with torch.no_grad():
                patch_img1s=(patch_img1s-torch.mean(patch_img1s,dim=(2,3),keepdim=True))/(torch.std(patch_img1s,dim=(2,3),keepdim=True)+1e-10)
                res_T=res_l51(res_l5(patch_img1s).view(-1,25088)).reshape(5,4096,1)
                res_T_2=res_l51(res_l5(transforms.functional.rotate(patch_img1s,90)).view(-1,25088)).reshape(5,4096,1)
                res_T_3=res_l51(res_l5(transforms.functional.rotate(patch_img1s,180)).view(-1,25088)).reshape(5,4096,1)
                res_T_4=res_l51(res_l5(transforms.functional.rotate(patch_img1s,270)).view(-1,25088)).reshape(5,4096,1)
                res_T_5=res_l51(res_l5(transforms.functional.rotate(patch_img1s,45)).view(-1,25088)).reshape(5,4096,1)
                res_T_6=res_l51(res_l5(transforms.functional.rotate(patch_img1s,135)).view(-1,25088)).reshape(5,4096,1)
                res_T_7=res_l51(res_l5(transforms.functional.rotate(patch_img1s,225)).view(-1,25088)).reshape(5,4096,1)
                res_T_8=res_l51(res_l5(transforms.functional.rotate(patch_img1s,315)).view(-1,25088)).reshape(5,4096,1)
                # pdb.set_trace()
                desc1s[0,:,i,j]=torch.sum((res_T+res_T_2+res_T_3+res_T_4+res_T_5+res_T_6+res_T_7+res_T_8)/8,dim=(0,2))
    for i in range(512):
        for j in range (512):
            kp2_x2,kp2_y2=j,i
            patch_img2=F.upsample(img2[:,:,max(int((kp2_y2-arange2)),0):min(int((kp2_y2+arange2)),512),max(int((kp2_x2-arange2)),0):min(int((kp2_x2+arange2)),512)],(224,224),mode='bilinear')
            patch_img2_arange3=F.upsample(img2[:,:,max(int((kp2_y2-arange3)),0):min(int((kp2_y2+arange3)),512),max(int((kp2_x2-arange3)),0):min(int((kp2_x2+arange3)),512)],(224,224),mode='bilinear')
            patch_img2_arange4=F.upsample(img2[:,:,max(int((kp2_y2-arange4)),0):min(int((kp2_y2+arange4)),512),max(int((kp2_x2-arange4)),0):min(int((kp2_x2+arange4)),512)],(224,224),mode='bilinear')
            patch_img2_arange5=F.upsample(img2[:,:,max(int((kp2_y2-arange5)),0):min(int((kp2_y2+arange5)),512),max(int((kp2_x2-arange5)),0):min(int((kp2_x2+arange5)),512)],(224,224),mode='bilinear')
            patch_img2_arange6=F.upsample(img2[:,:,max(int((kp2_y2-arange6)),0):min(int((kp2_y2+arange6)),512),max(int((kp2_x2-arange6)),0):min(int((kp2_x2+arange6)),512)],(224,224),mode='bilinear')
            if patch_img2_arange3.sum()<=0:
                continue
            patch_img2s[0,:,:,:]=patch_img2
            patch_img2s[1,:,:,:]=patch_img2_arange3
            patch_img2s[2,:,:,:]=patch_img2_arange4
            patch_img2s[3,:,:,:]=patch_img2_arange5
            patch_img2s[4,:,:,:]=patch_img2_arange6
            with torch.no_grad():
                patch_img2s=(patch_img2s-torch.mean(patch_img2s,dim=(2,3),keepdim=True))/(torch.std(patch_img2s,dim=(2,3),keepdim=True)+1e-10)
                res_T=res_l51(res_l5(patch_img2s).view(-1,25088)).reshape(5,4096,1)
                res_T_2=res_l51(res_l5(transforms.functional.rotate(patch_img2s,90)).view(-1,25088)).reshape(5,4096,1)
                res_T_3=res_l51(res_l5(transforms.functional.rotate(patch_img2s,180)).view(-1,25088)).reshape(5,4096,1)
                res_T_4=res_l51(res_l5(transforms.functional.rotate(patch_img2s,270)).view(-1,25088)).reshape(5,4096,1)
                res_T_5=res_l51(res_l5(transforms.functional.rotate(patch_img2s,45)).view(-1,25088)).reshape(5,4096,1)
                res_T_6=res_l51(res_l5(transforms.functional.rotate(patch_img2s,135)).view(-1,25088)).reshape(5,4096,1)
                res_T_7=res_l51(res_l5(transforms.functional.rotate(patch_img2s,225)).view(-1,25088)).reshape(5,4096,1)
                res_T_8=res_l51(res_l5(transforms.functional.rotate(patch_img2s,315)).view(-1,25088)).reshape(5,4096,1)
                desc2s[0,:,i,j]=torch.sum((res_T+res_T_2+res_T_3+res_T_4+res_T_5+res_T_6+res_T_7+res_T_8)/8,dim=(0,2))
    
    # scio.savemat('/data/wxy/association/Maskflownet_association/dense/vgg/'+str(step)+'_1.mat',{'data':desc1s.detach().cpu().numpy()})
    # scio.savemat('/data/wxy/association/Maskflownet_association/dense/vgg/'+str(step)+'_2.mat',{'data':desc2s.detach().cpu().numpy()})
    ## PCA数据降维
    desc1s_numpy=desc1s.reshape(4096,-1).transpose(0,1).detach().cpu().numpy()
    desc2s_numpy=desc2s.reshape(4096,-1).transpose(0,1).detach().cpu().numpy()
    pca.fit(desc1s_numpy)
    desc1s_pca = pca.transform(desc1s_numpy)# 降维后的数据
    desc1s_pca=desc1s_pca.transpose(0,1).reshape(1,128,512,512)
    pca.fit(desc2s_numpy)
    desc2s_pca = pca.transform(desc2s_numpy)# 降维后的数据
    desc2s_pca=desc2s_pca.transpose(0,1).reshape(1,128,512,512)
    scio.savemat('/data/wxy/association/Maskflownet_association/dense/vgg_pca128/'+str(step)+'_1.mat',{'data':desc1s_pca})
    scio.savemat('/data/wxy/association/Maskflownet_association/dense/vgg_pca128/'+str(step)+'_2.mat',{'data':desc2s_pca})
    return 0,0,0,0

def index_generator(n):
    indices = np.arange(0, n, dtype=np.int)
    while True:
        # np.random.shuffle(indices)
        yield from indices

from threading import Thread
from queue import Queue

def iterate_data(iq, gen,dataset):
    while True:
        i = next(gen)
        need_to_put = []
        for fid in train_pairs[i]:
            need_to_put.append(dataset[fid])
        iq.put(need_to_put)

def batch_samples(iq, oq, batch_size):
    while True:
        data_batch = []
        for i in range(batch_size):
            data_batch.append(iq.get())
        oq.put([np.stack(x, axis=0) for x in zip(*data_batch)])

def start_daemon(thread):
    thread.daemon = True
    thread.start()

def remove_file(iq):
    while True:
        f = iq.get()
        try:
            os.remove(f)
        except OSError as e:
            log.log('Remove failed' + e)


if __name__ == '__main__':
    cudnn.benchmark = True 
    dataset, groups, groups_train, groups_val = LoadANHIR()
