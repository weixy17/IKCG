import os
import sys

import argparse
import pdb
from timeit import default_timer
import yaml
import hashlib
import socket
import random
import scipy.io as scio
from skimage import io
import csv
# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = r'.'
# to CUDA\vX.Y\bin
# os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
# Flying Chairs Dataset
# chairs_path = r'F:\linge\data2\FlyingChairs\FlyingChairs_release\data'
# chairs_split_file = r'F:\linge\data2\FlyingChairs\FlyingChairs_release\FlyingChairs_train_val.txt'

import numpy as np
import mxnet as mx

# data readers
from reader.chairs import binary_reader, trainval, ppm, flo
from reader import sintel, kitti, hd1k, things3d
from reader.ANHIR import *
import cv2
import matplotlib.pyplot as plt
model_parser = argparse.ArgumentParser(add_help=False)
training_parser = argparse.ArgumentParser(add_help=False)
training_parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')

parser = argparse.ArgumentParser(parents=[model_parser, training_parser])

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('--relative', type=str, default="")
# proportion of data to be loaded
# for example, if shard = 4, then one fourth of data is loaded
# ONLY for things3d dataset (as it is very large)
parser.add_argument('-s', '--shard', type=int, default=1, help='')
parser.add_argument('-w', '--weight', type=int, default="", help='')

parser.add_argument('-g', '--gpu_device', type=str, default='0', help='Specify gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None, 
    help='model checkpoint to load; by default, the latest one.'
    'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('--clear_steps', action='store_true')
# the choice of network
parser.add_argument('-n', '--network', type=str, default='MaskFlownet') # gl, here only MastFlownet can be the input, contains Flownset_S and others.
# three modes
parser.add_argument('--debug', action='store_true', help='Do debug')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('--predict', action='store_true', help='Do prediction')
# inference resize for validation and prediction
parser.add_argument('--resize', type=str, default='')
parser.add_argument('--prep', type=str, default=None)
parser.add_argument('--validate_num', type=str, default=None)

args = parser.parse_args()
ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]
infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None

import network.config
# load network configuration
with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
    config =  network.config.Reader(yaml.safe_load(f))
# load training configuration
with open(os.path.join(repoRoot, 'network', 'config', args.dataset_cfg)) as f:
    dataset_cfg = network.config.Reader(yaml.safe_load(f))
validation_steps = dataset_cfg.validation_steps.value
checkpoint_steps = dataset_cfg.checkpoint_steps.value

# create directories
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
mkdir('logs')
mkdir(os.path.join('logs', 'val'))
mkdir(os.path.join('logs', 'debug'))
mkdir('weights')
mkdir('flows')

# find checkpoint
import path
import logger
steps = 0
if args.checkpoint is not None:
    if ':' in args.checkpoint:
        prefix, steps = args.checkpoint.split(':')
    else:
        prefix = args.checkpoint
        steps = None
    log_file, run_id = path.find_log(prefix)    
    if steps is None:
        checkpoint, steps = path.find_checkpoints(run_id)[-1]
    else:
        checkpoints = path.find_checkpoints(run_id)
        try:
            checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
        except StopIteration:
            print('The steps not found in checkpoints', steps, checkpoints)
            sys.stdout.flush()
            raise StopIteration
    steps = int(steps)
    if args.clear_steps:
        steps = 0
    else:
        _, exp_info = path.read_log(log_file)
        exp_info = exp_info[-1]
        for k in args.__dict__:
            if k in exp_info and k in ('tag',):
                setattr(args, k, eval(exp_info[k]))
                print('{}={}, '.format(k, exp_info[k]), end='')
        print()
    sys.stdout.flush()
# generate id
if args.checkpoint is None or args.clear_steps:
    uid = (socket.gethostname() + logger.FileLog._localtime().strftime('%b%d-%H%M') + args.gpu_device)
    tag = hashlib.sha224(uid.encode()).hexdigest()[:3] 
    run_id = tag + logger.FileLog._localtime().strftime('%b%d-%H%M')

# initiate
from network import get_pipeline
pipe = get_pipeline(args.network, ctx=ctx, config=config)
lr_schedule = dataset_cfg.optimizer.learning_rate.get(None)
if lr_schedule is not None:
    pipe.lr_schedule = lr_schedule

# load parameters from given checkpoint
if args.checkpoint is not None:
    print('Load Checkpoint {}'.format(checkpoint))
    sys.stdout.flush()
    network_class = getattr(config.network, 'class').get()
    # if train the head-stack network for the first time
    if network_class == 'MaskFlownet_S' and args.clear_steps and dataset_cfg.dataset.value == 'chairs':
        print('load the weight for the head network only')
        pipe.load_head(checkpoint)
    else:
        print('load the weight for the network')
        # pipe.load_from_MaskFlownet(checkpoint)
        pipe.load(checkpoint)
        # pipe.load_head(checkpoint)
    if network_class == 'MaskFlownet':
        print('fix the weight for the head network')
        pipe.fix_head()
    sys.stdout.flush()
    if not args.valid and not args.predict and not args.clear_steps:
        pipe.trainer.step(100, ignore_stale_grad=True)
        try:
            pipe.trainer.load_states(checkpoint.replace('params', 'states'))
        except:
            pass


# ======== If to do prediction ========

if args.predict:
    import predict
    checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
    predict.predict(pipe, os.path.join(repoRoot, 'flows', checkpoint_name), batch_size=args.batch, resize = infer_resize)
    sys.exit(0)
# if args.predict:
#   predict()


import pandas as pd


# ======== If to do validation ========
def validate():
    nums=[]
    if dataset_cfg.dataset.value == "ANHIR":
        size = len(train_data)
        mean_rtres=[]
        median_rtres=[]
        max_rtres=[]
        for i in range(0,size):
            
            batch_data = train_data[i]
            img1s = np.expand_dims(batch_data[0],axis=0)
            img2s = np.expand_dims(batch_data[1],axis=0)
            lmk1s = np.expand_dims(batch_data[26],axis=0)
            lmk2s = np.expand_dims(batch_data[27],axis=0)
            # pdb.set_trace()
            num=int(train_pairs[i][0].split('_')[0])
            
            mean_rtre,median_rtre,max_rtre= pipe.validate_for_ANHIR_train_evaluation(args.weight,img1s, img2s, lmk1s,lmk2s,num,checkpoint.split('/')[-1].split('.')[0])
            # pdb.set_trace()
            mean_rtres.append(mean_rtre[0][0])
            median_rtres.append(median_rtre[0][0])
            max_rtres.append(max_rtre[0][0])
            # pdb.set_trace()
            # if max_rtre[0][0]>5:
                # nums.append(num)
    # pdb.set_trace()
    # print(nums)
    return np.mean(mean_rtres),np.std(mean_rtres),np.median(mean_rtres),np.mean(median_rtres),np.std(median_rtres),np.median(median_rtres),np.mean(max_rtres),np.std(max_rtres),np.median(max_rtres)

def validate_6_evaluation(validate_num):
    nums=[]
    if dataset_cfg.dataset.value == "ANHIR":
        size = len(eval_data)
        mean_rtres=[]
        median_rtres=[]
        max_rtres=[]
        for i in [validate_num]:#range(0, size):
            batch_data = eval_data[i]
            img1s = np.expand_dims(batch_data[0],axis=0)
            img2s = np.expand_dims(batch_data[1],axis=0)
            lmk1s = np.expand_dims(batch_data[2],axis=0)
            lmk2s = np.expand_dims(batch_data[3],axis=0)
            num=int(eval_pairs[i][0].split('_')[0])
            nums.append(num)
            mean_rtre,median_rtre,max_rtre= pipe.validate_for_ANHIR_train_evaluation(args.weight,img1s, img2s, lmk1s,lmk2s,num,checkpoint.split('/')[-1].split('.')[0])
            mean_rtres.append(mean_rtre[0][0])
            median_rtres.append(median_rtre[0][0])
            max_rtres.append(max_rtre[0][0])
    return np.mean(mean_rtres),np.std(mean_rtres),np.median(mean_rtres),np.mean(median_rtres),np.std(median_rtres),np.median(median_rtres),np.mean(max_rtres),np.std(max_rtres),np.median(max_rtres),#aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax

# load training/validation datasets
validation_datasets = {}
samples = 32 if args.debug else -1

if dataset_cfg.dataset.value == 'ANHIR':
    print('loading ANHIR dataset...')
    t0 = default_timer()
    subset = dataset_cfg.subset.value
    print(subset)
    sys.stdout.flush()
    # # # ####produce ite1 kps
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_1iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_2iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_kps_pairing_3iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    # ####ite1 training
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_1iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7], f1[8], f2[8], f1[9], f2[9]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_2iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7], f1[8], f2[8], f1[9], f2[9], f1[10], f2[10]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7], f1[8], f2[8], f1[9], f2[9], f1[10], f2[10], f1[11], f2[11], f1[12], f2[12]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration_refine_for_evaluation(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7], f1[8], f2[8], f1[9], f2[9], f1[10], f2[10], f1[11], f2[11], f1[12], f2[12], f1[13], f2[13]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    train_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1],group[0][2],group[1][2],group[0][3], group[1][3],group[0][4],group[1][4],group[0][5],group[1][5],group[0][6], group[1][6],group[0][7],group[1][7],group[0][8],group[1][8],group[0][9], group[1][9],group[0][10],group[1][10],group[0][11],group[1][11],group[0][12],group[1][12],group[0][13],group[1][13]) for group in groups_train.values()]
    
    
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration_concatenate(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    
    
    eval_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1]) for group in groups_val.values()]
    eval_data = [[dataset[fid] for fid in record] for record in eval_pairs]
    train_data = [[dataset[fid] for fid in record] for record in train_pairs]
    # train_data=np.array(train_data)
    trainSize = len(train_pairs)
    validationSize = len(eval_data)
    # pdb.set_trace()
else:
    raise NotImplementedError

print('Using {}s'.format(default_timer() - t0))
sys.stdout.flush()



print('data read, train {} val {}'.format(trainSize, validationSize))
sys.stdout.flush()

#
batch_size=args.batch
assert batch_size % len(ctx) == 0
batch_size_card = batch_size // len(ctx)


if dataset_cfg.dataset.value == "ANHIR":
    raw_shape = [int(args.prep),int(args.prep)]
else:
    pass

orig_shape = dataset_cfg.orig_shape.get(raw_shape)

target_shape = dataset_cfg.target_shape.get(None)
if target_shape is None:
    target_shape = [shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape]

print(raw_shape, orig_shape, target_shape)
sys.stdout.flush()



# string='raw_1,reg_1,lmk_40,lmk_old2_40,lmk_512_6,lmk_512_2_6,lmk_LFS1_6,lmk_LFS2_6,lmk_large_16,lmk_large_1024_16,lmk_ite1_40,lmk_ite2_40,lmk_ite3_40,lmk_ite4_40'
string='raw_1,reg_0.05,lmk_20,lmk_old2_20,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_20,lmk_ite2_20,lmk_ite3_20,lmk_ite4_20'
# string='raw_1,reg_1,lmk_0,lmk_old2_0,lmk_512_0,lmk_512_2_0,lmk_LFS1_0,lmk_LFS2_0,lmk_large_0,lmk_large_1024_0,lmk_ite1_0,lmk_ite2_0,lmk_ite3_0,lmk_ite4_0'
# string='raw_1,reg_1,lmk1024_100,lmk512_100,lmk_hand_100,without_large'
# string='raw_1,reg_0.05,lmk1024_150,lmk512_0,lmk_hand_150'
# create log file
log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
# log.log('start={}, train={}, val={}, host={}, batch={},recursive,raw_multisift_1,reg_0.05,maskflownet_S-more_than_vgg_dist_1,maskflownet_S_more_than_vgg_dist_1,vgg_dist_20,vgg_large_dist_20,recursive_20,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
# log.log('start={}, train={}, val={}, host={}, batch={},recursive_kp_pairing,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
log.log(information)
# print('raw_1,reg_1/20,lmk_10,lmk_old2_4,lmk_old5_4,lmk_large_40,lmk_LFS1_6,lmk_LFS2_6,lmk_ite1_6+6,lmk_ite2_6')
# print('raw_1,reg_1/20,lmk_20,lmk_old2_20,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_20')
# print('raw_1,reg_1/20,lmk_15,lmk_old2_15,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_15')
# print('raw_1,reg_1/20,lmk_15,lmk_old2_15,lmk_512_2,lmk_512_2_2,lmk_LFS1_2,lmk_LFS2_2,lmk_large_8,lmk_large_1024_8,lmk_ite1_15,lr=5e-5')
# print('raw_1,reg_1/20,lmk_25,lmk_old2_25,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_25,lmk_ite2_25')
print(string)
# print('raw_1,reg_1/20,lmk_8,lmk_old2_4,lmk_old5_4,lmk_large_60,lmk_LFS1_4,lmk_LFS2_4,lmk_ite1_4+4,lmk_ite2_4,lmk_single_sparse_multi-dense_8')
# print('raw_1,reg_1,lmk_20,lmk_old2_5,lmk_old5_5,lmk_large_1,lmk_LFS1_5,lmk_LFS2_5,lmk_ite1_8+8')
# print('raw_1,reg_1,lmk_all_20')
# print('raw_1,reg_1/20,lmk_20,lmk_old2_5,lmk_old5_5,lmk_large_60,lmk_LFS1_5,lmk_LFS2_5,lmk_gt_10')
# print('recursive_kp_pairing')
# implement data augmentation
import augmentation
## start: gl, ANHIR program aug
aug_func = augmentation.Augmentation
if args.relative == "":
    aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card
                                            )
elif args.relative == "N":
    aug = aug_func(angle_range=(0, 0), zoom_range=(1, 1), translation_range=0, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card
                                            )
elif args.relative == "L":
    aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.9, 1 / 0.9)
                                            )
elif args.relative == "M":
    aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.25, relative_scale=(0.96, 1 / 0.96)
                                            )
elif args.relative == "S":
    aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card, relative_angle=0.16, relative_scale=(0.98, 1 / 0.98)
                                            )
elif args.relative == "U":
    aug = aug_func(angle_range=(-180, 180), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card
                                            )
elif args.relative == "UM":
    aug = aug_func(angle_range=(-180, 180), zoom_range=(0.5, 1.11), translation_range=0.1, target_shape=target_shape,
                                            orig_shape=orig_shape, batch_size=batch_size_card,
                                            relative_angle=0.1, relative_scale=(0.9, 1 / 0.9), relative_translation=0.05
                                            )
aug.hybridize()
## end: gl, ANHIR program aug
# chromatic augmentation
aug_func = augmentation.ColorAugmentation
if dataset_cfg.dataset.value == 'sintel':
    color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card, 
        shape=target_shape, noise_range=(0, 0), saturation=0.5, hue=0.5, eigen_aug = False)
elif dataset_cfg.dataset.value == 'kitti':
    color_aug = aug_func(contrast_range=(-0.2, 0.4), brightness_sigma=0.05, channel_range=(0.9, 1.2), batch_size=batch_size_card, 
        shape=target_shape, noise_range=(0, 0.02), saturation=0.25, hue=0.1, gamma_range=(-0.5, 0.5), eigen_aug = False)
else:
    color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card, 
        shape=target_shape, noise_range=(0, 0.04), saturation=0.5, hue=0.5, eigen_aug = False)
color_aug.hybridize()

def index_generator(n):
    indices = np.arange(0, n, dtype=np.int)
    while True:
        np.random.shuffle(indices)
        yield from indices
train_gen = index_generator(trainSize)
class MovingAverage:
    def __init__(self, ratio=0.95):
        self.sum = 0
        self.weight = 1e-8
        self.ratio = ratio

    def update(self, v):
        self.sum = self.sum * self.ratio + v
        self.weight = self.weight * self.ratio + 1

    @property
    def average(self):
        return self.sum / self.weight

class DictMovingAverage:
    def __init__(self, ratio=0.95):
        self.sum = {}
        self.weight = {}
        self.ratio = ratio

    def update(self, v):
        for key in v:
            if key not in self.sum:
                self.sum[key] = 0
                self.weight[key] = 1e-8
            self.sum[key] = self.sum[key] * self.ratio + v[key]
            self.weight[key] = self.weight[key] * self.ratio + 1

    @property
    def average(self):
        return dict([(key, self.sum[key] / self.weight[key]) for key in self.sum])
    
loading_time = MovingAverage()
total_time = MovingAverage()
train_avg = DictMovingAverage()

from threading import Thread
from queue import Queue

def iterate_data(iq,batch_num_queue, gen):
    while True:
        i = next(gen)
        if dataset_cfg.dataset.value == "ANHIR":
            batch_num_queue.put(i)
            need_to_put = []
            for count,fid in enumerate(train_pairs[i]):
                if count<26:
                    need_to_put.append(dataset[fid])
            iq.put(need_to_put)

def batch_samples(iq, oq, batch_size):
    while True:
        data_batch = []
        for i in range(batch_size):
            data_batch.append(iq.get())
        oq.put([np.stack(x, axis=0) for x in zip(*data_batch)])

def remove_file(iq):
    while True:
        f = iq.get()
        try:
            os.remove(f)
        except OSError as e:
            log.log('Remove failed' + e)

data_queue = Queue(maxsize=100)
batch_queue = Queue(maxsize=100)
batch_num_queue = Queue(maxsize=100)
remove_queue = Queue(maxsize=100)

def start_daemon(thread):
    thread.daemon = True
    thread.start()

start_daemon(Thread(target=iterate_data, args=(data_queue,batch_num_queue, train_gen)))
start_daemon(Thread(target=remove_file, args=(remove_queue,)))
for i in range(2):
    start_daemon(Thread(target=batch_samples, args=(data_queue, batch_queue, batch_size)))




all_data=np.zeros([len(os.listdir('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/evaluation_mean_rTRE/')),251])
for count,filename in enumerate(os.listdir('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/evaluation_mean_rTRE/')):
    data=scio.loadmat('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/evaluation_mean_rTRE/'+filename)
    data=data['rtre']
    all_data[count,:]=data
all_data_least=np.amin(all_data, axis=0)
print(all_data_least)
classes=['COAD_01','COAD_02','COAD_03','COAD_04','COAD_05','COAD_06','COAD_07','COAD_08','COAD_09','COAD_10',
              'COAD_11','COAD_12','COAD_13','COAD_14','COAD_15','COAD_16','COAD_17','COAD_18','COAD_19','COAD_20',
              'breast_1','breast_2','breast_3','breast_4','breast_5','gastric_1','gastric_2','gastric_3','gastric_4',
              'gastric_5','gastric_6','gastric_7','gastric_8','gastric_9','kidney_1','kidney_2','kidney_3','kidney_4'
              ,'kidney_5','mice-kidney_1']
nums=[]
nums_index=[]
with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'evaluation':
            nums.append(num)
            if (row[1].split('_')[0]+'_'+row[1].split('_')[1])==classes[0]:
                nums_index.append(len(nums)-1)
# print(len(nums))
t1 = None
checkpoints = []
maxkpval = float('inf')
minam=float('inf')
minmm=float('inf')
minaa=float('inf')
minma=float('inf')
count=0

import time
import copy
siftdir="/data/wxy/association/Maskflownet_association_1024/dataset/multi_siftflowmap_1024new/"
print(trainSize)
losses=[]
raw_losses=[]
reg_losses=[]
kp_losses=[]



validate_num=int(args.validate_num)
print(validate_num)
if args.valid:
    checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
    aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate()
    log.log('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
    aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate_6_evaluation()
    log.log('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
    
    sys.exit(0)
else:
    # minaa=all_data_least[nums_index].mean()
    minaa=all_data_least[validate_num]
    print(minaa)
    for steps in range(400):
        batch = []
        t0 = default_timer()
        if t1:
            total_time.update(t0 - t1)
        t1 = t0
        batch = batch_queue.get()
        name_nums=[]
        for i in range (batch_size):
            name_nums.append(batch_num_queue.get())
        img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4 = [batch[i] for i in range(26)]
        sift1 = np.zeros((batch_size, 384, int(args.prep), int(args.prep)))
        sift2 = np.zeros((batch_size, 384, int(args.prep), int(args.prep)))
        for i in range(batch_size):
            img1name=train_pairs[name_nums[i]][0]
            img2name=train_pairs[name_nums[i]][1]
            num=img1name.split('_')[0]
            motion_state1=img1name.split('_')[1].split('.')[0]
            motion_state2=img2name.split('_')[1].split('.')[0]
            sift1path=os.path.join(siftdir,num+'sift'+motion_state1+'.mat')
            sift2path=os.path.join(siftdir,num+'sift'+motion_state2+'.mat')
            siftdata11 = scio.loadmat(sift1path)
            siftdata1 = siftdata11["sift1new"]
            siftdata22 = scio.loadmat(sift2path)
            siftdata2 = siftdata22["sift2new"]
            siftdata1 = siftdata1.astype(float)
            siftdata2 = siftdata2.astype(float)
            siftdata13 = siftdata1####(1,128,512,512)
            siftdata23 = siftdata2
            sift1[i, :, :, :] = siftdata13
            sift2[i, :, :, :] = siftdata23
        train_log = pipe.train_batch_recursive_DLFSFG_multiscale_ite3(args.weight,img1, img2,sift1,sift2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4)
        train_avg.update(train_log)
        log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))
        # aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate_6_evaluation(nums_index)
        aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate_6_evaluation(validate_num)
        log.log('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        if aa<minaa:
            minaa=aa
            prefix = os.path.join(repoRoot, 'weights',str(validate_num), '{}_{}'.format(run_id, steps))
            if not os.path.exists(os.path.join(repoRoot, 'weights',str(validate_num))):
                os.mkdir(os.path.join(repoRoot, 'weights',str(validate_num)))
            pipe.save(prefix)
            print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
            checkpoints.append(prefix)
            while len(checkpoints) > 1:
                prefix = checkpoints[0]
                checkpoints = checkpoints[1:]
                remove_queue.put(prefix + '.params')
                remove_queue.put(prefix + '.states')
sys.exit(0)

