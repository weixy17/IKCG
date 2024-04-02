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
    # if network_class == 'MaskFlownet':
        # print('fix the weight for the head network')
        # pipe.fix_head()
    sys.stdout.flush()
    if not args.valid and not args.predict:
        pipe.trainer.step(100, ignore_stale_grad=True)
        pipe.trainer.load_states(checkpoint.replace('params', 'states'))


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
            # pdb.set_trace()
            img1s = np.expand_dims(batch_data[0],axis=0)
            img2s = np.expand_dims(batch_data[1],axis=0)
            lmk1s = np.expand_dims(batch_data[26],axis=0)
            lmk2s = np.expand_dims(batch_data[27],axis=0)
            # print(lmk1s.shape)
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
    return np.mean(mean_rtres),np.std(mean_rtres),np.median(mean_rtres),np.mean(median_rtres),np.std(median_rtres),np.median(median_rtres),np.mean(max_rtres),np.std(max_rtres),np.median(max_rtres),#aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax
def validate_6_evaluation():
    nums=[]
    if dataset_cfg.dataset.value == "ANHIR":
        size = len(eval_data)
        mean_rtres=[]
        median_rtres=[]
        max_rtres=[]
        for i in range(0, size):
            batch_data = eval_data[i]
            img1s = np.expand_dims(batch_data[0],axis=0)
            img2s = np.expand_dims(batch_data[1],axis=0)
            lmk1s = np.expand_dims(batch_data[2],axis=0)
            lmk2s = np.expand_dims(batch_data[3],axis=0)
            num=int(eval_pairs[i][0].split('_')[0])
            # if num not in [57,59,64,65,118,119,127,130,132]:
                # continue
            nums.append(num)
            mean_rtre,median_rtre,max_rtre= pipe.validate_for_ANHIR_train_evaluation(args.weight,img1s, img2s, lmk1s,lmk2s,num,checkpoint.split('/')[-1].split('.')[0])
            mean_rtres.append(mean_rtre)
            median_rtres.append(median_rtre)
            max_rtres.append(max_rtre)
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
    
    dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3], f1[4], f2[4], f1[5], f2[5], f1[6], f2[6], f1[7], f2[7], f1[8], f2[8], f1[9], f2[9], f1[10], f2[10], f1[11], f2[11], f1[12], f2[12], f1[13], f2[13]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    train_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1],group[0][2],group[1][2],group[0][3], group[1][3],group[0][4],group[1][4],group[0][5],group[1][5],group[0][6], group[1][6],group[0][7],group[1][7],group[0][8],group[1][8],group[0][9], group[1][9],group[0][10],group[1][10],group[0][11],group[1][11],group[0][12],group[1][12],group[0][13],group[1][13]) for group in groups_train.values()]
    
    # dataset, groups, groups_train, groups_val = LoadANHIR_recursive_DLFSFG_multiscale_training_3iteration_concatenate(args.prep, subset)
    # # train_pairs = [(f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    # train_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1],group[0][2], group[1][2],group[0][3],group[1][3]) for group in groups_train.values()]
    
    eval_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1]) for group in groups_val.values()]
    eval_data = [[dataset[fid] for fid in record] for record in eval_pairs]
    train_data = [[dataset[fid] for fid in record] for record in train_pairs]
    # train_data=np.array(train_data)
    trainSize = len(train_data)
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



# string='raw_1,reg_1,lmk1024_same_gt_200,lmk512_0,lmk_hand_0'#,without_large'
# string='raw_1,reg_0.05,lmk_20,lmk_old2_20,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_20,lmk_ite2_20,lmk_ite3_20,lmk_ite4_20'
string='raw_1,reg_0.05,lmk_20,lmk_old2_20,lmk_512_3,lmk_512_2_3,lmk_LFS1_3,lmk_LFS2_3,lmk_large_8,lmk_large_1024_8,lmk_ite1_20,lmk_ite2_20,lmk_ite3_20,lmk_ite4_20,lmk_gt_50'
# string='raw_1,reg_0.05,lmk_gt_40'
# string='raw_1,reg_1,lmk1024_100,lmk512_50,lmk_large_100'#,without_large'
# create log file
log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
log.log(information)
print(string)
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
                if count<28:
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
while True:
    if args.valid:
        checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
        aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate()
        # log.log('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate_6_evaluation()
        # log.log('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        
        sys.exit(0)
    ##########produce kps
    if (steps+1)%100000000==0:#(steps+1)%5000==0:
        count=count+1
        for i in range (200,400,2):
        # for i in range (0,len(train_pairs),2):
            data_batch = []
            for fid in train_pairs[i]:
                data_batch.append(dataset[fid])
            numname=int(fid.split('_')[0])
            # if numname!=41 and numname!=71:
                # continue
            print(i)
            batch=[np.stack(x, axis=0) for x in zip(*[data_batch])]
            img1, img2,img1_256, img2_256,img1_1024, img2_1024,kp1,kp2,kp1_old2,kp2_old2,kp1_ite1,kp2_ite1,orb1,orb2 = [batch[i] for i in range(14)]
            train_log = pipe.kp_pairs_multiscale(args.weight,img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1,orb2,i,numname)
        sys.exit(0)
    #########training
    else:
        batch = []
        t0 = default_timer()
        if t1:
            total_time.update(t0 - t1)
        t1 = t0
        batch = batch_queue.get()
        name_nums=[]
        for i in range (batch_size):
            name_nums.append(batch_num_queue.get())
        # img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1 = [batch[i] for i in range(20)]
        # img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2 = [batch[i] for i in range(22)]
        # img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4 = [batch[i] for i in range(26)]
        img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4,kp1_gt,kp2_gt = [batch[i] for i in range(28)]
        # img1, img2,kp1,kp2,kp1_512,kp2_512,kp1_hand,kp2_hand = [batch[i] for i in range(8)]
        # img1, img2,kp1,kp2,kp1_2,kp2_2= [batch[i] for i in range(6)]
        
        
        # train_log = pipe.train_batch_sparse_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(args.weight,img1, img2,kp1,kp2,kp1_512,kp2_512,kp1_hand,kp2_hand)
        # train_log = pipe.train_batch_recursive_DLFS_SFG_multiscale_ite3(args.weight,img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4)
        train_log = pipe.train_batch_recursive_DLFS_SFG_multiscale_ite3_with_gt(args.weight,img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_512,kp2_512,kp1_512_2,kp2_512_2,kp1_LFS1,kp2_LFS1,kp1_LFS2,kp2_LFS2,kp1_large,kp2_large,kp1_large_1024,kp2_large_1024,kp1_ite1,kp2_ite1,kp1_ite2,kp2_ite2,kp1_ite3,kp2_ite3,kp1_ite4,kp2_ite4,kp1_gt,kp2_gt )
        if  steps<100 or steps % 10 == 0:
            train_avg.update(train_log)
            log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))
        if steps%10==0: #or steps <= 20:
            if validationSize > 0:
                train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax = validate()
                log.log('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                # aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax = validate_6_evaluation()
                # log.log('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
        
                # save parameters
                if 1:#steps % checkpoint_steps == 0 or steps <= 20:
                    # if aa<2.805:
                        # prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
                        # pipe.save(prefix)
                        # print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
                        # print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                
                    if train_aa < minaa: # gl, keep the best model for test dataset.
                        minaa = train_aa
                        if train_ma<minma:
                            minma = train_ma
                        if train_am<minam:
                            minam = train_am
                        if train_mm<minmm:
                            minmm = train_mm
                        prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
                        pipe.save(prefix)
                        # print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
                        print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                        if not(train_aa<0.39):
                            checkpoints.append(prefix)
                            #########remove the older checkpoints
                            while len(checkpoints) > 10:
                                prefix = checkpoints[0]
                                checkpoints = checkpoints[1:]
                                remove_queue.put(prefix + '.params')
                                remove_queue.put(prefix + '.states')
                    elif train_ma<minma:
                        minma = train_ma
                        if train_am<minam:
                            minam = train_am
                        if train_mm<minmm:
                            minmm = train_mm
                        prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
                        pipe.save(prefix)
                        # print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
                        print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                        if not(train_aa<0.39):
                            checkpoints.append(prefix)
                            #########remove the older checkpoints
                            while len(checkpoints) > 10:
                                prefix = checkpoints[0]
                                checkpoints = checkpoints[1:]
                                remove_queue.put(prefix + '.params')
                                remove_queue.put(prefix + '.states')
                    elif train_am<minam:
                        minam = train_am
                        if train_mm<minmm:
                            minmm = train_mm
                        prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
                        pipe.save(prefix)
                        # print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
                        print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                        if not(train_aa<0.39):
                            checkpoints.append(prefix)
                            #########remove the older checkpoints
                            while len(checkpoints) > 10:
                                prefix = checkpoints[0]
                                checkpoints = checkpoints[1:]
                                remove_queue.put(prefix + '.params')
                                remove_queue.put(prefix + '.states')
                    elif train_mm<minmm:
                        minmm = train_mm
                        prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
                        pipe.save(prefix)
                        # print('steps= {} 6e_aa={},6e_aa_std={},6e_ma={},6e_am={},6e_am_std={},6e_mm={},6e_amax={},6e_amax_std={},6e_mmax={}'.format(steps,aa,aa_std,ma,am,am_std,mm,amax,amax_std,mmax))
                        print('steps= {} train_aa={},train_aa_std={},train_ma={},train_am={},train_am_std={},train_mm={},train_amax={},train_amax_std={},train_mmax={}'.format(steps,train_aa,train_aa_std,train_ma,train_am,train_am_std,train_mm,train_amax,train_amax_std,train_mmax))
                        if not(train_aa<0.39):
                            checkpoints.append(prefix)
                            #########remove the older checkpoints
                            while len(checkpoints) > 10:
                                prefix = checkpoints[0]
                                checkpoints = checkpoints[1:]
                                remove_queue.put(prefix + '.params')
                                remove_queue.put(prefix + '.states')
    steps=steps+1