import os
import sys
os.system('source activate mxnet')
os.system('export MXNET_USE_FUSION=0')
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
from reader.TMI_rebuttle_ANHIR import LoadANHIR_vgg_vgglarge_multiscale_new
import cv2

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
        pipe.load(checkpoint)
    if network_class == 'MaskFlownet':
        print('fix the weight for the head network')
        pipe.fix_head()
    sys.stdout.flush()
    if not args.valid and not args.predict and not args.clear_steps:
        pipe.trainer.step(100, ignore_stale_grad=True)
        # pipe.trainer.load_states(checkpoint.replace('params', 'states'))


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
    if dataset_cfg.dataset.value == "ANHIR":
        bs = 1#len(ctx)
        size = len(eval_data)
        img1s = np.zeros((bs, 3, int(args.prep), int(args.prep)))
        img2s = np.zeros((bs, 3, int(args.prep), int(args.prep)))
        lmk1s = np.zeros((bs, 200, 2))
        lmk2s = np.zeros((bs, 200, 2))
        mean_rtres=[]
        median_rtres=[]
        median75th_rtres=[]
        jacobi_means=[]
        jacobi_stds=[]
        jacobi_foldings=[]
        lmk_dist_oris=[]
        lmk_dist_large_mean1s = []
        lmk_dist_large_mean2s = []
        lmk_dist_large_mean3s = []
        lmk_dist_large_mean4s = []
        lmk_dist_large_mean5s = []
        for i in range(0, size, bs):
            batch_data = eval_data[i: i + bs]
            for p in range(0, bs):
                img1s[p, :, :, :] = batch_data[p][0]
                img2s[p, :, :, :] = batch_data[p][1]
                if p==0:
                    lmk1s=np.expand_dims(batch_data[p][2],0)
                    lmk2s=np.expand_dims(batch_data[p][3],0)
                else:
                    lmk1s=np.concatenate((lmk1s,np.expand_dims(batch_data[p][2],0)),0)
                    lmk2s=np.concatenate((lmk2s,np.expand_dims(batch_data[p][3],0)),0)
            mean_rtre,median_rtre,median75th_rtre,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,lmk_dist_large_mean4,lmk_dist_large_mean5\
                                            ,jacobi_mean,jacobi_std,jacobi_folding,lmk_dist_ori= pipe.validate_for_4_indicators_original(args.weight,img1s, img2s, lmk1s, lmk2s,i+1)
            # mean_rtre,median_rtre= pipe.validate_for_4_indicators(args.weight,img1s, img2s, lmk1s, lmk2s,i+1)
            #io.imsave(os.path.join('/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_1024/flows/',batch_data[p][0].split('/')[-1]),np.squeeze(img1[0,0,:,:]))
            #io.imsave(os.path.join('/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_1024/flows/',batch_data[p][1].split('/')[-1]),np.squeeze(warp[0,0,:,:]))
            #io.imsave(os.path.join('/data/gl/re_do_from_ori/program/slice_s4_deformable_reg_baseline/s4_deformable_reg_baseline_1024/flows_2/',batch_data[p][1].split('/')[-1]),np.squeeze(img2s[0,0,:,:]))
            mean_rtres.append(mean_rtre)
            median_rtres.append(median_rtre)
            median75th_rtres.append(median75th_rtre)
            jacobi_means.append(jacobi_mean)
            jacobi_stds.append(jacobi_std)
            jacobi_foldings.append(jacobi_folding)
            lmk_dist_oris.append(lmk_dist_ori.squeeze().tolist())
            if lmk_dist_large_mean1>0:
                lmk_dist_large_mean1s.append(lmk_dist_large_mean1)
            if lmk_dist_large_mean2>0:
                lmk_dist_large_mean2s.append(lmk_dist_large_mean2)
            if lmk_dist_large_mean3>0:
                lmk_dist_large_mean3s.append(lmk_dist_large_mean3)
            if lmk_dist_large_mean4>0:
                lmk_dist_large_mean4s.append(lmk_dist_large_mean4)
            if lmk_dist_large_mean5>0:
                lmk_dist_large_mean5s.append(lmk_dist_large_mean5)
    # lmk_dist_oris=np.array(lmk_dist_oris).reshape(-1)
    # lmk_dist_oris=lmk_dist_oris[np.where(lmk_dist_oris)[0]]
    
    return np.mean(mean_rtres),np.std(mean_rtres),np.median(mean_rtres),np.percentile(mean_rtres,75),np.mean(median_rtres),np.std(median_rtres),np.median(median_rtres)\
    ,np.mean(median75th_rtres),np.std(median75th_rtres),np.percentile(median75th_rtres,75),np.mean(lmk_dist_large_mean1s),np.std(lmk_dist_large_mean1s),np.mean(lmk_dist_large_mean2s)\
    ,np.std(lmk_dist_large_mean2s),np.mean(lmk_dist_large_mean3s),np.std(lmk_dist_large_mean3s),np.mean(lmk_dist_large_mean4s),np.std(lmk_dist_large_mean4s)\
    ,np.mean(lmk_dist_large_mean5s),np.std(lmk_dist_large_mean5s),np.mean(jacobi_stds),np.mean(jacobi_foldings),np.std(jacobi_foldings)#aa,aa_std,ma,am,am_std,mm

# load training/validation datasets
validation_datasets = {}
samples = 32 if args.debug else -1

if dataset_cfg.dataset.value == 'ANHIR':
    print('loading ANHIR dataset...')
    t0 = default_timer()
    subset = dataset_cfg.subset.value
    print(subset)
    sys.stdout.flush()
    dataset, groups, groups_train, groups_val = LoadANHIR_vgg_vgglarge_multiscale_new(args.prep, subset)
    # train_pairs = [(f1[0], f2[0], f1[1], f2[1],f1[2], f2[2], f1[3], f2[3],f1[4], f2[4]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    train_pairs = [(f1[0], f2[0], f1[1], f2[1],f1[2], f2[2]) for group in groups_train.values() for f1 in group for f2 in group if f1 is not f2]
    eval_pairs = [(group[0][0], group[1][0],group[0][1],group[1][1]) for group in groups_val.values()]
    eval_data = [[dataset[fid] for fid in record] for record in eval_pairs]
    trainSize = len(train_pairs)
    validationSize = len(eval_data)
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

# orig_shape = dataset_cfg.orig_shape.get(orig_shape)
# target_shape = dataset_cfg.target_shape.get([shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape])
# print('original shape: ' + str(orig_shape))
# print('target shape: ' + str(target_shape))
# sys.stdout.flush()
if dataset_cfg.dataset.value == "ANHIR":
    raw_shape = [int(args.prep),int(args.prep)]#dataset[train_pairs[0][0]].shape[1: 3]
else:
    #raw_shape = trainImg1[0].shape[:2]
    pass

orig_shape = dataset_cfg.orig_shape.get(raw_shape)

target_shape = dataset_cfg.target_shape.get(None)
if target_shape is None:
    target_shape = [shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape]

print(raw_shape, orig_shape, target_shape)
sys.stdout.flush()

# create log file
string='raw_1,reg_1,kps_25,kps_large_40'
log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
# log.log('start={}, train={}, val={}, host={}, batch={},raw_1,reg_0.05,maskflownet_S&gt_20,maskflownet_S-(maskflownet_S&gt)_10,vgg_dist_20,vgg_large_dist_20,network_dist_20,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
log.log(information)
print(string)

# print('vgg_dense_raw_1,reg_1')
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
        # print('i:{}'.format(i))
        if dataset_cfg.dataset.value == "ANHIR":
            batch_num_queue.put(i)
            need_to_put = []
            for count,fid in enumerate(train_pairs[i]):
                if count<10:
                    need_to_put.append(dataset[fid])
            iq.put(need_to_put)
def batch_samples(iq, oq, batch_size):
    while True:
        data_batch = []
        for i in range(batch_size):
            # print('iq.size2:{}'.format(iq.qsize()))
            data_batch.append(iq.get())
            # print('iq.size3:{}'.format(iq.qsize()))
            # print('len(data_batch): {}'.format(len(data_batch)))
        oq.put([np.stack(x, axis=0) for x in zip(*data_batch)])
        # print('oq.size: {}'.format(oq.qsize()))

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

while True:
    if args.valid:
        checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
        aa,aa_std,ma,m75tha,am,am_std,mm,am75th,am75th_std,m75thm75th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
                            mJacstd,mJacfold,sJacfold = validate()
        print("""steps= {} aa={:.4f},aa_std={:.4f},ma={:.4f} m75tha={:.4f} am={:.4f},am_std={:.4f},mm={:.4f},am75th={:.4f},am75th_std={:.4f},m75thm75th={:.4f},large1={:.4f},\
large1_std={:.4f},large2={:.4f},large2_std={:.4f},large3={:.4f},large3_std={:.4f},large4={:.4f},large4_std={:.4f},large5={:.4f},large5_std={:.4f},\
mJacstd={:.8f},mJacfold={:.12f},sJacfold={:.12f}""".format(steps,aa,aa_std,ma,m75tha,am,am_std,mm,am75th,am75th_std,m75thm75th,large1,large1_std,large2,large2_std,\
                    large3,large3_std,large4,large4_std,large5,large5_std,mJacstd,mJacfold,sJacfold))
        sys.exit(0)
    
    steps=steps+1
    batch = []
    t0 = default_timer()
    if t1:
        total_time.update(t0 - t1)
    t1 = t0
    batch = batch_queue.get()
    name_nums=[]
    for i in range (batch_size):
        name_nums.append(batch_num_queue.get())
    # img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_old5,kp2_old5,kp1_large,kp2_large = [batch[i] for i in range(10)]
    img1, img2,kp1,kp2,kp1_large,kp2_large = [batch[i] for i in range(6)]
    # train_log = pipe.train_batch_sparse_vgg_vgglarge_multiscale(args.weight,img1, img2,kp1,kp2,kp1_old2,kp2_old2,kp1_old5,kp2_old5,kp1_large,kp2_large)
    # train_log = pipe.train_batch_S_SFG(args.weight,img1, img2,kp1,kp2)
    train_log = pipe.train_batch_supervised_LFS_SFG(args.weight,img1, img2,kp1,kp2,kp1_large,kp2_large)
    # update log
    if  1:#steps<100 or steps % 10 == 0:
        train_avg.update(train_log)
        log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))

    # do valiation
    if steps % validation_steps == 0 or steps <= 100: #or steps <= 20:
        if validationSize > 0:
            aa,aa_std,ma,m75tha,am,am_std,mm,am75th,am75th_std,m75thm75th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
                            mJacstd,mJacfold,sJacfold = validate()
            string_log=("""steps= {} aa={:.4f},aa_std={:.4f},ma={:.4f} m75tha={:.4f} am={:.4f},am_std={:.4f},mm={:.4f},am75th={:.4f},am75th_std={:.4f},m75thm75th={:.4f},large1={:.4f},\
large1_std={:.4f},large2={:.4f},large2_std={:.4f},large3={:.4f},large3_std={:.4f},large4={:.4f},large4_std={:.4f},large5={:.4f},large5_std={:.4f},\
mJacstd={:.8f},mJacfold={:.12f},sJacfold={:.12f}""".format(steps,aa,aa_std,ma,m75tha,am,am_std,mm,am75th,am75th_std,m75thm75th,large1,large1_std,large2,large2_std,\
                    large3,large3_std,large4,large4_std,large5,large5_std,mJacstd,mJacfold,sJacfold))
            log.log(string_log)
            if aa < minaa: # gl, keep the best model for test dataset.
                minaa = aa
                if ma<minma:
                    minma = ma
                if am<minam:
                    minam = am
                if mm<minmm:
                    minmm = mm
                prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                print(string_log)
                if not(aa<4.2):#(aa<3.045 and ma<2.425 and am<2.445 and mm<1.975):
                    checkpoints.append(prefix)
                    #########remove the older checkpoints
                    while len(checkpoints) > 10:
                        prefix = checkpoints[0]
                        checkpoints = checkpoints[1:]
                        remove_queue.put(prefix + '.params')
                        remove_queue.put(prefix + '.states')
            elif ma<minma and aa<4.3:
                minma = ma
                if am<minam:
                    minam = am
                if mm<minmm:
                    minmm = mm
                prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                print(string_log)
                if not(aa<4.2):#(aa<3.045 and ma<2.425 and am<2.445 and mm<1.975):
                    checkpoints.append(prefix)
                    #########remove the older checkpoints
                    while len(checkpoints) > 10:
                        prefix = checkpoints[0]
                        checkpoints = checkpoints[1:]
                        remove_queue.put(prefix + '.params')
                        remove_queue.put(prefix + '.states')
            elif am<minam and aa<4.3:
                minam = am
                if mm<minmm:
                    minmm = mm
                prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                print(string_log)
                if not(aa<4.2):#(aa<3.045 and ma<2.425 and am<2.445 and mm<1.975):
                    checkpoints.append(prefix)
                    #########remove the older checkpoints
                    while len(checkpoints) > 10:
                        prefix = checkpoints[0]
                        checkpoints = checkpoints[1:]
                        remove_queue.put(prefix + '.params')
                        remove_queue.put(prefix + '.states')
            elif mm<minmm and aa<4.3:
                minmm = mm
                prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                print(string_log)
                if not(aa<4.2):#(aa<3.045 and ma<2.425 and am<2.445 and mm<1.975):
                    checkpoints.append(prefix)
                    #########remove the older checkpoints
                    while len(checkpoints) > 10:
                        prefix = checkpoints[0]
                        checkpoints = checkpoints[1:]
                        remove_queue.put(prefix + '.params')
                        remove_queue.put(prefix + '.states')