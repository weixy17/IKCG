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
import copy
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
from reader.TMI_rebuttle_ACROBAT import *
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
parser.add_argument('--gkps', action='store_true', help='Do gkps')
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
        run_id=prefix
    else:
        prefix = args.checkpoint
        steps = None
      
    if steps is None:
        log_file, run_id = path.find_log(prefix)  
        checkpoint, steps = path.find_checkpoints(run_id)[-1]
    else:
        # checkpoints = path.find_checkpoints(run_id)
        # checkpoint = "/ssd1/wxy/association/Maskflownet_association/weights/"+run_id+'_'+steps+'.pth.tar'#path.find_checkpoints(run_id)
        checkpoint = "/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/"+run_id+'_'+steps+'.params'#path.find_checkpoints(run_id)
        # try:
            # checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
        # except StopIteration:
            # print('The steps not found in checkpoints', steps, checkpoints)
            # sys.stdout.flush()
            # raise StopIteration
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
    print(run_id)

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
    if not args.valid and not args.predict:# and not args.clear_steps:
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

# load training/validation datasets
validation_datasets = {}
samples = 32 if args.debug else -1

if dataset_cfg.dataset.value == 'ACROBAT':
    print('loading ACROBAT dataset...')
    t0 = default_timer()
    subset = dataset_cfg.subset.value
    print(subset)
    sys.stdout.flush()
    # dataset, train_pairs, valid_pairs = LoadACROBAT_baseline512_PWC()
    # dataset, train_pairs, valid_pairs = LoadACROBAT_baseline512()
    dataset, train_pairs, valid_pairs = LoadACROBAT_DFS_SFG_multiscale512()
    # dataset, train_pairs, valid_pairs = LoadACROBAT_PCG_kps_ite1_multiscale512()
    # dataset, train_pairs, valid_pairs = LoadACROBAT_PCG_kps_ite2_multiscale512()
    train_data=[[dataset[fid] for fid in record] for record in train_pairs]
    valid_data = [[dataset[fid] for fid in record] for record in valid_pairs]
    trainSize = len(train_pairs)
    validationSize = len(valid_pairs)
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


if dataset_cfg.dataset.value == "ACROBAT":
    raw_shape = [int(args.prep),int(args.prep)]
else:
    pass

orig_shape = dataset_cfg.orig_shape.get(raw_shape)

target_shape = dataset_cfg.target_shape.get(None)
if target_shape is None:
    target_shape = [shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape]

print(raw_shape, orig_shape, target_shape)
sys.stdout.flush()

# # # # # # create log file
# # # # # log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
# # # # # # log.log('start={}, train={}, val={}, host={}, batch={},recursive,raw_multisift_1,reg_0.05,maskflownet_S-more_than_vgg_dist_1,maskflownet_S_more_than_vgg_dist_1,vgg_dist_20,vgg_large_dist_20,recursive_20,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
# # # # # log.log('start={}, train={}, val={}, host={}, batch={},raw_1,reg_1/20,lmk_10,lmk_old2_5,lmk_old5_5,lmk_large_60,lmk_LFS1_5,lmk_LFS2_5,lmk_ite1_4+4,lmk_ite2_4,lmk_ite3_4,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
# # # # # # log.log('start={}, train={}, val={}, host={}, batch={},recursive_kp_pairing,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint))
# # # # # information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
# # # # # log.log(information)
# # # # # # print('raw_1,reg_1/20,lmk_10,lmk_old2_4,lmk_old5_4,lmk_large_40,lmk_LFS1_6,lmk_LFS2_6,lmk_ite1_6+6,lmk_ite2_6')
# # # # # print('raw_1,reg_1/20,lmk_10,lmk_old2_5,lmk_old5_5,lmk_large_60,lmk_LFS1_5,lmk_LFS2_5,lmk_ite1_4+4,lmk_ite2_4,lmk_ite3_4')
# # # # # # print('raw_1,reg_1/20,lmk_8,lmk_old2_4,lmk_old5_4,lmk_large_60,lmk_LFS1_4,lmk_LFS2_4,lmk_ite1_4+4,lmk_ite2_4,lmk_single_sparse_multi-dense_8')
# # # # # # print('raw_1,reg_1,lmk_20,lmk_old2_5,lmk_old5_5,lmk_large_1,lmk_LFS1_5,lmk_LFS2_5,lmk_ite1_8+8')
# # # # # # print('raw_1,reg_1,lmk_all_20')
# # # # # # print('raw_1,reg_1/20,lmk_20,lmk_old2_5,lmk_old5_5,lmk_large_60,lmk_LFS1_5,lmk_LFS2_5,lmk_gt_10')
# # # # # # print('recursive_kp_pairing')


# create log file
if not args.valid:
    # string='start={}, train={}, val={}, host={}, batch={},raw_1,reg_1,lmk_10,lmk_old2_10,lmk_old5_10,lmk_large_20,lmk_LFS1_5,lmk_LFS2_5,lmk_ite1_4+4,lmk_ite2_4,lmk_ite3_4,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint)
    string='start={}, train={}, val={}, host={}, batch={},rawSIFT_1,reg_1,kp1_30,kp2_30,kp3_0,Load Checkpoint {}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size,checkpoint)
    log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
    log_partial = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}-partial.log'.format(run_id)))
    log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
    log_partial.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
    information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
    log.log(information)
    log_partial.log(information)
    print(string)



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
        if dataset_cfg.dataset.value == "ACROBAT":
            batch_num_queue.put(i)
            need_to_put = []
            for count,fid in enumerate(train_pairs[i]):
                if count<22:
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
minmm90th=float('inf')
minam90th=float('inf')
count=0
import time
import copy
siftdir="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/SIFT_features/train_4096_to512/"
print(trainSize)
while True:
    if args.valid:
        dist_means,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
            mMI,sMI,mm75th,mm90th = pipe.validate_ACROBAT_large(valid_data)
        print("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},am75th={:.4f}({:.4f}),m75thm75th={:.4f},\
am90th={:.4f}({:.4f}),m90thm90th={:.4f},large1={:.4f}({:.4f}),large2={:.4f}({:.4f}),large3={:.4f}({:.4f}),large4={:.4f}({:.4f}),\
large5={:.4f}({:.4f}),mMI={:.4f}({:.4f}),mm75th={:.4f},mm90th={:.4f},""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,\
                    large3,large3_std,large4,large4_std,large5,large5_std,mMI,sMI,mm75th,mm90th))
        # pdb.set_trace()
        sys.exit(0)
    if args.gkps:
        train_pairs_nums=[int(temp[0].split('_')[0]) for temp in train_pairs]
        train_pairs_nums_unique=np.unique(train_pairs_nums, return_index=False,axis=0)
        count=count+1
        for i in range (494,train_pairs_nums_unique.shape[0]):
            print(i)
            numname=train_pairs_nums_unique[i]
            idxs=np.where(train_pairs_nums==numname)[0]
            for count,idx in enumerate(idxs):
                print(train_pairs[idx])
                data_batch = []
                for fid in train_pairs[idx]:
                    data_batch.append(dataset[fid])
                batch=[np.stack(x, axis=0) for x in zip(*[data_batch])]
                img1, img2,img1_512, img2_512,img1_2048, img2_2048,orb1,orb2 = [batch[i] for i in range(8)]
                if count==0:
                    normalized_desc2s = pipe.rebuttle_kp_pairs_multiscale(args.weight,img1, img2,img1_512, img2_512,img1_2048, img2_2048,orb1,orb2,train_pairs[idx][0].split('.')[0])
                else:
                    normalized_desc2s = pipe.rebuttle_kp_pairs_multiscale_speed(args.weight,img1, img2,img1_512, img2_512,img1_2048, img2_2048,orb1,orb2,train_pairs[idx][0].split('.')[0],normalized_desc2s)
            del normalized_desc2s
        sys.exit(0)
    steps=steps+1
    batch=[]
    
    
    
    sift1 = np.zeros((batch_size, 128, 512,512))
    sift2 = np.zeros((batch_size, 128, 512,512))
    for i in range(batch_size):
        name_num = next(train_gen)
        need_to_put=[]
        for count,fid in enumerate(train_pairs[name_num]):
            need_to_put.append(dataset[fid])
        batch.append(need_to_put)
        img1name=train_pairs[name_num][0]
        img2name=train_pairs[name_num][1]
        if 0:#img1name.split('.')[0]=='57_PGR_val' or img1name.split('.')[0]=='57_HE_val' or img1name.split('.')[0]=='83_KI67_val':
            sift1[i, 0:3, :, :] = dataset[img1name]
            sift2[i, 0:3, :, :] = dataset[img2name]
        else:
            sift1path=os.path.join(siftdir,img1name.split('.')[0]+'.mat')
            sift2path=os.path.join(siftdir,img2name.split('.')[0]+'.mat')
            siftdata1 = scio.loadmat(sift1path)["sift"].astype(float)
            siftdata2 = scio.loadmat(sift2path)["sift"].astype(float)
            sift1[i, :, :, :] = np.transpose(siftdata1, (2, 0, 1))
            sift2[i, :, :, :] = np.transpose(siftdata2, (2, 0, 1))
            
            
            
            
    batch=[np.stack(x, axis=0) for x in zip(*batch)]
    img1, img2,kp1,kp2,kp1_2,kp2_2,kp1_3,kp2_3 = [batch[i] for i in range(len(train_pairs[name_num]))]
    # img1, img2 = [batch[i] for i in range(len(train_pairs[name_num]))]
    # img1, img2 = [batch[i] for i in range(2)]

    train_log = pipe.train_batch_densewithsparse_multisift_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(args.weight,img1, img2,sift1,sift2,kp1,kp2,kp1_2,kp2_2,kp1_3,kp2_3,steps)
    # train_log = pipe.train_batch_baseline(args.weight,img1, img2)
    # train_log = pipe.train_batch_vgg_dense(args.weight,img1, img2,sift1,sift2)
    # update log
    train_avg.update(train_log)
    log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))
    if 1:
        dist_means,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
        mMI,sMI,mm75th,mm90th = pipe.validate_ACROBAT_large(valid_data)
        # print(dist_means)
        string_log=("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},am75th={:.4f}({:.4f}),m75thm75th={:.4f},\
am90th={:.4f}({:.4f}),m90thm90th={:.4f},large1={:.4f}({:.4f}),large2={:.4f}({:.4f}),large3={:.4f}({:.4f}),large4={:.4f}({:.4f}),\
large5={:.4f}({:.4f}),mMI={:.4f}({:.4f}),mm75th={:.4f},mm90th={:.4f},dist_means={:.4f}""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,\
                large3,large3_std,large4,large4_std,large5,large5_std,mMI,sMI,mm75th,mm90th,dist_means[0]))
        log.log(string_log)
        # print(dist_means[80])
        # if aa<=87 and aa>=83.8926 and ma>=52.0262 and m75tha>=101.2515 and m90tha>=175.9617 and am>=59.7428 and mm>=40.5079 and am75th>=100.4014\
         # and m75thm75th>=110.2767 and am90th>=163.6223 and m90thm90th>=393.2043 and large1>=150.7580 and large2>=179.8576 and large3>=212.1112\
          # and large4>=268.8957 and large5>=567.5790 and mMI<=0.6950 and mm75th>=59.0672 and mm90th>=86.9850:
        # if aa<85.5232 and large1<151.0668 and large2<178.6926 and large3<211.1173 and large4<245.5146 and large5<463.0018 and mMI>0.6933:# and dist_means<50:
        # if aa<85.5232 and large1<151.0668 and large2<178.6926 and large3<211.1173 and large4<245.5146 and large5<431 and mMI>0.6933:# and dist_means<50:
        # # if aa<85.5232 and ma<55.4723 and m75tha<104.2676 and m90tha<174.8413 and am<63.0435 and mm<41.8051 and am75th<101.9874 and m75thm75th<119.7238\
            # # and am90th<165.7467 and m90thm90th<379.3284 and large1<151.0668 and large2<178.6926 and large3<211.1173 and large4<245.5146 and large5<463.0018\
            # # and mMI>0.6933 and mm75th<60.7892 and mm90th<88.4293:# and dist_means<50:
        # if aa <= 86: # gl, keep the best model for test dataset.
        if aa<=90 and m75tha<=111.2498 and am75th<=109.7667 and m75thm75th<=116.2204 and mMI>=0.6763:
        # if aa<=86.7508 and m75tha<=102.1889 and am75th<=102.1868 and m75thm75th<=116.1017:# and mMI>=0.6853:
            prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
            pipe.save(prefix)
            print(string_log)
            log_partial.log(string_log)

        
        
        
        # if aa < minaa: # gl, keep the best model for test dataset.
            # minaa = aa
            # if ma<minma:
                # minma = ma
            # if mm90th<minmm90th:
                # minmm90th = mm90th
            # if am90th<minam90th:
                # minam90th = am90th
            # if aa<200:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # # checkpoints.append(prefix)
            # # #########remove the older checkpoints
            # # while len(checkpoints) > 40:
                # # prefix = checkpoints[0]
                # # checkpoints = checkpoints[1:]
                # # remove_queue.put(prefix + '.params')
                # # remove_queue.put(prefix + '.states')
        # elif ma<minma:
            # minma = ma
            # if mm90th<minmm90th:
                # minmm90th = mm90th
            # if am90th<minam90th:
                # minam90th = am90th
            # if aa<200:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # # checkpoints.append(prefix)
            # # #########remove the older checkpoints
            # # while len(checkpoints) > 40:
                # # prefix = checkpoints[0]
                # # checkpoints = checkpoints[1:]
                # # remove_queue.put(prefix + '.params')
                # # remove_queue.put(prefix + '.states')
        # elif mm90th<minmm90th:
            # minmm90th = mm90th
            # if am90th<minam90th:
                # minam90th = am90th
            # if aa<200:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # # checkpoints.append(prefix)
            # # #########remove the older checkpoints
            # # while len(checkpoints) > 40:
                # # prefix = checkpoints[0]
                # # checkpoints = checkpoints[1:]
                # # remove_queue.put(prefix + '.params')
                # # remove_queue.put(prefix + '.states')
        # elif am90th<minam90th:
            # minam90th = am90th
            # if aa<200:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # # checkpoints.append(prefix)
            # # #########remove the older checkpoints
            # # while len(checkpoints) > 40:
                # # prefix = checkpoints[0]
                # # checkpoints = checkpoints[1:]
                # # remove_queue.put(prefix + '.params')
                # # remove_queue.put(prefix + '.states')
        