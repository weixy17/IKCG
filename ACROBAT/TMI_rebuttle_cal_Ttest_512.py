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
import scipy.stats
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
    if not args.valid and not args.predict:
        pipe.trainer.step(100, ignore_stale_grad=True)
        if checkpoint.split('/')[-1]!='2afApr28-1544_475000.params':
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


validation_datasets = {}
samples = 32 if args.debug else -1

if dataset_cfg.dataset.value == 'ACROBAT':
    print('loading ACROBAT dataset...')
    t0 = default_timer()
    subset = dataset_cfg.subset.value
    print(subset)
    sys.stdout.flush()
    dataset, train_pairs, valid_pairs = LoadACROBAT_image_visualization512()
    # dataset, train_pairs, valid_pairs = LoadACROBAT_image_visualization1024()
    valid_data = [[dataset[fid] for fid in record] for record in valid_pairs]
    # idxs=scio.loadmat('idx.mat')['data']
    # for idx in idxs.squeeze().tolist()[-80:-20]:
        # imgidx=np.int32(np.floor(idx/200))
        # kpsidx=idx-imgidx*200
        # valid_data[imgidx][3][kpsidx,0]=0
        # valid_data[imgidx][3][kpsidx,1]=0
        # # print(valid_pairs[imgidx])
        # # print(kpsidx)
    # idxs2=scio.loadmat('idx2.mat')['data']
    # temp=idxs2.squeeze().tolist()[-600:-100]
    # # temp=[temp2 for temp2 in idxs2.squeeze().tolist()[-650:] if temp2 in idxs.squeeze().tolist()[-650:]]
    # # temp2=[temp3 for temp3 in idxs2.squeeze().tolist()[-650:] if temp3 in idxs.squeeze().tolist()[-1400:-750]]
    # # temp.extend(temp2)
    # # pdb.set_trace()
    # # pdb.set_trace()
    # for idx in temp:
        # imgidx=np.int32(np.floor(idx/200))
        # kpsidx=idx-imgidx*200
        # valid_data[imgidx][3][kpsidx,0]=0
        # valid_data[imgidx][3][kpsidx,1]=0
        # # # print(valid_pairs[imgidx])
        # # # print(kpsidx)
    trainSize =0
    validationSize = len(valid_pairs)
else:
    raise NotImplementedError

print('Using {}s'.format(default_timer() - t0))
sys.stdout.flush()



print('data read, train {} val {}'.format(trainSize, validationSize))
sys.stdout.flush()

batch_size=args.batch
assert batch_size % len(ctx) == 0
batch_size_card = batch_size // len(ctx)

sys.stdout.flush()

lmk_dist_valids,lmk_dist_oris,lmk_dist_oris_valid= pipe.validate_Ttest(valid_data)
lmk_dist_oris2=lmk_dist_oris[np.where(lmk_dist_oris>0)[0]]
savepath='/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/'+checkpoint.split('/')[-1].split('.')[0]+'.mat'
scio.savemat(savepath,{'data':lmk_dist_valids})
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/795Feb03-1544_1559.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/7d9Feb07-1704_1419.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/6a6Feb12-2057_5415.mat")
lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/11fFeb10-2002_127.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/7e6Jan31-1230_4512.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/7e6Jan31-1230_4512.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/8ccFeb18-1251_986.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/068Feb28-0911_1812.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/f54Feb01-1158_11466.mat")
# lmk_dist_comparison=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_Ttest/0c5Feb25-1003_7.mat")
lmk_dist_comparison=lmk_dist_comparison['data']
tstat, pval = scipy.stats.ttest_ind(a=np.squeeze(lmk_dist_valids), b=np.squeeze(lmk_dist_comparison),)
print(tstat)
print(pval)
pdb.set_trace()
sys.exit(0)

