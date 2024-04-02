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
    dataset, train_pairs, valid_pairs = LoadACROBAT_DFS_SFG_multiscale512()
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




sys.stdout.flush()

# create log file
if not args.valid:
    string='raw_1,reg_1,vgg16_0,LFS_0'
    log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
    log_partial = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}-partial.log'.format(run_id)))
    log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
    log_partial.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
    information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
    log.log(information)
    log_partial.log(information)
    print(string)

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
minmm90th=float('inf')
minam90th=float('inf')
minaa=float('inf')
minma=float('inf')
count=0
import time
import copy

while True:
    if args.valid:
        checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
        aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
            mMI,sMI,mm75th,mm90th = pipe.validate_ACROBAT(valid_data)
        print("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},am75th={:.4f}({:.4f}),m75thm75th={:.4f},\
am90th={:.4f}({:.4f}),m90thm90th={:.4f},large1={:.4f}({:.4f}),large2={:.4f}({:.4f}),large3={:.4f}({:.4f}),large4={:.4f}({:.4f}),\
large5={:.4f}({:.4f}),mMI={:.4f}({:.4f}),mm75th={:.4f},mm90th={:.4f},""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,\
                    large3,large3_std,large4,large4_std,large5,large5_std,mMI,sMI,mm75th,mm90th))
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
    img1, img2,kp1,kp2,kp1_2,kp2_2= [batch[i] for i in range(6)]
    train_log = pipe.train_batch_supervised_LFS_SFG(args.weight,img1, img2,kp1,kp2,kp1_2,kp2_2)
    # update log
    if  1:#steps<100 or steps % 10 == 0:
        train_avg.update(train_log)
        log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))
    # do valiation
    # if steps % 100 == 0 or steps > 10000:
    if 1:
        if validationSize > 0:
            aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,large3,large3_std,large4,large4_std,large5,large5_std,\
            mMI,sMI,mm75th,mm90th = pipe.validate_ACROBAT(valid_data)
            string_log=("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},am75th={:.4f}({:.4f}),m75thm75th={:.4f},\
am90th={:.4f}({:.4f}),m90thm90th={:.4f},large1={:.4f}({:.4f}),large2={:.4f}({:.4f}),large3={:.4f}({:.4f}),large4={:.4f}({:.4f}),\
large5={:.4f}({:.4f}),mMI={:.4f}({:.4f}),mm75th={:.4f},mm90th={:.4f},""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,am75th,am75th_std,m75thm75th,am90th,am90th_std,m90thm90th,large1,large1_std,large2,large2_std,\
                    large3,large3_std,large4,large4_std,large5,large5_std,mMI,sMI,mm75th,mm90th))
            log.log(string_log)
            if aa<95 and aa>=88.1873 and ma>=61.4732 and m75tha>=98.8262 and m90tha>=201.7828 and am>=63.0877 and mm>=46.5192 and am75th>=100.0237 and m75thm75th>=121.5771\
                 and am90th>=173.3430 and m90thm90th>=358.3577 and large1>=153.2274 and large2>=176.9050 and large3>=197.5885 and large4>=228.7451 and large5>=465.8333 and mMI<=0.6929\
                 and mm75th>=70.9302 and mm90th>=99.1460:
            # if aa < 89.9447 and ma<57.7633 and m75tha<97.1213 and m90tha<209.7879 and am<68.4549 and mm<45.7838 and am75th<105.6889 and m75thm75th<116.5427\
                # and am90th<166.2152 and m90thm90th<380.8161 and large1<67.7344 and large2<37.2186 and large3<28.2852 and large4<17.5468 and large5<10.7378\
                # and mMI<0.6868 and mm75th<65.8447 and mm90th<102.6337: 
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # if aa<=86:
            # if aa<=85.8776 and ma<=59.4125 and m75tha<=97.4855 and m90tha<=182.0828 and am<=60.8917 and mm<=41.5041 and am75th<=99.4257 and m75thm75th<115.4043\
                # and am90th<=165.0903 and m90thm90th<=369.1824 and large1<=155.3453 and large2<=182.4061 and large3<=203.9012 and large4<=241.9868 and large5<=450.7090\
                # and mMI>=0.6948 and mm75th<=62.0117 and mm90th<=101.9141:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
            # if aa<=88 and ma<=62.0071 and m75tha<=110.4108 and m90tha<=205.9262 and am<=68.7343 and mm<=47.9317\
                # and am75th<=107.0342 and m75thm75th<=134.0805 and am90th<=173.4238 and m90thm90th<=381.4566 and large1<=156.6180\
                # and large2<=184.7221 and large3<=206.2061 and large4<=246.7777 and large5<=501.6511 and mMI>=0.6824 and mm75th<=68.2964\
                # and mm90th<=102.4616:
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
                # if output
            # if aa<=92.4098 and ma<=62.0071 and m75tha<=110.4108 and m90tha<=205.9262 and am<=68.7343 and mm<=47.9317\
                # and am75th<=107.0342 and m75thm75th<=134.0805 and am90th<=173.4238 and m90thm90th<=381.4566 and large1<=156.6180\
                # and large2<=184.7221 and large3<=206.2061 and large4<=246.7777 and large5<=501.6511 and mMI>=0.6824 and mm75th<=68.2964\
                # and mm90th<=102.4616:
                # if aa>=85.8776 and ma>=59.4125 and m75tha>=97.4855 and m90tha>=182.0828 and am>=60.8917 and mm>=41.5041 and am75th>=99.4257\
                    # and m75thm75th>115.4043 and am90th>=165.0903 and m90thm90th>=369.1824 and large1>=155.3453 and large2>=182.4061\
                    # and large3>=203.9012 and large4>=241.9868 and large5>=450.7090 and mMI<=0.6948 and mm75th>=62.0117 and mm90th>=101.9141:
            # if aa < 86 and ma<61 and m75tha<100 and m90tha<200 and am<63 and mm<45 and am75th<100 and m75thm75th<117\
                # and am90th<165 and m90thm90th<375 and large1<151.7147 and large2<177.8506 and large3<197.4440 and large4<232.2027 and large5<433.5936\
                # and mMI>0.6868 and mm75th<66 and mm90th<103: 
                    prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                    pipe.save(prefix)
                    print(string_log)
                    log_partial.log(string_log)
            # else:
            # if aa < minaa: # gl, keep the best model for test dataset.
                # minaa = aa
                # if ma<minma:
                    # minma = ma
                # if mm90th<minmm90th:
                    # minmm90th = mm90th
                # if am90th<minam90th:
                    # minam90th = am90th
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
                # checkpoints.append(prefix)
                # #########remove the older checkpoints
                # while len(checkpoints) > 40:
                    # prefix = checkpoints[0]
                    # checkpoints = checkpoints[1:]
                    # remove_queue.put(prefix + '.params')
                    # remove_queue.put(prefix + '.states')
            # elif ma<minma:
                # minma = ma
                # if mm90th<minmm90th:
                    # minmm90th = mm90th
                # if am90th<minam90th:
                    # minam90th = am90th
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
                # checkpoints.append(prefix)
                # #########remove the older checkpoints
                # while len(checkpoints) > 40:
                    # prefix = checkpoints[0]
                    # checkpoints = checkpoints[1:]
                    # remove_queue.put(prefix + '.params')
                    # remove_queue.put(prefix + '.states')
            # elif mm90th<minmm90th:
                # minmm90th = mm90th
                # if am90th<minam90th:
                    # minam90th = am90th
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
                # checkpoints.append(prefix)
                # #########remove the older checkpoints
                # while len(checkpoints) > 40:
                    # prefix = checkpoints[0]
                    # checkpoints = checkpoints[1:]
                    # remove_queue.put(prefix + '.params')
                    # remove_queue.put(prefix + '.states')
            # elif am90th<minam90th:
                # minam90th = am90th
                # prefix = os.path.join(repoRoot, 'weights_TMI_rebuttle', '{}_{}'.format(run_id, steps))
                # pipe.save(prefix)
                # print(string_log)
                # log_partial.log(string_log)
                # checkpoints.append(prefix)
                # #########remove the older checkpoints
                # while len(checkpoints) > 40:
                    # prefix = checkpoints[0]
                    # checkpoints = checkpoints[1:]
                    # remove_queue.put(prefix + '.params')
                    # remove_queue.put(prefix + '.states')
                
                
