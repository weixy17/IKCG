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
repoRoot = r'.'
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
import numpy as np
import mxnet as mx

# data readers
from reader.ACROBAT import *
import cv2

model_parser = argparse.ArgumentParser(add_help=False)
training_parser = argparse.ArgumentParser(add_help=False)
training_parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')

parser = argparse.ArgumentParser(parents=[model_parser, training_parser])

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
parser.add_argument('--relative', type=str, default="")
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
parser.add_argument('--gkps', action='store_true', help='Generate key point pairs')
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
        checkpoint = "./weights/"+run_id+'_'+steps+'.params'
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
    print('load the weight for the network')
    pipe.load(checkpoint)
    sys.stdout.flush()
    if not args.valid and not args.predict:
        pipe.trainer.step(100, ignore_stale_grad=True)
        pipe.trainer.load_states(checkpoint.replace('params', 'states'))

if dataset_cfg.dataset.value == 'ACROBAT':
    print('loading ACROBAT dataset...')
    t0 = default_timer()
    subset = dataset_cfg.subset.value
    print(subset)
    sys.stdout.flush()
    dataset, train_pairs, valid_pairs = LoadACROBAT_LF_KCG_multiscale512()
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
if not args.valid and not args.gkps:
    string='rawSIFT_1,reg_1,vgg16_30,LFS_30,ite1_30'############regularization parameters
    log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
    log.log('start={}, train={}, val={}, host={}, batch={},'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size)+string+',Load Checkpoint {}'.format(checkpoint))
    information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
    log.log(information)
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
minaa=float('inf')
count=0
import time
import copy
siftdir="./SIFT_features/train_4096_to512/"
while True:
    if args.valid:
        aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,mMI,sMI = pipe.validate_ACROBAT(valid_data)
        print("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},mMI={:.4f}({:.4f}),""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,mMI,sMI))
        sys.exit(0)
    if args.gkps:
        train_pairs_nums=[int(temp[0].split('_')[0]) for temp in train_pairs]
        train_pairs_nums_unique=np.unique(train_pairs_nums, return_index=False,axis=0)
        count=count+1
        for i in range (0,train_pairs_nums_unique.shape[0]):
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
    sift1 = np.zeros((batch_size, 384, 512,512))
    sift2 = np.zeros((batch_size, 384, 512,512))
    for i in range(batch_size):
        name_num = next(train_gen)
        need_to_put=[]
        for count,fid in enumerate(train_pairs[name_num]):
            need_to_put.append(dataset[fid])
        batch.append(need_to_put)
        img1name=train_pairs[name_num][0]
        img2name=train_pairs[name_num][1]
        sift1path=os.path.join(siftdir,img1name.split('.')[0]+'.mat')
        sift2path=os.path.join(siftdir,img2name.split('.')[0]+'.mat')
        siftdata1 = scio.loadmat(sift1path)["sift"].astype(float)
        siftdata2 = scio.loadmat(sift2path)["sift"].astype(float)
        sift1[i, :, :, :] = np.transpose(siftdata1, (2, 0, 1))
        sift2[i, :, :, :] = np.transpose(siftdata2, (2, 0, 1))
    batch=[np.stack(x, axis=0) for x in zip(*batch)]
    img1, img2,kp1,kp2,kp1_2,kp2_2 = [batch[i] for i in range(len(train_pairs[name_num]))]
    train_log = pipe.train_batch_KCG(args.weight,img1, img2,sift1,sift2,kp1,kp2,kp1_2,kp2_2)
    # update log
    if  steps<100 or steps % 10 == 0:
        train_avg.update(train_log)
        log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}= {}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))
    # do valiation
    if steps % 100 == 0:
        if validationSize > 0:
            aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,mMI,sMI = pipe.validate_ACROBAT(valid_data)
            string_log=("""steps= {} aa={:.4f}({:.4f}),ma={:.4f}, m75tha={:.4f}, m90tha={:.4f}, am={:.4f}({:.4f}),mm={:.4f},mMI={:.4f}({:.4f}),""".format(steps,aa,aa_std,ma,m75tha,m90tha,am,am_std,mm,mMI,sMI))
            log.log(string_log)
            if aa < minaa: 
                minaa = aa
                prefix = os.path.join(repoRoot, './weights', '{}_{}'.format(run_id, steps))
                pipe.save(prefix)
                checkpoints.append(prefix)
                #########remove the older checkpoints
                while len(checkpoints) > 5:
                    prefix = checkpoints[0]
                    checkpoints = checkpoints[1:]
                    remove_queue.put(prefix + '.states')
                    remove_queue.put(prefix + '.params')
            