import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
import pdb
from .MaskFlownet import *
# from .MaskFlownet_attention import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth
import copy
import skimage.io
import os
import pandas as pd
import random
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .MI import *
import cv2
import time
import scipy.io as scio

def build_network(name):
    return eval(name)

def get_coords(img):
    shape = img.shape
    range_x = nd.arange(shape[2], ctx = img.context).reshape(shape = (1, 1, -1, 1)).tile(reps = (shape[0], 1, 1, shape[3]))
    range_y = nd.arange(shape[3], ctx = img.context).reshape(shape = (1, 1, 1, -1)).tile(reps = (shape[0], 1, shape[2], 1))
    return nd.concat(range_x, range_y, dim = 1)

arange6=30*2
arange5=25*2
arange4=20*2
arange3=15*2
arange2=10*2
class PipelineFlownet:
    _lr = None

    def __init__(self, ctx, config):
        self.ctx = ctx
        self.network = build_network(getattr(config.network, 'class').get('MaskFlownet'))(config=config)
        self.network.hybridize()
        self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4})
        self.strides = self.network.strides or [64, 32, 16, 8, 4]

        self.scale = self.strides[-1]
        self.upsampler = Upsample(self.scale)
        self.upsampler_mask = Upsample(self.scale)

        self.epeloss = EpeLoss()
        self.epeloss.hybridize()
        self.epeloss_with_mask = EpeLossWithMask()
        self.epeloss_with_mask.hybridize()

        ## start:gl add loss function
        self.raw_weight = 1
        self.raw_loss_op = CorrelationLoss()
        self.raw_loss_op.hybridize()

        self.regularization_op = RegularizatonLoss()
        self.regularization_op.hybridize()
        self.reg_weight = config.optimizer.regularization.get(0)
        self.boundary_loss_op = BoundaryLoss()
        self.boundary_loss_op.hybridize()
        self.boundary_weight = config.optimizer.boundary.get(0)
        ## end: gl add loss function

        multiscale_weights = config.network.mw.get([.005, .01, .02, .08, .32])
        if len(multiscale_weights) != 5:
            multiscale_weights = [.005, .01, .02, .08, .32]
        self.multiscale_epe = MultiscaleEpe(
            scales = self.strides, weights = multiscale_weights, match = 'upsampling',
            eps = 1e-8, q = config.optimizer.q.get(None))
        self.multiscale_epe.hybridize()

        self.reconstruction = Reconstruction2DSmooth(3)
        self.reconstruction.hybridize()

        self.lr_schedule = config.optimizer.learning_rate.value

    def save(self, prefix):
        self.network.save_parameters(prefix + '.params')
        self.trainer.save_states(prefix + '.states')
    
    def load(self, checkpoint):
        self.network.load_parameters(checkpoint, ctx=self.ctx)

    def load_head(self, checkpoint):
        self.network.load_head(checkpoint, ctx=self.ctx)

    def fix_head(self):
        self.network.fix_head()

    def set_learning_rate(self, steps):
        i = 0
        while i < len(self.lr_schedule) and steps > self.lr_schedule[i][0]:
            i += 1
        try:
            lr = self.lr_schedule[i][1]
        except IndexError:
            return False
        self.trainer.set_learning_rate(lr)
        self._lr = lr
        return True 

    @property
    def lr(self):
        return self._lr

    def loss(self, pred, occ_masks, labels, masks):
        loss = self.multiscale_epe(labels, masks, *pred)
        return loss
    def centralize_kp_pairs_multiscale(self, img1, img2):
        rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
        return img1 - img1.mean(axis = (2, 3)).reshape((-2, 1, 1)), img2 - img2.mean(axis = (2, 3)).reshape((-2, 1, 1)), rgb_mean
    def centralize(self, img1, img2):
        rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def train_batch_F_KCG(self, dist_weight, img1, img2,kp1_sfgs,kp2_sfgs):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2,kp1_sfgs,kp2_sfgs = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,kp1_sfgs,kp2_sfgs))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,kp1_sfg,kp2_sfg in zip(img1, img2,kp1_sfgs,kp2_sfgs):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow)
                dist_loss2, _, _ = self.landmark_dist(kp1_sfg, kp2_sfg, flows)
                dist_weight2=30
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight+dist_loss2*dist_weight2
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses2.append(dist_loss2*dist_weight2)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2]))}
    
    
    def train_batch_baseline(self, dist_weight, img1, img2):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s in zip(img1, img2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s)
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow)
                self.raw_weight=1
                self.reg_weight = 1
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
    
    
    def train_batch_KCG(self, dist_weight, img1, img2,sift1s,sift2s, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, sift1s, sift2s, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, sift1s, sift2s, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, sift1, sift2, kp1_gt,kp2_gt,kp1_sfg,kp2_sfg in zip(img1, img2, sift1s, sift2s, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow)
                dist_loss, _, _ = self.landmark_dist(kp1_gt, kp2_gt, flows)
                dist_loss2, _, _ = self.landmark_dist(kp1_sfg, kp2_sfg, flows)
                dist_weight = 30#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=30
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight+ dist_loss*dist_weight +dist_loss2*dist_weight2
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2]))}

    def train_batch_LF_KCG(self, dist_weight, img1, img2, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, kp1_gt,kp2_gt,kp1_sfg,kp2_sfg in zip(img1, img2, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow)
                dist_loss, _, _ = self.landmark_dist(kp1_gt, kp2_gt, flows)
                dist_loss2, _, _ = self.landmark_dist(kp1_sfg, kp2_sfg, flows)
                dist_weight = 30#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=30
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight+ dist_loss*dist_weight +dist_loss2*dist_weight2
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2]))}

    
    
    
    def train_batch_IKCG(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,steps):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s)
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow)
                self.raw_weight=1
                self.reg_weight = 1
                dist_weight = 30
                dist_weight2=30
                dist_weight3=30
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))}
    
    
    def landmark_dist(self, lmk1, lmk2, flows):
        lmk1=lmk1[:,:,0:2]
        lmk2=lmk2[:,:,0:2]
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)#mean rtre######问题在于点的数目取mean
            lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*(np.shape(lmk2)[1]) # 消除当kp数目为0的时候的影响,考虑到batch可能大于1
            lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            return lmk_dist / (shape[0]*1.414), warped_lmk, lmk2
        else:
            return 0, [], []
    
    
    def rebuttle_kp_pairs_multiscale(self, dist_weight, img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s,name_num):
        img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        if 1:
        # with autograd.record():
            for img1s, img2s,img1s_256, img2s_256,img1s_1024, img2s_1024,orb1,orb2 in zip(img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s):
                if 1:
                    shape = img1s.shape
                    savepath1="./KCG_ite1_multiscale_kps_0.99_0.85/"
                    savepath3="./KCG_ite1_multiscale_kps_0.99_0.85/"
                    # savepath1="./KCG_ite2_multiscale_kps_0.99_0.85/"
                    # savepath3="./KCG_ite2_multiscale_kps_0.99_0.85/"
                    # savepath1="./KCG_ite3_multiscale_kps_0.995_0.75/"
                    # savepath3="./KCG_ite3_multiscale_kps_0.995_0.75/"

                    if not os.path.exists(savepath1):
                        os.mkdir(savepath1)
                    if not os.path.exists(savepath3):
                        os.mkdir(savepath3)
                    orb2_numpy=orb2.squeeze().asnumpy()
                    orb1_numpy=orb1.squeeze().asnumpy()
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp1=lmk_temp#.tolist()
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp2=lmk_temp#.tolist()
                    except:
                        pass
                    else:
                        dis_temp=np.sqrt(np.expand_dims((orb1_numpy**2).sum(1),1)+np.expand_dims((lmk_temp1**2).sum(1),0)-2*np.dot(orb1_numpy,lmk_temp1.transpose((1,0))))
                        orb1=orb1[:,np.where(dis_temp.min(1)>5)[0],:]
                        dis_temp=np.sqrt(np.expand_dims((orb2_numpy**2).sum(1),1)+np.expand_dims((lmk_temp2**2).sum(1),0)-2*np.dot(orb2_numpy,lmk_temp2.transpose((1,0))))
                        orb2=orb2[:,np.where(dis_temp.min(1)>5)[0],:]
                    
                    print('features extracting:{}'.format(name_num))
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=nd.zeros((15,1568*2,shape[0],orb1.shape[1]),ctx=img1s.context)
                    desc2s=nd.zeros((15,1568*2,shape[0],orb2.shape[1]),ctx=img1s.context)
                    for k in range (shape[0]):
                        for i in range(int(np.ceil(orb1.shape[1]/2))):
                            kp1_x1,kp1_y1=orb1[k,2*i,1].asnumpy(),orb1[k,2*i,0].asnumpy()
                            patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb1[k,2*i+1,1].asnumpy(),orb1[k,2*i+1,0].asnumpy()
                                patch_img1s_2=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)###(15,3,:,:)
                            except:
                                patch_img1s_2=patch_img1s
                            patch_img1s, patch_img1s_2, _ = self.centralize_kp_pairs_multiscale(patch_img1s, patch_img1s_2)#######(15,3,:,:)
                            patch_img1s_all=nd.concat(patch_img1s,mx.image.imrotate(patch_img1s,90),mx.image.imrotate(patch_img1s,180),mx.image.imrotate(patch_img1s,270),\
                                mx.image.imrotate(patch_img1s,45),mx.image.imrotate(patch_img1s,135),mx.image.imrotate(patch_img1s,225),mx.image.imrotate(patch_img1s,315),\
                                mx.image.imrotate(patch_img1s,22.5),mx.image.imrotate(patch_img1s,67.5),mx.image.imrotate(patch_img1s,112.5),mx.image.imrotate(patch_img1s,157.5),\
                                mx.image.imrotate(patch_img1s,202.5),mx.image.imrotate(patch_img1s,247.5),mx.image.imrotate(patch_img1s,292.5),mx.image.imrotate(patch_img1s,337.5),dim=0)###(15*16,3,:,:)
                            patch_img1s_2_all=nd.concat(patch_img1s_2,mx.image.imrotate(patch_img1s_2,90),mx.image.imrotate(patch_img1s_2,180),mx.image.imrotate(patch_img1s_2,270),\
                                mx.image.imrotate(patch_img1s_2,45),mx.image.imrotate(patch_img1s_2,135),mx.image.imrotate(patch_img1s_2,225),mx.image.imrotate(patch_img1s_2,315),\
                                mx.image.imrotate(patch_img1s_2,22.5),mx.image.imrotate(patch_img1s_2,67.5),mx.image.imrotate(patch_img1s_2,112.5),mx.image.imrotate(patch_img1s_2,157.5),\
                                mx.image.imrotate(patch_img1s_2,202.5),mx.image.imrotate(patch_img1s_2,247.5),mx.image.imrotate(patch_img1s_2,292.5),mx.image.imrotate(patch_img1s_2,337.5),dim=0)###(15*16,3,:,:)
                            _, c1s, c2s,_= self.network(patch_img1s_all, patch_img1s_2_all)#15*16,196,1,1
                            desc1s[:,:,k,2*i]=nd.transpose(c1s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()
                            try:
                                desc1s[:,:,k,2*i+1]=nd.transpose(c2s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()#(5,196*8,1,1)
                            except:
                                pass
                        for i in range(int(np.ceil(orb2.shape[1]/2))):
                            kp1_x1,kp1_y1=orb2[k,2*i,1].asnumpy(),orb2[k,2*i,0].asnumpy()
                            patch_img2s=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb2[k,2*i+1,1].asnumpy(),orb2[k,2*i+1,0].asnumpy()
                                patch_img2s_2=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            except:
                                patch_img2s_2=patch_img2s
                            patch_img2s, patch_img2s_2, _ = self.centralize_kp_pairs_multiscale(patch_img2s, patch_img2s_2)
                            patch_img2s_all=nd.concat(patch_img2s,mx.image.imrotate(patch_img2s,90),mx.image.imrotate(patch_img2s,180),mx.image.imrotate(patch_img2s,270),\
                                mx.image.imrotate(patch_img2s,45),mx.image.imrotate(patch_img2s,135),mx.image.imrotate(patch_img2s,225),mx.image.imrotate(patch_img2s,315),\
                                mx.image.imrotate(patch_img2s,22.5),mx.image.imrotate(patch_img2s,67.5),mx.image.imrotate(patch_img2s,112.5),mx.image.imrotate(patch_img2s,157.5),\
                                mx.image.imrotate(patch_img2s,202.5),mx.image.imrotate(patch_img2s,247.5),mx.image.imrotate(patch_img2s,292.5),mx.image.imrotate(patch_img2s,337.5),dim=0)###(15*16,3,:,:)
                            patch_img2s_2_all=nd.concat(patch_img2s_2,mx.image.imrotate(patch_img2s_2,90),mx.image.imrotate(patch_img2s_2,180),mx.image.imrotate(patch_img2s_2,270),\
                                mx.image.imrotate(patch_img2s_2,45),mx.image.imrotate(patch_img2s_2,135),mx.image.imrotate(patch_img2s_2,225),mx.image.imrotate(patch_img2s_2,315),\
                                mx.image.imrotate(patch_img2s_2,22.5),mx.image.imrotate(patch_img2s_2,67.5),mx.image.imrotate(patch_img2s_2,112.5),mx.image.imrotate(patch_img2s_2,157.5),\
                                mx.image.imrotate(patch_img2s_2,202.5),mx.image.imrotate(patch_img2s_2,247.5),mx.image.imrotate(patch_img2s_2,292.5),mx.image.imrotate(patch_img2s_2,337.5),dim=0)###(15*16,3,:,:)
                            _, c1s, c2s,_= self.network(patch_img2s_all, patch_img2s_2_all)#15*16,196,1,1
                            desc2s[:,:,k,2*i]=nd.transpose(c1s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()
                            try:
                                desc2s[:,:,k,2*i+1]=nd.transpose(c2s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()#(5,196*8,1,1)
                            except:
                                pass
                    del patch_img1s,patch_img1s_2,patch_img2s,patch_img2s_2,c1s,c2s
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    normalized_desc1s=nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(15,N,K1,196*16)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(15,N,196*16,K2)
                    del desc1s,desc2s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(15,N,K1,K2)
                    del normalized_desc1s
                    sim_mat_12=nd.mean(sim_mats,0)#(N,K1,K2)
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2
                    # mask_zone1=dis>=(0.04**2)*(shape[2]**2)*2#0.028 0.015 0.04
                    # mask_zone2=dis<(0.11**2)*(shape[2]**2)*2#0.015 0.04
                    # mask_zone=mask_zone1*mask_zone2==1
                    
                    mid_indices, mask12 = self.associate(sim_mat_12*mask_zone,orb2)#(N,K1,1)
                    max_indices,mask21 = self.associate(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K2,1)
                    indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K1
                    indices_2 = indices*mask12.squeeze(2)#N,K
                    mask=nd.broadcast_equal(indices,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask2=nd.broadcast_equal(indices_2,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask=mask*mask2==1##N,K
                    print(mask.sum())
                    mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K1,2)
                    coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    mid_sim_mat_21_warp=nd.diag(nd.diag(nd.gather_nd(nd.swapaxes(sim_mat_21,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)),axis1=1,axis2=2)###(N,K1)
                    weights=mid_sim_mat_21_warp*mask###(N,K1)
                    coor1=nd.concat(coor1,weights.expand_dims(2),dim=2)#.asnumpy().tolist()#####weighted   (N,K1,3)
                    coor2=nd.concat(coor2,weights.expand_dims(2),dim=2)#.asnumpy().tolist()#####weighted   (N,K1,3)
                    time3=time.time()
                    print(time3-time2)

                    for k in range (shape[0]):
                        index_nonzero=np.where(nd.elemwise_mul(nd.elemwise_mul(coor1[0,:,0],coor1[0,:,1]),nd.elemwise_mul(coor2[0,:,0],coor2[0,:,1])).asnumpy()!=0)[0]
                        if index_nonzero.shape[0]>0:
                            kp1=coor1[k,index_nonzero,:].asnumpy().tolist()
                            kp2=coor2[k,index_nonzero,:].asnumpy().tolist()
                        else:
                            kp1=[]
                            kp2=[]
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1,3]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1,3]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                    del sim_mat_12,sim_mat_21,dis,mask_zone,mid_indices,mask12,max_indices,mask21,indices,indices_2
                    del mask,mask2,mid_orb_warp

                    im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                    if len(kp1)>0:
                        plt.figure()
                        plt.imshow(im1)
                        plt.plot([np.array(kp1)[:,1],np.array(kp2)[:,1]+shape[2]],[np.array(kp1)[:,0],np.array(kp2)[:,0]], '#FF0033',linewidth=0.5)
                        plt.title(str(name_num))
                        plt.savefig(savepath1+str(name_num)+'_'+str(len(kp1))+'.jpg', dpi=600)
                        plt.close()
                        name = ['X', 'Y','W']
                        outlmk1 = pd.DataFrame(columns=name, data=np.asarray(kp1)[:,[1,0,2]])
                        outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                        outlmk2 = pd.DataFrame(columns=name, data=np.asarray(kp2)[:,[1,0,2]])
                        outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
        return normalized_desc2s
    
    
    
    
    
    
    def rebuttle_kp_pairs_multiscale_speed(self, dist_weight, img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s,name_num,normalized_desc2s):
        img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        if 1:
        # with autograd.record():
            for img1s, img2s,img1s_256, img2s_256,img1s_1024, img2s_1024,orb1,orb2 in zip(img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s):
                if 1:
                    shape = img1s.shape
                    savepath1="./KCG_ite1_multiscale_kps_0.99_0.85/"
                    savepath3="./KCG_ite1_multiscale_kps_0.99_0.85/"
                    # savepath1="./KCG_ite2_multiscale_kps_0.99_0.85/"
                    # savepath3="./KCG_ite2_multiscale_kps_0.99_0.85/"
                    # savepath1="./KCG_ite3_multiscale_kps_0.995_0.75/"
                    # savepath3="./KCG_ite3_multiscale_kps_0.995_0.75/"
                    if not os.path.exists(savepath1):
                        os.mkdir(savepath1)
                    if not os.path.exists(savepath3):
                        os.mkdir(savepath3)
                    orb2_numpy=orb2.squeeze().asnumpy()
                    orb1_numpy=orb1.squeeze().asnumpy()
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp1=lmk_temp#.tolist()
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp2=lmk_temp#.tolist()
                    except:
                        pass
                    else:
                        dis_temp=np.sqrt(np.expand_dims((orb1_numpy**2).sum(1),1)+np.expand_dims((lmk_temp1**2).sum(1),0)-2*np.dot(orb1_numpy,lmk_temp1.transpose((1,0))))
                        orb1=orb1[:,np.where(dis_temp.min(1)>5)[0],:]
                        dis_temp=np.sqrt(np.expand_dims((orb2_numpy**2).sum(1),1)+np.expand_dims((lmk_temp2**2).sum(1),0)-2*np.dot(orb2_numpy,lmk_temp2.transpose((1,0))))
                        orb2=orb2[:,np.where(dis_temp.min(1)>5)[0],:]
                    
                    print('features extracting:{}'.format(name_num))
                    
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=nd.zeros((15,1568*2,shape[0],orb1.shape[1]),ctx=img1s.context)
                    for k in range (shape[0]):
                        for i in range(int(np.ceil(orb1.shape[1]/2))):
                            kp1_x1,kp1_y1=orb1[k,2*i,1].asnumpy(),orb1[k,2*i,0].asnumpy()
                            patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb1[k,2*i+1,1].asnumpy(),orb1[k,2*i+1,0].asnumpy()
                                patch_img1s_2=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange2)),0):min(int((kp1_y1/2+arange2)),int(shape[2]/2)),max(int((kp1_x1/2-arange2)),0):min(int((kp1_x1/2+arange2)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange3)),0):min(int((kp1_y1/2+arange3)),int(shape[2]/2)),max(int((kp1_x1/2-arange3)),0):min(int((kp1_x1/2+arange3)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange4)),0):min(int((kp1_y1/2+arange4)),int(shape[2]/2)),max(int((kp1_x1/2-arange4)),0):min(int((kp1_x1/2+arange4)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange5)),0):min(int((kp1_y1/2+arange5)),int(shape[2]/2)),max(int((kp1_x1/2-arange5)),0):min(int((kp1_x1/2+arange5)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_256[k:(k+1),:,max(int((kp1_y1/2-arange6)),0):min(int((kp1_y1/2+arange6)),int(shape[2]/2)),max(int((kp1_x1/2-arange6)),0):min(int((kp1_x1/2+arange6)),int(shape[2]/2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange2)),0):min(int((kp1_y1*2+arange2)),int(shape[2]*2)),max(int((kp1_x1*2-arange2)),0):min(int((kp1_x1*2+arange2)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange3)),0):min(int((kp1_y1*2+arange3)),int(shape[2]*2)),max(int((kp1_x1*2-arange3)),0):min(int((kp1_x1*2+arange3)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange4)),0):min(int((kp1_y1*2+arange4)),int(shape[2]*2)),max(int((kp1_x1*2-arange4)),0):min(int((kp1_x1*2+arange4)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange5)),0):min(int((kp1_y1*2+arange5)),int(shape[2]*2)),max(int((kp1_x1*2-arange5)),0):min(int((kp1_x1*2+arange5)),int(shape[2]*2))],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)###(15,3,:,:)
                            except:
                                patch_img1s_2=patch_img1s######(15,3,:,:)
                            patch_img1s, patch_img1s_2, _ = self.centralize_kp_pairs_multiscale(patch_img1s, patch_img1s_2)#######(15,3,:,:)
                            patch_img1s_all=nd.concat(patch_img1s,mx.image.imrotate(patch_img1s,90),mx.image.imrotate(patch_img1s,180),mx.image.imrotate(patch_img1s,270),\
                                mx.image.imrotate(patch_img1s,45),mx.image.imrotate(patch_img1s,135),mx.image.imrotate(patch_img1s,225),mx.image.imrotate(patch_img1s,315),\
                                mx.image.imrotate(patch_img1s,22.5),mx.image.imrotate(patch_img1s,67.5),mx.image.imrotate(patch_img1s,112.5),mx.image.imrotate(patch_img1s,157.5),\
                                mx.image.imrotate(patch_img1s,202.5),mx.image.imrotate(patch_img1s,247.5),mx.image.imrotate(patch_img1s,292.5),mx.image.imrotate(patch_img1s,337.5),dim=0)###(15*16,3,:,:)
                            patch_img1s_2_all=nd.concat(patch_img1s_2,mx.image.imrotate(patch_img1s_2,90),mx.image.imrotate(patch_img1s_2,180),mx.image.imrotate(patch_img1s_2,270),\
                                mx.image.imrotate(patch_img1s_2,45),mx.image.imrotate(patch_img1s_2,135),mx.image.imrotate(patch_img1s_2,225),mx.image.imrotate(patch_img1s_2,315),\
                                mx.image.imrotate(patch_img1s_2,22.5),mx.image.imrotate(patch_img1s_2,67.5),mx.image.imrotate(patch_img1s_2,112.5),mx.image.imrotate(patch_img1s_2,157.5),\
                                mx.image.imrotate(patch_img1s_2,202.5),mx.image.imrotate(patch_img1s_2,247.5),mx.image.imrotate(patch_img1s_2,292.5),mx.image.imrotate(patch_img1s_2,337.5),dim=0)###(15*16,3,:,:)
                            _, c1s, c2s,_= self.network(patch_img1s_all, patch_img1s_2_all)#15*16,196,1,1
                            desc1s[:,:,k,2*i]=nd.transpose(c1s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()
                            try:
                                desc1s[:,:,k,2*i+1]=nd.transpose(c2s.reshape((-1,15,196,1,1)),axes=(1,0,2,3,4)).reshape(15,-1).squeeze()#(5,196*8,1,1)
                            except:
                                pass
                        
                    del patch_img1s,patch_img1s_2,c1s,c2s
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    normalized_desc1s=nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(15,N,K1,196*16)
                    del desc1s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(15,N,K1,K2)
                    del normalized_desc1s
                    sim_mat_12=nd.mean(sim_mats,0)#(N,K1,K2)
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # mask_zone1=dis>=(0.04**2)*(shape[2]**2)*2#0.028 0.015 0.04
                    # mask_zone2=dis<(0.11**2)*(shape[2]**2)*2#0.015 0.04
                    # mask_zone=mask_zone1*mask_zone2==1
                    
                    mid_indices, mask12 = self.associate(sim_mat_12*mask_zone,orb2)#(N,K1,1)
                    max_indices,mask21 = self.associate(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K2,1)
                    indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K1
                    indices_2 = indices*mask12.squeeze(2)#N,K
                    mask=nd.broadcast_equal(indices,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask2=nd.broadcast_equal(indices_2,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask=mask*mask2==1##N,K
                    print(mask.sum())
                    mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K1,2)
                    coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    mid_sim_mat_21_warp=nd.diag(nd.diag(nd.gather_nd(nd.swapaxes(sim_mat_21,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)),axis1=1,axis2=2)###(N,K1)
                    weights=mid_sim_mat_21_warp*mask###(N,K1)
                    coor1=nd.concat(coor1,weights.expand_dims(2),dim=2)#.asnumpy().tolist()#####weighted   (N,K1,3)
                    coor2=nd.concat(coor2,weights.expand_dims(2),dim=2)#.asnumpy().tolist()#####weighted   (N,K1,3)
                    time3=time.time()
                    print(time3-time2)
                    
                    for k in range (shape[0]):
                        index_nonzero=np.where(nd.elemwise_mul(nd.elemwise_mul(coor1[0,:,0],coor1[0,:,1]),nd.elemwise_mul(coor2[0,:,0],coor2[0,:,1])).asnumpy()!=0)[0]
                        if index_nonzero.shape[0]>0:
                            kp1=coor1[k,index_nonzero,:].asnumpy().tolist()
                            kp2=coor2[k,index_nonzero,:].asnumpy().tolist()
                        else:
                            kp1=[]
                            kp2=[]
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1,3]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1,3]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                    del sim_mat_12,sim_mat_21,dis,mask_zone,mid_indices,mask12,max_indices,mask21,indices,indices_2
                    del mask,mask2,mid_orb_warp

                    im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                    if len(kp1)>0:
                        plt.figure()
                        plt.imshow(im1)
                        plt.plot([np.array(kp1)[:,1],np.array(kp2)[:,1]+shape[2]],[np.array(kp1)[:,0],np.array(kp2)[:,0]], '#FF0033',linewidth=0.5)
                        plt.title(str(name_num))
                        plt.savefig(savepath1+str(name_num)+'_'+str(len(kp1))+'.jpg', dpi=600)
                        plt.close()
                        name = ['X', 'Y','W']
                        outlmk1 = pd.DataFrame(columns=name, data=np.asarray(kp1)[:,[1,0,2]])
                        outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                        outlmk2 = pd.DataFrame(columns=name, data=np.asarray(kp2)[:,[1,0,2]])
                        outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
        return normalized_desc2s

    
    def associate(self, sim_mat,fkp):
        #############sim_mat:(N,K1,K2)  fkp(N,K2,2)    
        indice = nd.stop_gradient(nd.topk(sim_mat, axis=2, k=2, ret_typ='indices'))#(N,K1,2)
        fkp_ref=nd.stop_gradient(nd.diag(nd.gather_nd(nd.swapaxes(fkp,0,1),nd.transpose(nd.slice_axis(indice,axis=2,begin=0,end=1),axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)))#(K,2,N)#(N,K,N,2)
        d_temp=nd.stop_gradient(nd.abs(nd.sum(fkp_ref*fkp_ref,axis=2,keepdims=True)+nd.sum(fkp*fkp,axis=2,keepdims=True).transpose((0,2,1))-2*nd.batch_dot(fkp_ref,fkp.transpose((0,2,1)))))#N,K,K
        mask_nms1=d_temp>=(0.004**2)*(1024**2)*2#0.005
        mask_nms2=d_temp==0
        mask_nms=nd.stop_gradient(((mask_nms1+mask_nms2)>=1))
        sim = nd.stop_gradient(nd.topk(sim_mat*mask_nms,axis=2, k=2, ret_typ='value'))
        mask1=nd.stop_gradient(nd.broadcast_lesser(nd.slice_axis(sim,axis=2,begin=1,end=2),(nd.slice_axis(sim,axis=2,begin=0,end=1)*0.97)))#0.97
        mask2=nd.stop_gradient(nd.slice_axis(sim,axis=2,begin=0,end=1)>0.9)#0.9#.85#.9
        mask=nd.stop_gradient(mask1*mask2==1)
        return indice[:,:,0:1],mask#(N,K,1)

    def appendimages(self,im1, im2):
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
