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
select_num=8
search_range=16
step_range=8
def build_network(name):
    return eval(name)

def get_coords(img):
    shape = img.shape
    range_x = nd.arange(shape[2], ctx = img.context).reshape(shape = (1, 1, -1, 1)).tile(reps = (shape[0], 1, 1, shape[3]))
    range_y = nd.arange(shape[3], ctx = img.context).reshape(shape = (1, 1, 1, -1)).tile(reps = (shape[0], 1, shape[2], 1))
    return nd.concat(range_x, range_y, dim = 1)

orb = cv2.ORB_create(800, scaleFactor=1.5)
orb_2 = cv2.ORB_create(select_num, scaleFactor=1.5,nlevels=3,edgeThreshold=2,patchSize=2)#,patchSize=16)
arange6=30
arange5=25
arange4=20
arange3=15
arange2=10
# arange6=15
# arange5=10
# arange4=10
# arange3=5
# arange2=5
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
    def load_from_MaskFlownet(self, checkpoint):
        # save_dict = mx.nd.load(checkpoint)
        # new_params = {}   
        # for k, v in save_dict.items(): 
            # # pdb.set_trace()
            # if len(k.split('maskflownet_s0_'))==2:
                # new_name=k.split('maskflownet_s0_')[1]
                # new_params[new_name] = v  
        # nd.save('/data/wxy/association/Maskflownet_association_1024/test.params',new_params)
        self.network.load_parameters("/data/wxy/association/Maskflownet_association_1024/test.params", ctx=self.ctx)
        # self.network.load_dict(new_params, ctx=self.ctx)
    def load(self, checkpoint):
        self.network.load_parameters(checkpoint, ctx=self.ctx)
        # self.network.load_head("/data/wxy/association/Maskflownet_association_1024/weights/e6eMay08-2235_2100.params", ctx=self.ctx)

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
    
    def addsiftkp(self,kp):
        ## init , raw 0 is 0, 这样后面匹配到第一个点的匹配不会被当成不匹配而消去
        # coords.append([0, 0])
        coords=[]
        for i in range(np.shape(kp)[0]):
            siftco = [int(round(kp[i].pt[1])), int(round(kp[i].pt[0]))]
            #siftco = array(siftco)
            if siftco not in coords:
                coords.append(siftco)
        return coords
    
    def addsiftkp2(self, kp,row,col):
        coords=[]
        for i in range(np.shape(kp)[0]):
            siftco = [int(round(kp[i].pt[1]))+step_range*row, int(round(kp[i].pt[0]))+step_range*col]
            if siftco not in coords:
                coords.append(siftco)
        return coords
    
    
    # def train_batch(self, dist_weight, img1, img2, lmk1s, lmk2s, sift1, sift2, color_aug, aug):
    def train_batch(self, dist_weight, img1, img2, lmk1s, lmk2s,orb1s,orb2s,name_num,steps):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2,orb1,orb2 in zip(img1, img2, lmk1s, lmk2s,orb1s,orb2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # warp = self.reconstruction(sift2s, flow)
                warp = self.reconstruction(img2s, flow)
                #############可以尝试将img2和warp输入到网络中看warp2和img2的误差
                flows = []
                flows.append(flow)
                # raw loss calculation
                # raw_loss = self.raw_loss_op(sift1s, warp)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=10
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                if name_num not in [206,207,374,375,376,377] and (steps-1)%100==0:
                    # ######################800_whole_image
                    # # orb_warp=nd.zeros((shape[0],800,2), ctx=flow.context)
                    # orb_maskflownet=nd.zeros((shape[0],800,2), ctx=flow.context)
                    # orb1=nd.zeros((shape[0],800,2), ctx=flow.context)
                    # for k in range (shape[0]):
                        # kp2, _ = orb.detectAndCompute((img2s[k,0,:,:]*255).asnumpy().astype('uint8'), None)
                        # kp1, _ = orb.detectAndCompute((img1s[k,0,:,:]*255).asnumpy().astype('uint8'), None)
                        # filtered_coords2s=self.addsiftkp(kp2)
                        # filtered_coords1s=self.addsiftkp(kp1)
                        # for m in range (orb1.shape[1]-len(filtered_coords2s)):
                            # filtered_coords2s.extend([[0,0]])
                        # for m in range (orb1.shape[1]-len(filtered_coords1s)):
                            # filtered_coords1s.extend([[0,0]])
                    
                        # plt.figure()
                        # plt.imshow(img2s[k,0,:,:].asnumpy())
                        # for i in range (len(filtered_coords2s)):
                            # plt.scatter(filtered_coords2s[i][1],filtered_coords2s[i][0],s=0.05)
                        # plt.savefig('/data/wxy/association/Maskflownet_association/images/img2s_key_points/'+str(steps)+'_'+str(k)+'_warp.jpg',dpi=600)
                        # plt.close()
                        # orb_maskflownet[k,:,:]=nd.array(filtered_coords2s,ctx=flow.context)
                        # orb1[k,:,:]=nd.array(filtered_coords1s,ctx=flow.context)
                    
                    # ###########################16s8
                    # # orb_maskflownet=nd.zeros((shape[0],orb1.shape[1],2), ctx=flow.context)
                    # orb_maskflownet=np.zeros((shape[0],orb1.shape[1],2))
                    # for k in range (shape[0]):
                        # filtered_coords2s=[]
                        # for i in range(int(np.floor((shape[2]-search_range)/step_range))):
                            # for j in range(int(np.floor((shape[2]-search_range)/step_range))):
                                # # select_roi=(warp[k,0,step_range*i:(step_range*i+search_range),step_range*j:(step_range*j+search_range)]*255).asnumpy().astype('uint8')
                                # select_roi=(img2s[k,0,step_range*i:(step_range*i+search_range),step_range*j:(step_range*j+search_range)]*255).asnumpy().astype('uint8')
                                # if np.abs((select_roi-select_roi.mean())).sum()>0:
                                    # kp2, _ = orb_2.detectAndCompute(select_roi, None)
                                    # if kp2!=():
                                        # filtered_coords2= self.addsiftkp2(kp2,i,j)
                                        # filtered_coords2s.extend(filtered_coords2)
                        # for m in range (orb1.shape[1]-len(filtered_coords2s)):
                            # filtered_coords2s.extend([[0,0]])
                        # # orb_maskflownet[k,:,:]=nd.array(filtered_coords2s,ctx=flow.context)
                        # orb_maskflownet[k,:,:]=np.array(filtered_coords2s)
                        # plt.figure()
                        # # plt.imshow(warp[k,0,:,:].asnumpy())
                        # plt.imshow(img2s[k,0,:,:].asnumpy())
                        # for i in range (orb1.shape[1]):
                            # # plt.scatter(orb_maskflownet[k,i,1].asnumpy(),orb_maskflownet[k,i,0].asnumpy(),s=0.05)
                            # plt.scatter(orb_maskflownet[k,i,1],orb_maskflownet[k,i,0],s=0.05)
                        # plt.savefig('/data/wxy/association/Maskflownet_association/images/img2s_key_points/'+str(steps)+'_'+str(k)+'_warp.jpg',dpi=600)
                    savepath1='/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_0.965_0.98_with_network_update/'
                    savepath3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.965_0.98_with_network_update/'
                    orb2_list=orb2.squeeze().asnumpy().tolist()
                    orb1_list=orb1.squeeze().asnumpy().tolist()
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp1=lmk_temp.tolist()
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp2=lmk_temp.tolist()
                    except:
                        pass
                    else:
                        # pdb.set_trace()
                        for i in range(len(lmk_temp1)):
                            if lmk_temp1[i] in orb1_list:
                                orb1_list.remove(lmk_temp1[i])
                            if lmk_temp2[i] in orb2_list:
                                orb2_list.remove(lmk_temp2[i])
                        orb2=nd.expand_dims(nd.array(np.asarray(orb2_list),ctx=flow.context),axis=0)
                        orb1=nd.expand_dims(nd.array(np.asarray(orb1_list),ctx=flow.context),axis=0)
                    print('features extracting')
                    if orb2.shape[1]>8000:
                        if steps%2==0:
                            orb2=orb2[:,:int(orb2.shape[1]/2),:]
                        else:
                            orb2=orb2[:,int(orb2.shape[1]/2):,:]
                    if orb1.shape[1]>8000:
                        if steps%2==0:
                            orb1=orb1[:,:int(orb1.shape[1]/2),:]
                        else:
                            orb1=orb1[:,int(orb1.shape[1]/2):,:]
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=np.zeros([5,1568,shape[0],orb1.shape[1]])
                    desc2s=np.zeros([5,1568,shape[0],orb2.shape[1]])
                    for k in range (shape[0]):
                        for i in range(int(np.ceil(orb1.shape[1]/2))):
                            kp1_x1,kp1_y1=orb1[k,2*i,1].asnumpy(),orb1[k,2*i,0].asnumpy()
                            patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb1[k,2*i+1,1].asnumpy(),orb1[k,2*i+1,0].asnumpy()
                                patch_img1s_2=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            except:
                                patch_img1s_2=patch_img1s
                            patch_img1s, patch_img1s_2, _ = self.centralize(patch_img1s, patch_img1s_2)
                            _, c1s, c2s,_= self.network(patch_img1s, patch_img1s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s,45), mx.image.imrotate(patch_img1s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s,90), mx.image.imrotate(patch_img1s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s,135), mx.image.imrotate(patch_img1s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s,180), mx.image.imrotate(patch_img1s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s,225), mx.image.imrotate(patch_img1s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s,270), mx.image.imrotate(patch_img1s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s,315), mx.image.imrotate(patch_img1s_2,315))#5,196,1,1
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            # pdb.set_trace()
                            desc1s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)
                            try:
                                desc1s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)#(5,196*8,1,1)
                            except:
                                pass
                            # if i==0:
                                # desc1=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)#(5,196*8,1,1)
                                # desc2=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)#(5,196*8,1,1)
                            # else:
                                # desc1=np.concatenate((desc1,np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)),3)#(5,196*8,1,K)
                                # desc2=np.concatenate((desc2,np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)),3)#(5,196*8,1,K)
                        # if k==0:
                            # desc1s=desc1
                            # desc2s=desc2
                        # else:
                            # desc1s=np.concatenate((desc1s,desc1),dim=2)#(5,196*8,N,K)
                            # desc2s=np.concatenate((desc2s,desc2),dim=2)#(5,196*8,N,K)
                        for i in range(int(np.ceil(orb2.shape[1]/2))):
                            kp1_x1,kp1_y1=orb2[k,2*i,1].asnumpy(),orb2[k,2*i,0].asnumpy()
                            patch_img2s=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb2[k,2*i+1,1].asnumpy(),orb2[k,2*i+1,0].asnumpy()
                                patch_img2s_2=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            except:
                                patch_img2s_2=patch_img2s

                            patch_img2s, patch_img2s_2, _ = self.centralize(patch_img2s, patch_img2s_2)
                            _, c1s, c2s,_= self.network(patch_img2s, patch_img2s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img2s,45), mx.image.imrotate(patch_img2s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img2s,90), mx.image.imrotate(patch_img2s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img2s,135), mx.image.imrotate(patch_img2s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img2s,180), mx.image.imrotate(patch_img2s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img2s,225), mx.image.imrotate(patch_img2s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img2s,270), mx.image.imrotate(patch_img2s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img2s,315), mx.image.imrotate(patch_img2s_2,315))#5,196,1,1
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            desc2s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)
                            try:
                                desc2s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)#(5,196*8,1,1)
                            except:
                                pass
                            # if i==0:
                                # desc1=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)#(5,196*8,1,1)
                                # desc2=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)#(5,196*8,1,1)
                            # else:
                                # desc1=np.concatenate((desc1,np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)),3)#(5,196*8,1,K)
                                # desc2=np.concatenate((desc2,np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)),3)#(5,196*8,1,K)
                        # if k==0:
                            # desc1s=desc1
                            # desc2s=desc2
                        # else:
                            # desc1s=np.concatenate((desc1s,desc1),dim=2)#(5,196*8,N,K)
                            # desc2s=np.concatenate((desc2s,desc2),dim=2)#(5,196*8,N,K)
                    
                    
                    
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    ################################GPU
                    desc1s=nd.array(desc1s,ctx=flow.context)#(5,196*8,N,K1)
                    desc2s=nd.array(desc2s,ctx=flow.context)#(5,196*8,N,K2)
                    normalized_desc1s = nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(5,N,K1,K2)
                    sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K1,K2)
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    # pdb.set_trace()
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # print(mask_zone.sum().asnumpy()[0])
                    # pdb.set_trace()
                    mid_indices, mask12 = self.associate(sim_mat_12*mask_zone,orb2)#(N,K1,1)
                    max_indices,mask21 = self.associate(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K2,1)
                    indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K1
                    indices_2 = indices*mask12.squeeze(2)#N,K
                    mask=nd.broadcast_equal(indices,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=flow.context),0))##N,K1
                    mask2=nd.broadcast_equal(indices_2,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=flow.context),0))##N,K1
                    mask=mask*mask2==1##N,K
                    print(mask.sum())
                    # pdb.set_trace()
                    # pdb.set_trace()
                    mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K1,2)
                    coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    
                    
                    # ################################CPU
                    # # pdb.set_trace()
                    # normalized_desc1s = np.transpose(desc1s/np.linalg.norm(desc1s,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    # normalized_desc2s = np.transpose(desc2s/np.linalg.norm(desc2s,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    # sim_mats = np.matmul(normalized_desc1s, normalized_desc2s)#(5,N,K,K)
                    # sim_mat_12=np.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K,K)
                    # sim_mat_21=sim_mat_12.transpose((0,2,1))#(N,K,K)
                    # ####orb1(N,K,2)    orb_warp(N,K,2)    
                    # # dis=np.absolute(np.sum(orb1.asnumpy()*orb1.asnumpy(),axis=2,keepdims=True)+np.transpose(np.sum(orb_maskflownet.asnumpy()*orb_maskflownet.asnumpy(),axis=2,keepdims=True),(0,2,1))-2*np.matmul(orb1.asnumpy(),np.transpose(orb_maskflownet.asnumpy(),(0,2,1))))#N,K,K
                    # # dis=np.absolute(np.sum(orb1.asnumpy()*orb1.asnumpy(),axis=2,keepdims=True)+np.transpose(np.sum(orb_maskflownet*orb_maskflownet,axis=2,keepdims=True),(0,2,1))-2*np.matmul(orb1.asnumpy(),np.transpose(orb_maskflownet,(0,2,1))))#N,K,K
                    # dis=np.absolute(np.sum(orb1.asnumpy()*orb1.asnumpy(),axis=2,keepdims=True)+np.transpose(np.sum(orb2.asnumpy()*orb2.asnumpy(),axis=2,keepdims=True),(0,2,1))-2*np.matmul(orb1.asnumpy(),np.transpose(orb2.asnumpy(),(0,2,1))))#N,K,K
                    # mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # # print(mask_zone.sum())
                    
                    # mid_indices, mask12 = self.associate_numpy(sim_mat_12*mask_zone,orb2.asnumpy())#(N,K,1)
                    # max_indices, mask21 = self.associate_numpy(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1.asnumpy())#(N,K,1)
                    # indices = np.diagonal(np.take((max_indices+1)*(mask21*2-1)-1,mid_indices.squeeze(2),axis=1),axis1=0,axis2=1).transpose((2,0,1)).squeeze(2)##N,K
                    # indices_2 = indices*(mask12.squeeze(2))#N,K
                    # # pdb.set_trace()
                    # mask=indices==np.expand_dims(np.arange(orb1.asnumpy().shape[1]),0)##N,K
                    # mask2=indices_2==np.expand_dims(np.arange(orb1.asnumpy().shape[1]),0)##N,K
                    # mask=mask*mask2==1##N,K
                    # print(mask.sum())
                    # mid_orb_warp=np.diagonal(np.take(orb2.asnumpy(),mid_indices.squeeze(2),axis=1),axis1=0,axis2=1).transpose((2,0,1))#(N,K,2)
                    # coor1=orb1*nd.array(np.expand_dims(mask,axis=2),ctx=flow.context)
                    # coor2=nd.array(mid_orb_warp*(np.expand_dims(mask,axis=2)),ctx=flow.context)
                    
                    
                    # ################################CPU+GPU
                    # # pdb.set_trace()
                    # normalized_desc1s = np.transpose(desc1s/np.linalg.norm(desc1s,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    # normalized_desc2s = np.transpose(desc2s/np.linalg.norm(desc2s,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    # sim_mats = np.matmul(normalized_desc1s, normalized_desc2s)#(5,N,K,K)
                    # sim_mat_12=np.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K,K)
                    # sim_mat_21=sim_mat_12.transpose((0,2,1))#(N,K,K)
                    # ####orb1(N,K,2)    orb_warp(N,K,2)    
                    # dis=np.absolute(np.sum(orb1.asnumpy()*orb1.asnumpy(),axis=2,keepdims=True)+np.transpose(np.sum(orb2.asnumpy()*orb2.asnumpy(),axis=2,keepdims=True),(0,2,1))-2*np.matmul(orb1.asnumpy(),np.transpose(orb2.asnumpy(),(0,2,1))))#N,K,K
                    # mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # print(mask_zone.sum())
                    # mid_indices, mask12 = self.associate(nd.array(sim_mat_12*mask_zone,ctx=flow.context),orb2)#(N,K,1)
                    # max_indices, mask21 = self.associate(nd.array(sim_mat_21*(mask_zone.transpose((0,2,1))),ctx=flow.context),orb1)#(N,K,1)
                    # indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K
                    # indices_2 = indices*mask12.squeeze(2)#N,K
                    # mask=nd.broadcast_equal(indices,nd.expand_dims(nd.array(np.arange(orb1.asnumpy().shape[1]),ctx=flow.context),0))##N,K
                    # mask2=nd.broadcast_equal(indices_2,nd.expand_dims(nd.array(np.arange(orb1.asnumpy().shape[1]),ctx=flow.context),0))##N,K
                    # mask=mask*mask2==1##N,K
                    # mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K,2)
                    # coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    # coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    
                    time3=time.time()
                    print(time3-time2)
                    
                    # savepath1='/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_0.9_0.98/'
                    # savepath2='/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_one_shot_0.9_0.98/'
                    # savepath3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.9_0.98/'
                    # if not os.path.exists(savepath1):
                        # os.mkdir(savepath1)
                    # if not os.path.exists(savepath2):
                        # os.mkdir(savepath2)
                    # if not os.path.exists(savepath3):
                        # os.mkdir(savepath3)
                    # for k in range (shape[0]):
                        # kp1=[]
                        # kp2=[]
                        # # im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),warp[k, 0, :, :].asnumpy())
                        # im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                        # plt.figure()
                        # plt.imshow(im1)
                        # for i in range (orb1.shape[1]):
                            # if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                # plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                        # plt.savefig(savepath1+str(steps)+'_pairs.jpg', dpi=600)
                        # plt.close()
                        # for i in range (orb1.shape[1]):
                            # if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                # kp1.append(coor1[k,i,:].asnumpy().tolist())
                                # kp2.append(coor2[k,i,:].asnumpy().tolist())
                                # plt.figure()
                                # plt.imshow(im1)
                                # plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                                # plt.savefig(savepath2+str(steps)+'_'+str(i)+'_'+str(sim_mat_12[k,i,mid_indices[k,i,0]])+'_pairs.jpg', dpi=600)
                                # plt.close()
                        # if len(kp1)!=0:
                            # name = ['X', 'Y']
                            # kp1=np.asarray(kp1)
                            # kp1=kp1[:,[1,0]]#.transpose(0,1)
                            # kp2=np.asarray(kp2)#.transpose(0,1)
                            # kp2=kp2[:,[1,0]]
                            # outlmk1 = pd.DataFrame(columns=name, data=kp1)
                            # outlmk1.to_csv(savepath3+str(steps)+'_1.csv')
                            # outlmk2 = pd.DataFrame(columns=name, data=kp2)
                            # outlmk2.to_csv(savepath3+str(steps)+'_2.csv')
                    
                    
                    if not os.path.exists(savepath1):
                        os.mkdir(savepath1)
                    if not os.path.exists(savepath3):
                        os.mkdir(savepath3)
                    
                    for k in range (shape[0]):
                        kp1=[]
                        kp2=[]
                        # im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),warp[k, 0, :, :].asnumpy())
                        im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                        plt.figure()
                        plt.imshow(im1)
                        count_pair=0
                        for i in range (coor1.shape[1]):
                            if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                count_pair=count_pair+1
                                kp1.append(coor1[k,i,:].asnumpy().tolist())
                                kp2.append(coor2[k,i,:].asnumpy().tolist())
                                plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                        plt.title(str(name_num))
                        plt.savefig(savepath1+str(steps)+'_'+str(count_pair)+'.jpg', dpi=600)
                        plt.close()
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                        if len(kp1)!=0:
                            name = ['X', 'Y']
                            kp1=np.asarray(kp1)
                            kp2=np.asarray(kp2)#.transpose(0,1)
                            outlmk1 = pd.DataFrame(columns=name, data=kp1[:,[1,0]])
                            outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                            outlmk2 = pd.DataFrame(columns=name, data=kp2[:,[1,0]])
                            outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
                            kp1 = np.pad(kp1, ((0, 1000 - len(kp1)), (0, 0)), "constant")
                            kp2 = np.pad(kp2, ((0, 1000 - len(kp2)), (0, 0)), "constant")
                        else:
                            kp1 = np.zeros((1000, 2), dtype=np.int64)
                            kp2 = np.zeros((1000, 2), dtype=np.int64)
                        if k==0:
                            coor1s=np.expand_dims(kp1,axis=0)
                            coor2s=np.expand_dims(kp2,axis=0)
                        else:
                            coor1s=np.concatenate((coor1s,np.expand_dims(kp1,axis=0)),axis=0)
                            coor2s=np.concatenate((coor2s,np.expand_dims(kp2,axis=0)),axis=0)
                    coor1s=nd.array(coor1s,ctx=flow.context)
                    coor2s=nd.array(coor2s,ctx=flow.context)
                    
                    
                    dist_loss2, _, _ = self.landmark_dist(coor1s, coor2s, flows)
                    # dist_loss2 = nd.sum(nd.sqrt(nd.sum(nd.square(coor1 - coor2), axis=-1) + 1e-5), axis=-1)/ (shape[2]*1.414) /(nd.sum(mask,axis=1)+1e-5)*(np.sum(mask, axis=1)!=0)

                    loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                    losses.append(loss)
                    reg_losses.append(reg_loss)
                    raw_losses.append(raw_loss)
                    dist_losses.append(dist_loss*dist_weight)
                    dist_losses2.append(dist_loss2*dist_weight2)
                else:
                    loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                    losses.append(loss)
                    reg_losses.append(reg_loss)
                    raw_losses.append(raw_loss)
                    dist_losses.append(dist_loss*dist_weight)
                    dist_losses2.append(dist_loss*0)

        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2]))}
        # return 0
    
    
    
    
    def train_batch_vgg_dense(self, dist_weight, img1, img2,sift1s,sift2s):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, sift1s, sift2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, sift1s, sift2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, sift1, sift2 in zip(img1, img2, sift1s, sift2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow)
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
    
    
    
    
    
    def train_batch_SFG(self, dist_weight, img1, img2,sift1s,sift2s,kp1_sfgs,kp2_sfgs):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, sift1s, sift2s,kp1_sfgs,kp2_sfgs = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, sift1s, sift2s,kp1_sfgs,kp2_sfgs))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, sift1, sift2,kp1_sfg,kp2_sfg in zip(img1, img2, sift1s, sift2s,kp1_sfgs,kp2_sfgs):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # pdb.set_trace()
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow)
                dist_loss2, _, _ = self.landmark_dist(kp1_sfg, kp2_sfg, flows)
                dist_weight2=30
                self.raw_weight=1
                self.reg_weight = 1#1/20 #0.2
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
    
    
    
    
    
    def train_batch_S_SFG(self, dist_weight, img1, img2,kp1_sfgs,kp2_sfgs,name_num):
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
                
                # savepath='/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/training_visualization_DFS_SFG1024/'
                # if not os.path.exists(savepath):
                    # os.mkdir(savepath)
                # im1=self.appendimages(img1s[0, 0, :, :].squeeze().asnumpy(),img2s[0, 0, :, :].squeeze().asnumpy())
                # plt.figure()
                # plt.imshow(im1)
                # plt.plot([kp1_sfg[0,:,1].asnumpy(),kp2_sfg[0,:,1].asnumpy()+shape[2]],[kp1_sfg[0,:,0].asnumpy(),kp2_sfg[0,:,0].asnumpy()], '#FF0033',linewidth=0.5)
                # plt.savefig(savepath+str(name_num)+'_ite1.jpg', dpi=600)
                # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
                
                
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
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # pdb.set_trace()
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow)
                
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
    
    
    def train_batch_supervised_SFG(self, dist_weight, img1, img2,sift1s,sift2s, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
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
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # pdb.set_trace()
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
    
    
    
    
    
    
    
    
    def train_batch_supervised_LFS_SFG(self, dist_weight, img1, img2, kp1_gts,kp2_gts,kp1_sfgs,kp2_sfgs):
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
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # pdb.set_trace()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_batch_convergence(self, dist_weight, img1, img2, lmk1s, lmk2s,name_nums,steps):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2 in zip(img1, img2, lmk1s, lmk2s):
                # print(lmk1.shape)
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # warp = self.reconstruction(sift2s, flow)
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                # raw loss calculation
                # raw_loss = self.raw_loss_op(sift1s, warp)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                savepath3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.985_0.975_with_network_update/'
                for i in range (len(name_nums)):#name_num not in [206,207,374,375,376,377] and (steps-1)%100==0:
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp1 = lmk_temp[:, [2, 1]]
                        lmk_temp1 = np.pad(lmk_temp1, ((0, 1000 - len(lmk_temp1)), (0, 0)), "constant")
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp2= lmk_temp[:, [2, 1]]
                        lmk_temp2 = np.pad(lmk_temp2, ((0, 1000 - len(lmk_temp2)), (0, 0)), "constant")
                    except:
                        lmk_temp1 = np.zeros((1000, 2), dtype=np.int64)
                        lmk_temp2 = np.zeros((1000, 2), dtype=np.int64)
                    lmk_temp1=nd.expand_dims(nd.array(lmk_temp1,ctx=flow.context),axis=0)
                    lmk_temp2=nd.expand_dims(nd.array(lmk_temp2,ctx=flow.context),axis=0)
                    if i==0:
                        lmk_temp1s=lmk_temp1
                        lmk_temp2s=lmk_temp2
                    else:
                        lmk_temp1s=nd.concat(lmk_temp1s,lmk_temp1,dim=0)
                        lmk_temp2s=nd.concat(lmk_temp2s,lmk_temp2,dim=0)
                # print(lmk_temp1s.shape)
                dist_loss2, _, _ = self.landmark_dist(lmk_temp1s, lmk_temp2s, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_batch_convergence_with_mask(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_mask, lmk2s_mask,name_nums,steps):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_mask, lmk2s_mask = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_mask, lmk2s_mask))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_mask, lmk2_mask in zip(img1, img2, lmk1s, lmk2s, lmk1s_mask, lmk2s_mask):
                # print(lmk1.shape)
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # warp = self.reconstruction(sift2s, flow)
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                # raw loss calculation
                # raw_loss = self.raw_loss_op(sift1s, warp)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=10
                dist_weight3=10
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask, lmk2_mask, flows)
                savepath3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.985_0.975_with_network_update/'
                for i in range (len(name_nums)):#name_num not in [206,207,374,375,376,377] and (steps-1)%100==0:
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp1 = lmk_temp[:, [2, 1]]
                        lmk_temp1 = np.pad(lmk_temp1, ((0, 1000 - len(lmk_temp1)), (0, 0)), "constant")
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp2= lmk_temp[:, [2, 1]]
                        lmk_temp2 = np.pad(lmk_temp2, ((0, 1000 - len(lmk_temp2)), (0, 0)), "constant")
                    except:
                        lmk_temp1 = np.zeros((1000, 2), dtype=np.int64)
                        lmk_temp2 = np.zeros((1000, 2), dtype=np.int64)
                    lmk_temp1=nd.expand_dims(nd.array(lmk_temp1,ctx=flow.context),axis=0)
                    lmk_temp2=nd.expand_dims(nd.array(lmk_temp2,ctx=flow.context),axis=0)
                    if i==0:
                        lmk_temp1s=lmk_temp1
                        lmk_temp2s=lmk_temp2
                    else:
                        lmk_temp1s=nd.concat(lmk_temp1s,lmk_temp1,dim=0)
                        lmk_temp2s=nd.concat(lmk_temp2s,lmk_temp2,dim=0)
                # print(lmk_temp1s.shape)
                dist_loss3, _, _ = self.landmark_dist(lmk_temp1s, lmk_temp2s, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
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
    
    def train_batch_DLFS_SFG_multiscale(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2 in zip(img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1.3
                self.reg_weight = 1 #0.2
                dist_weight = 30#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=15
                dist_weight3=15
                dist_weight4=30
                dist_weight5=15
                dist_weight6=15
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6]))}
    
    
    def train_batch_supervised_sparse_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, lmk1_gt, lmk2_gt in zip(img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=5
                dist_weight3=5
                dist_weight4=20
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_gt, lmk2_gt, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    
    
    
    
    def train_batch_supervised_sparse_vgg_vgglarge(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_gt, lmk2s_gt):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_gt, lmk2s_gt = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_gt, lmk2s_gt))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_gt, lmk2_gt in zip(img1, img2, lmk1s, lmk2s, lmk1s_gt, lmk2s_gt):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight4=0
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_gt, lmk2_gt, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    
    def train_batch_sparse_vgg_vgglarge(self, dist_weight, img1, img2, lmk1s, lmk2s):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2 in zip(img1, img2, lmk1s, lmk2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1#1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))}
    
    
    
    
    def train_batch_sparse_vgg_vgglarge_multiscale(self, dist_weight, img1, img2,kp1s_same,kp2s_same,kp1s_5_2,kp2s_5_2,kp1s_2_5,kp2s_2_5,kp1s_large,kp2s_large):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        batch_size = img1.shape[0]
        img1, img2,kp1s_same,kp2s_same,kp1s_5_2,kp2s_5_2,kp1s_2_5,kp2s_2_5,kp1s_large,kp2s_large = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,kp1s_same,kp2s_same,kp1s_5_2,kp2s_5_2,kp1s_2_5,kp2s_2_5,kp1s_large,kp2s_large))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,kp1_same,kp2_same,kp1_5_2,kp2_5_2,kp1_2_5,kp2_2_5,kp1_large,kp2_large in zip(img1, img2,kp1s_same,kp2s_same,kp1s_5_2,kp2s_5_2,kp1s_2_5,kp2s_2_5,kp1s_large,kp2s_large):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1#1/20 #0.2
                dist_weight = 40#200#0#50 # 10#1 #50 #100 #200
                dist_weight2 = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight3 = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight4 = 40#200#0#50 # 10#1 #50 #100 #200
                dist_loss, _, _ = self.landmark_dist(kp1_same, kp2_same, flows)
                dist_loss2, _, _ = self.landmark_dist(kp1_5_2, kp2_5_2, flows)
                dist_loss3, _, _ = self.landmark_dist(kp1_2_5, kp2_2_5, flows)
                dist_loss4, _, _ = self.landmark_dist(kp1_large, kp2_large, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight+ dist_loss2*dist_weight2+ dist_loss3*dist_weight3+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    

    
    def train_batch_sparse_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        # if 1:
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg in zip(img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                # pdb.set_trace()
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20#5
                dist_weight3=20#10
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
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
    
    
    def train_batch_sparse_vgg_vgglarge_maskmorethanvgg_masksamewithvgg_with_corrected_LFS(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,lmk1s_corrected,lmk2s_corrected):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,lmk1s_corrected,lmk2s_corrected = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,lmk1s_corrected,lmk2s_corrected))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg,lmk1_corrected,lmk2_corrected in zip(img1, img2, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,lmk1s_corrected,lmk2s_corrected):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 15#20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=5
                dist_weight3=10
                dist_weight4=10
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_corrected, lmk2_corrected, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    def train_batch_supervised_densewithsparse_multisift_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4 = []
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, lmk1_gt, lmk2_gt in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg, lmk1s_gt, lmk2s_gt):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20
                dist_weight3=20
                dist_weight4=20
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_gt, lmk2_gt, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    
    
    
    
    
    
    
    
    def train_batch_DLFSFG_multiscale(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                raw_loss_img = self.raw_loss_op(img1s, warp_img)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.raw_img_weight=0
                self.reg_weight = 1 #0.2
                dist_weight = 10#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=5
                dist_weight3=5
                dist_weight4=10
                dist_weight5=3
                dist_weight6=3
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                loss = self.raw_img_weight*raw_loss_img+raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                raw_losses_img.append(raw_loss_img*self.raw_img_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses])), "raw loss img": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses_img]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6]))}
    
    def train_batch_DLFSFG_multiscale5(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 15#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=15
                dist_weight3=8
                dist_weight4=8
                dist_weight5=8
                dist_weight6=8
                dist_weight7=8
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))}
    
    def train_batch_recursive_DLFSFG_multiscale_ite1(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        dist_losses8=[]
        dist_losses9=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512,lmk1_large_1024,lmk2_large_1024 ,lmk1_ite1,lmk2_ite1 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 15#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=15
                dist_weight3=2
                dist_weight4=2
                dist_weight5=2
                dist_weight6=2
                dist_weight7=8
                dist_weight9=8
                dist_weight8=15
                
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                dist_loss8, _, _ = self.landmark_dist(lmk1_ite1,lmk2_ite1, flows)
                dist_loss9, _, _ = self.landmark_dist(lmk1_large_1024,lmk2_large_1024, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7+dist_loss8*dist_weight8+dist_loss9*dist_weight9## gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
                dist_losses8.append(dist_loss8*dist_weight8)
                dist_losses9.append(dist_loss9*dist_weight9)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))
        , "dist loss8": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses8])), "dist loss9": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses9]))}
    def train_batch_recursive_DLFSFG_multiscale_ite2(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        dist_losses8=[]
        dist_losses9=[]
        dist_losses10=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512,lmk1_large_1024,lmk2_large_1024 ,lmk1_ite1,lmk2_ite1,lmk1_ite2,lmk2_ite2 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 25#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=25
                dist_weight3=3
                dist_weight4=3
                dist_weight5=3
                dist_weight6=3
                dist_weight7=8
                dist_weight9=8
                dist_weight8=25
                dist_weight10=25
                
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                dist_loss8, _, _ = self.landmark_dist(lmk1_ite1,lmk2_ite1, flows)
                dist_loss9, _, _ = self.landmark_dist(lmk1_large_1024,lmk2_large_1024, flows)
                dist_loss10, _, _ = self.landmark_dist(lmk1_ite2,lmk2_ite2, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7+dist_loss8*dist_weight8+dist_loss9*dist_weight9+dist_loss10*dist_weight10## gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
                dist_losses8.append(dist_loss8*dist_weight8)
                dist_losses9.append(dist_loss9*dist_weight9)
                dist_losses10.append(dist_loss10*dist_weight10)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))
        , "dist loss8": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses8])), "dist loss9": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses9]))
        , "dist loss10": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses10]))}
    def train_batch_recursive_DLFSFG_multiscale_ite3(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        dist_losses8=[]
        dist_losses9=[]
        dist_losses10=[]
        dist_losses11=[]
        dist_losses12=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512,lmk1_large_1024,lmk2_large_1024 ,lmk1_ite1,lmk2_ite1,lmk1_ite2,lmk2_ite2,lmk1_ite3,lmk2_ite3,lmk1_ite4,lmk2_ite4 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20
                dist_weight3=3
                dist_weight4=3
                dist_weight5=3
                dist_weight6=3
                dist_weight7=8
                dist_weight9=8
                dist_weight8=20
                dist_weight10=20
                dist_weight11=20
                dist_weight12=20
                
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                dist_loss8, _, _ = self.landmark_dist(lmk1_ite1,lmk2_ite1, flows)
                dist_loss9, _, _ = self.landmark_dist(lmk1_large_1024,lmk2_large_1024, flows)
                dist_loss10, _, _ = self.landmark_dist(lmk1_ite2,lmk2_ite2, flows)
                dist_loss11, _, _ = self.landmark_dist(lmk1_ite3,lmk2_ite3, flows)
                dist_loss12, _, _ = self.landmark_dist(lmk1_ite4,lmk2_ite4, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7+dist_loss8*dist_weight8+dist_loss9*dist_weight9+dist_loss10*dist_weight10+dist_loss11*dist_weight11+dist_loss12*dist_weight12## gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
                dist_losses8.append(dist_loss8*dist_weight8)
                dist_losses9.append(dist_loss9*dist_weight9)
                dist_losses10.append(dist_loss10*dist_weight10)
                dist_losses11.append(dist_loss11*dist_weight11)
                dist_losses12.append(dist_loss12*dist_weight12)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        # return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        # , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))
        , "dist loss8": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses8])), "dist loss9": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses9]))
        , "dist loss10": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses10])), "dist loss11": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses11]))
        , "dist loss12": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses12]))}
        
    def train_batch_recursive_DLFS_SFG_multiscale_ite3(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        dist_losses8=[]
        dist_losses9=[]
        dist_losses10=[]
        dist_losses11=[]
        dist_losses12=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512,lmk1_large_1024,lmk2_large_1024 ,lmk1_ite1,lmk2_ite1,lmk1_ite2,lmk2_ite2,lmk1_ite3,lmk2_ite3,lmk1_ite4,lmk2_ite4 in zip(img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 0.05 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20
                dist_weight3=3
                dist_weight4=3
                dist_weight5=3
                dist_weight6=3
                dist_weight7=8
                dist_weight9=8
                dist_weight8=20
                dist_weight10=20
                dist_weight11=20
                dist_weight12=20
                
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                dist_loss8, _, _ = self.landmark_dist(lmk1_ite1,lmk2_ite1, flows)
                dist_loss9, _, _ = self.landmark_dist(lmk1_large_1024,lmk2_large_1024, flows)
                dist_loss10, _, _ = self.landmark_dist(lmk1_ite2,lmk2_ite2, flows)
                dist_loss11, _, _ = self.landmark_dist(lmk1_ite3,lmk2_ite3, flows)
                dist_loss12, _, _ = self.landmark_dist(lmk1_ite4,lmk2_ite4, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7+dist_loss8*dist_weight8+dist_loss9*dist_weight9+dist_loss10*dist_weight10+dist_loss11*dist_weight11+dist_loss12*dist_weight12## gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
                dist_losses8.append(dist_loss8*dist_weight8)
                dist_losses9.append(dist_loss9*dist_weight9)
                dist_losses10.append(dist_loss10*dist_weight10)
                dist_losses11.append(dist_loss11*dist_weight11)
                dist_losses12.append(dist_loss12*dist_weight12)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))
        , "dist loss8": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses8])), "dist loss9": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses9]))
        , "dist loss10": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses10])), "dist loss11": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses11]))
        , "dist loss12": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses12]))}
    def train_batch_recursive_DLFS_SFG_multiscale_ite3_with_gt(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4,lmk1s_gt,lmk2s_gt):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        dist_losses7=[]
        dist_losses8=[]
        dist_losses9=[]
        dist_losses10=[]
        dist_losses11=[]
        dist_losses12=[]
        dist_losses13=[]
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4,lmk1s_gt,lmk2s_gt = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4,lmk1s_gt,lmk2s_gt))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_old2, lmk2_old2, lmk1_old5, lmk2_old5, lmk1_large, lmk2_large,lmk1_LFS1,lmk2_LFS1,lmk1_LFS2,lmk2_LFS2,lmk1_512,lmk2_512,lmk1_large_1024,lmk2_large_1024 ,lmk1_ite1,lmk2_ite1,lmk1_ite2,lmk2_ite2,lmk1_ite3,lmk2_ite3,lmk1_ite4,lmk2_ite4,lmk1_gt,lmk2_gt in zip(img1, img2, lmk1s, lmk2s, lmk1s_old2, lmk2s_old2, lmk1s_old5, lmk2s_old5, lmk1s_large, lmk2s_large,lmk1s_LFS1,lmk2s_LFS1,lmk1s_LFS2,lmk2s_LFS2,lmk1s_512,lmk2s_512,lmk1s_large_1024,lmk2s_large_1024,lmk1s_ite1,lmk2s_ite1,lmk1s_ite2,lmk2s_ite2,lmk1s_ite3,lmk2s_ite3,lmk1s_ite4,lmk2s_ite4,lmk1s_gt,lmk2s_gt):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=20
                dist_weight3=3
                dist_weight4=3
                dist_weight5=3
                dist_weight6=3
                dist_weight7=8
                dist_weight9=8
                dist_weight8=20
                dist_weight10=20
                dist_weight11=20
                dist_weight12=20
                dist_weight13=50
                
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_old2, lmk2_old2, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_old5, lmk2_old5, flows)
                dist_loss4, _, _ = self.landmark_dist(lmk1_large, lmk2_large, flows)
                dist_loss5, _, _ = self.landmark_dist(lmk1_LFS1, lmk2_LFS1, flows)
                dist_loss6, _, _ = self.landmark_dist(lmk1_LFS2, lmk2_LFS2, flows)
                dist_loss7, _, _ = self.landmark_dist(lmk1_512,lmk2_512, flows)
                dist_loss8, _, _ = self.landmark_dist(lmk1_ite1,lmk2_ite1, flows)
                dist_loss9, _, _ = self.landmark_dist(lmk1_large_1024,lmk2_large_1024, flows)
                dist_loss10, _, _ = self.landmark_dist(lmk1_ite2,lmk2_ite2, flows)
                dist_loss11, _, _ = self.landmark_dist(lmk1_ite3,lmk2_ite3, flows)
                dist_loss12, _, _ = self.landmark_dist(lmk1_ite4,lmk2_ite4, flows)
                dist_loss13, _, _ = self.landmark_dist(lmk1_gt,lmk2_gt, flows)
                loss =raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4+dist_loss5*dist_weight5+dist_loss6*dist_weight6+dist_loss7*dist_weight7+dist_loss8*dist_weight8+dist_loss9*dist_weight9+dist_loss10*dist_weight10+dist_loss11*dist_weight11+dist_loss12*dist_weight12+dist_loss13*dist_weight13## gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
                dist_losses5.append(dist_loss5*dist_weight5)
                dist_losses6.append(dist_loss6*dist_weight6)
                dist_losses7.append(dist_loss7*dist_weight7)
                dist_losses8.append(dist_loss8*dist_weight8)
                dist_losses9.append(dist_loss9*dist_weight9)
                dist_losses10.append(dist_loss10*dist_weight10)
                dist_losses11.append(dist_loss11*dist_weight11)
                dist_losses12.append(dist_loss12*dist_weight12)
                dist_losses13.append(dist_loss13*dist_weight13)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        # if np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))*60/dist_weight7>0.2:
            # pdb.set_trace()
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4])), "dist loss5": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses5]))
        , "dist loss6": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses6])), "dist loss7": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses7]))
        , "dist loss8": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses8])), "dist loss9": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses9]))
        , "dist loss10": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses10])), "dist loss11": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses11]))
        , "dist loss12": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses12])), "dist loss13": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses13]))}
    def train_batch_DLFSFG_multiscale2(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s= map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                raw_loss_img = self.raw_loss_op(img1s, warp_img)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.raw_img_weight=0
                self.reg_weight = 1 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                loss = self.raw_img_weight*raw_loss_img+raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                raw_losses_img.append(raw_loss_img*self.raw_img_weight)
                dist_losses.append(dist_loss*dist_weight)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses])), "raw loss img": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses_img]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))}
    
    
    def train_batch_DLFSFG_multiscale4(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_2, lmk2s_2):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_2, lmk2s_2= map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_2, lmk2s_2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_2, lmk2_2 in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_2, lmk2s_2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                # raw_loss_img = self.raw_loss_op(img1s, warp_img)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.raw_img_weight=0
                self.reg_weight = 1 #0.2
                dist_weight = 15#200#0#50 # 10#1 #50 #100 #200
                dist_weight2 = 30#200#0#50 # 10#1 #50 #100 #200
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_2, lmk2_2, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight+ dist_loss2*dist_weight2 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2]))}
    
    
    
    def train_batch_DLFSFG_multiscale3(self, dist_weight, img1, img2,sift1s,sift2s):
        losses = []
        reg_losses = []
        raw_losses = []
        raw_losses_img = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4=[]
        dist_losses5=[]
        dist_losses6=[]
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s= map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2 in zip(img1, img2,sift1s,sift2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                warp_img = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                raw_loss_img = self.raw_loss_op(img1s, warp_img)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.raw_img_weight=0
                self.reg_weight = 1 #0.2
                # dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                loss = self.raw_img_weight*raw_loss_img+raw_loss * self.raw_weight + reg_loss * self.reg_weight # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss* self.raw_weight)
                raw_losses_img.append(raw_loss_img*self.raw_img_weight)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses])), "raw loss img": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses_img]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
    
    
    
    
    
    
    def train_batch_densewithsparse_multisift_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,steps):
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
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                # pdb.set_trace()
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                dist_weight = 30#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=30
                dist_weight3=0
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                
                # im1=self.appendimages(img1s[0, 0, :, :].asnumpy(),img2s[0, 0, :, :].asnumpy())
                # plt.figure()
                # plt.imshow(im1)
                # plt.scatter(lmk1[0,:,1].asnumpy(),lmk1[0,:,0].asnumpy(), color='red', alpha=0.2)
                # plt.scatter(lmk2[0,:,1].asnumpy()+img1s.shape[3],lmk2[0,:,0].asnumpy(), color='red', alpha=0.2)
                # # plt.plot([warped_lmk[0,:,1].asnumpy()[0],lmk2[0,:,1].asnumpy()[0]+512],[warped_lmk[0,:,0].asnumpy()[0],lmk2[0,:,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                # plt.savefig('/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/training_visualization0308/'+str(steps)+'.jpg', dpi=600)
                # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
                
                
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))}
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_batch_recursive_densewithsparse_multisift_vgg_vgglarge_maskmorethanvgg_masksamewithvgg(self, dist_weight, img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg,name_nums):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4 = []
        batch_size = img1.shape[0]
        img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s,sift1,sift2, lmk1, lmk2, lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg in zip(img1, img2,sift1s,sift2s, lmk1s, lmk2s, lmk1s_mask_same_with_vgg, lmk2s_mask_same_with_vgg, lmk1s_mask_more_than_vgg, lmk2s_mask_more_than_vgg):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(sift2, flow)
                flows = []
                flows.append(flow)
                raw_loss = self.raw_loss_op(sift1, warp)
                reg_loss = self.regularization_op(flow) #+ self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=1#5
                dist_weight3=1#10
                dist_weight4=20
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask_same_with_vgg, lmk2_mask_same_with_vgg, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_more_than_vgg, lmk2_mask_more_than_vgg, flows)
                
                
                savepath3="/data/wxy/association/Maskflownet_association/kps/d0fNov09_5074_img2s_key_points_rotate16_0.975_0.98_with_network_update/"
                for i in range (len(name_nums)):#name_num not in [206,207,374,375,376,377] and (steps-1)%100==0:
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp1 = lmk_temp[:, [2, 1]]
                        lmk_temp1 = np.pad(lmk_temp1, ((0, 1000 - len(lmk_temp1)), (0, 0)), "constant")
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp2= lmk_temp[:, [2, 1]]
                        lmk_temp2 = np.pad(lmk_temp2, ((0, 1000 - len(lmk_temp2)), (0, 0)), "constant")
                    except:
                        lmk_temp1 = np.zeros((1000, 2), dtype=np.int64)
                        lmk_temp2 = np.zeros((1000, 2), dtype=np.int64)
                    lmk_temp1=nd.expand_dims(nd.array(lmk_temp1,ctx=flow.context),axis=0)
                    lmk_temp2=nd.expand_dims(nd.array(lmk_temp2,ctx=flow.context),axis=0)
                    if i==0:
                        lmk_temp1s=lmk_temp1
                        lmk_temp2s=lmk_temp2
                    else:
                        lmk_temp1s=nd.concat(lmk_temp1s,lmk_temp1,dim=0)
                        lmk_temp2s=nd.concat(lmk_temp2s,lmk_temp2,dim=0)
                dist_loss4, _, _ = self.landmark_dist(lmk_temp1s, lmk_temp2s, flows)
                
                
                
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4# gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    
    
    
    
    def train_batch_convergence_with_mask_with_gt(self, dist_weight, img1, img2, lmk1s, lmk2s, lmk1s_mask_gt, lmk2s_mask_gt, lmk1s_mask, lmk2s_mask,name_nums,steps):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4 = []
        batch_size = img1.shape[0]
        img1, img2, lmk1s, lmk2s, lmk1s_mask_gt, lmk2s_mask_gt, lmk1s_mask, lmk2s_mask = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, lmk1s, lmk2s, lmk1s_mask_gt, lmk2s_mask_gt, lmk1s_mask, lmk2s_mask))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, lmk1, lmk2, lmk1_mask_gt, lmk2_mask_gt, lmk1_mask, lmk2_mask in zip(img1, img2, lmk1s, lmk2s, lmk1s_mask_gt, lmk2s_mask_gt, lmk1s_mask, lmk2s_mask):
                # print(lmk1.shape)
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                # pdb.set_trace()
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # warp = self.reconstruction(sift2s, flow)
                warp = self.reconstruction(img2s, flow)
                flows = []
                flows.append(flow)
                # raw loss calculation
                # raw_loss = self.raw_loss_op(sift1s, warp)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=10
                dist_weight3=20
                dist_weight4=20
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(lmk1_mask, lmk2_mask, flows)
                dist_loss3, _, _ = self.landmark_dist(lmk1_mask_gt, lmk2_mask_gt, flows)
                savepath3='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.975_0.975_with_network_update/'
                for i in range (len(name_nums)):#name_num not in [206,207,374,375,376,377] and (steps-1)%100==0:
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp1 = lmk_temp[:, [2, 1]]
                        lmk_temp1 = np.pad(lmk_temp1, ((0, 1000 - len(lmk_temp1)), (0, 0)), "constant")
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_nums[i])+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp2= lmk_temp[:, [2, 1]]
                        lmk_temp2 = np.pad(lmk_temp2, ((0, 1000 - len(lmk_temp2)), (0, 0)), "constant")
                    except:
                        lmk_temp1 = np.zeros((1000, 2), dtype=np.int64)
                        lmk_temp2 = np.zeros((1000, 2), dtype=np.int64)
                    lmk_temp1=nd.expand_dims(nd.array(lmk_temp1,ctx=flow.context),axis=0)
                    lmk_temp2=nd.expand_dims(nd.array(lmk_temp2,ctx=flow.context),axis=0)
                    if i==0:
                        lmk_temp1s=lmk_temp1
                        lmk_temp2s=lmk_temp2
                    else:
                        lmk_temp1s=nd.concat(lmk_temp1s,lmk_temp1,dim=0)
                        lmk_temp2s=nd.concat(lmk_temp2s,lmk_temp2,dim=0)
                # print(lmk_temp1s.shape)
                dist_loss4, _, _ = self.landmark_dist(lmk_temp1s, lmk_temp2s, flows)
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2+dist_loss3*dist_weight3+dist_loss4*dist_weight4 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                dist_losses.append(dist_loss*dist_weight)
                dist_losses2.append(dist_loss2*dist_weight2)
                dist_losses3.append(dist_loss3*dist_weight3)
                dist_losses4.append(dist_loss4*dist_weight4)

        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "dist loss": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses]))
        , "dist loss2": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses2])), "dist loss3": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses3]))
        , "dist loss4": np.mean(np.concatenate([loss.asnumpy() for loss in dist_losses4]))}
    
    
    
    
    
    def kp_pairs(self, dist_weight, img1, img2,orb1s,orb2s,name_num,steps):
        img1, img2,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        if 1:
        # with autograd.record():
            for img1s, img2s,orb1,orb2 in zip(img1, img2,orb1s,orb2s):
                if 1:
                    shape = img1s.shape
                    savepath1="/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_rotate16_0.98_0.95_corrected/"
                    savepath3="/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_rotate16_0.98_0.95_corrected/"
                    if not os.path.exists(savepath1):
                        os.mkdir(savepath1)
                    if not os.path.exists(savepath3):
                        os.mkdir(savepath3)
                    orb2_list=orb2.squeeze().asnumpy().tolist()
                    orb1_list=orb1.squeeze().asnumpy().tolist()
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp1=lmk_temp.tolist()
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp2=lmk_temp.tolist()
                    except:
                        pass
                    else:
                        for i in range(len(lmk_temp1)):
                            if lmk_temp1[i] in orb1_list:
                                orb1_list.remove(lmk_temp1[i])
                            if lmk_temp2[i] in orb2_list:
                                orb2_list.remove(lmk_temp2[i])
                        orb2=nd.expand_dims(nd.array(np.asarray(orb2_list),ctx=img1s.context),axis=0)
                        orb1=nd.expand_dims(nd.array(np.asarray(orb1_list),ctx=img1s.context),axis=0)
                    print('features extracting:{}'.format(name_num))
                    if orb2.shape[1]>8000:
                        if random.random()>0.5:
                            orb2=orb2[:,:int(orb2.shape[1]/2),:]
                        else:
                            orb2=orb2[:,int(orb2.shape[1]/2):,:]
                    if orb1.shape[1]>8000:
                        if random.random()>0.5:
                            orb1=orb1[:,:int(orb1.shape[1]/2),:]
                        else:
                            orb1=orb1[:,int(orb1.shape[1]/2):,:]
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=np.zeros([5,1568*2,shape[0],orb1.shape[1]])
                    desc2s=np.zeros([5,1568*2,shape[0],orb2.shape[1]])
                    for k in range (shape[0]):
                        for i in range(int(np.ceil(orb1.shape[1]/2))):
                            kp1_x1,kp1_y1=orb1[k,2*i,1].asnumpy(),orb1[k,2*i,0].asnumpy()
                            patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb1[k,2*i+1,1].asnumpy(),orb1[k,2*i+1,0].asnumpy()
                                patch_img1s_2=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            except:
                                patch_img1s_2=patch_img1s
                            patch_img1s, patch_img1s_2, _ = self.centralize(patch_img1s, patch_img1s_2)
                            _, c1s, c2s,_= self.network(patch_img1s, patch_img1s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s,45), mx.image.imrotate(patch_img1s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s,90), mx.image.imrotate(patch_img1s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s,135), mx.image.imrotate(patch_img1s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s,180), mx.image.imrotate(patch_img1s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s,225), mx.image.imrotate(patch_img1s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s,270), mx.image.imrotate(patch_img1s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s,315), mx.image.imrotate(patch_img1s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img1s,22.5), mx.image.imrotate(patch_img1s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img1s,67.5), mx.image.imrotate(patch_img1s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img1s,112.5), mx.image.imrotate(patch_img1s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img1s,157.5), mx.image.imrotate(patch_img1s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img1s,202.5), mx.image.imrotate(patch_img1s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img1s,247.5), mx.image.imrotate(patch_img1s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img1s,292.5), mx.image.imrotate(patch_img1s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img1s,337.5), mx.image.imrotate(patch_img1s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc1s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc1s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
                            except:
                                pass
                        for i in range(int(np.ceil(orb2.shape[1]/2))):
                            kp1_x1,kp1_y1=orb2[k,2*i,1].asnumpy(),orb2[k,2*i,0].asnumpy()
                            patch_img2s=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            try:
                                kp1_x1,kp1_y1=orb2[k,2*i+1,1].asnumpy(),orb2[k,2*i+1,0].asnumpy()
                                patch_img2s_2=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                    nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                            except:
                                patch_img2s_2=patch_img2s

                            patch_img2s, patch_img2s_2, _ = self.centralize(patch_img2s, patch_img2s_2)
                            _, c1s, c2s,_= self.network(patch_img2s, patch_img2s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img2s,45), mx.image.imrotate(patch_img2s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img2s,90), mx.image.imrotate(patch_img2s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img2s,135), mx.image.imrotate(patch_img2s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img2s,180), mx.image.imrotate(patch_img2s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img2s,225), mx.image.imrotate(patch_img2s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img2s,270), mx.image.imrotate(patch_img2s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img2s,315), mx.image.imrotate(patch_img2s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img2s,22.5), mx.image.imrotate(patch_img2s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img2s,67.5), mx.image.imrotate(patch_img2s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img2s,112.5), mx.image.imrotate(patch_img2s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img2s,157.5), mx.image.imrotate(patch_img2s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img2s,202.5), mx.image.imrotate(patch_img2s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img2s,247.5), mx.image.imrotate(patch_img2s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img2s,292.5), mx.image.imrotate(patch_img2s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img2s,337.5), mx.image.imrotate(patch_img2s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc2s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc2s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
                            except:
                                pass
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    del patch_img1s,patch_img1s_2,patch_img2s,patch_img2s_2,c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16
                    del c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16
                    ################################GPU
                    desc1s=nd.array(desc1s,ctx=img1s.context)#(5,196*8,N,K1)
                    desc2s=nd.array(desc2s,ctx=img1s.context)#(5,196*8,N,K2)
                    normalized_desc1s = nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    del desc1s,desc2s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(5,N,K1,K2)
                    del normalized_desc1s,normalized_desc2s
                    sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K1,K2)
                    del sim_mats
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    mid_indices, mask12 = self.associate_single(sim_mat_12*mask_zone,orb2)#(N,K1,1)
                    max_indices,mask21 = self.associate_single(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K2,1)
                    indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K1
                    indices_2 = indices*mask12.squeeze(2)#N,K
                    mask=nd.broadcast_equal(indices,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0))##N,K1
                    mask2=nd.broadcast_equal(indices_2,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0))##N,K1
                    mask=mask*mask2==1##N,K
                    print(mask.sum())
                    mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K1,2)
                    coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    
                    time3=time.time()
                    print(time3-time2)
                    
                    
                    
                    for k in range (shape[0]):
                        kp1=[]
                        kp2=[]
                        im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                        plt.figure()
                        plt.imshow(im1)
                        count_pair=0
                        for i in range (coor1.shape[1]):
                            if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                count_pair=count_pair+1
                                kp1.append(coor1[k,i,:].asnumpy().tolist())
                                kp2.append(coor2[k,i,:].asnumpy().tolist())
                                plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                        if count_pair>0:
                            plt.title(str(name_num))
                            plt.savefig(savepath1+str(name_num)+'_'+str(steps)+'_'+str(count_pair)+'.jpg', dpi=600)
                        plt.close()
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                        if len(kp1)!=0:
                            name = ['X', 'Y']
                            kp1=np.asarray(kp1)
                            kp2=np.asarray(kp2)#.transpose(0,1)
                            outlmk1 = pd.DataFrame(columns=name, data=kp1[:,[1,0]])
                            outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                            # outlmk1.to_csv(savepath3+str(name_num+1)+'_2.csv')
                            outlmk2 = pd.DataFrame(columns=name, data=kp2[:,[1,0]])
                            outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
                            # outlmk2.to_csv(savepath3+str(name_num+1)+'_1.csv')
        return 0
    
    
    def kp_pairs_multiscale(self, dist_weight, img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s,num400,name_num):
        img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        if 1:
        # with autograd.record():
            for img1s, img2s,img1s_256, img2s_256,img1s_1024, img2s_1024,orb1,orb2 in zip(img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s):
                if 1:
                    shape = img1s.shape
                    # savepath1="/data/wxy/association/Maskflownet_association_1024/images/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
                    # savepath1="/data/wxy/association/Maskflownet_association_1024/images/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    # savepath1="/data/wxy/association/Maskflownet_association_1024/images/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
                    # savepath1="/data/wxy/association/Maskflownet_association_1024/images/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    # savepath1="/data/wxy/association/Maskflownet_association_1024/images/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    savepath1="/data/wxy/association/Maskflownet_association_1024/images/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"
                    # savepath3="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    # savepath3="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_with_ORB16s8_1024_0.972_0.92_large/"
                    # savepath3="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite2_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    # savepath3="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92/"
                    savepath3="/data/wxy/association/Maskflownet_association_1024/kps/recursive_DLFSFG_ite3_kps_1024_with_ORB16s8_1024_0.972_0.92_2/"
                    # savepath3="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024_512_2_256_1_1024_1/"
                    if not os.path.exists(savepath1):
                        os.mkdir(savepath1)
                    if not os.path.exists(savepath3):
                        os.mkdir(savepath3)
                    orb2_list=orb2.squeeze().asnumpy().tolist()
                    orb1_list=orb1.squeeze().asnumpy().tolist()
                    try:
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp1=lmk_temp.tolist()
                        lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                        lmk_temp = np.array(lmk_temp)
                        lmk_temp = lmk_temp[:, [2, 1]]
                        lmk_temp2=lmk_temp.tolist()
                    except:
                        pass
                    else:
                        for i in range(len(lmk_temp1)):
                            if lmk_temp1[i] in orb1_list:
                                orb1_list.remove(lmk_temp1[i])
                            if lmk_temp2[i] in orb2_list:
                                orb2_list.remove(lmk_temp2[i])
                        orb2=nd.expand_dims(nd.array(np.asarray(orb2_list),ctx=img1s.context),axis=0)
                        orb1=nd.expand_dims(nd.array(np.asarray(orb1_list),ctx=img1s.context),axis=0)
                    print('features extracting:{}'.format(name_num))
                    orb2_ori=orb2.copy()
                    orb1_ori=orb1.copy()
                    max_orb_num=8000
                    if orb2_ori.shape[1]>max_orb_num:
                        orb2=orb2_ori[:,:min(int(orb2_ori.shape[1]/2),max_orb_num),:]
                    if orb1_ori.shape[1]>max_orb_num:
                        orb1=orb1_ori[:,:min(int(orb1_ori.shape[1]/2),max_orb_num),:]
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=np.zeros([15,1568*2,shape[0],orb1.shape[1]])
                    desc2s=np.zeros([15,1568*2,shape[0],orb2.shape[1]])
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
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            except:
                                patch_img1s_2=patch_img1s
                            # pdb.set_trace()
                            # patch_img1s=nd.mean(nd.reshape(patch_img1s,(3,5,-1,64,64)),0)
                            # patch_img1s_2=nd.mean(nd.reshape(patch_img1s_2,(3,5,-1,64,64)),0)
                            # patch_img1s, patch_img1s_2, _ = self.centralize(patch_img1s, patch_img1s_2)
                            patch_img1s, patch_img1s_2, _ = self.centralize_kp_pairs_multiscale(patch_img1s, patch_img1s_2)
                            
                            
                            
                            patch_img1s_all=torch.cat((patch_img1s.unsqueeze(1),transforms.functional.rotate(patch_img1s,90).unsqueeze(1),transforms.functional.rotate(patch_img1s,180).unsqueeze(1),transforms.functional.rotate(patch_img1s,270).unsqueeze(1),
                                transforms.functional.rotate(patch_img1s,45).unsqueeze(1),transforms.functional.rotate(patch_img1s,135).unsqueeze(1),transforms.functional.rotate(patch_img1s,225).unsqueeze(1),transforms.functional.rotate(patch_img1s,315).unsqueeze(1)),1)
                            patch_img1s_256_all=torch.cat((patch_img1s_256.unsqueeze(1),transforms.functional.rotate(patch_img1s_256,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,270).unsqueeze(1),
                                transforms.functional.rotate(patch_img1s_256,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_256,315).unsqueeze(1)),1)
                            patch_img1s_1024_all=torch.cat((patch_img1s_1024.unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,90).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,180).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,270).unsqueeze(1),
                                transforms.functional.rotate(patch_img1s_1024,45).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,135).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,225).unsqueeze(1),transforms.functional.rotate(patch_img1s_1024,315).unsqueeze(1)),1)###(5,8,3,224,224)
                            patch_img1s_all_multi=torch.cat((patch_img1s_all.view(-1,3,224,224),patch_img1s_256_all.view(-1,3,224,224),patch_img1s_1024_all.view(-1,3,224,224)),0)####(120,3,224,224)#######8+8+8+8+8=40 40*3=120
                                                
                            
                            
                            
                            _, c1s, c2s,_= self.network(patch_img1s, patch_img1s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s,45), mx.image.imrotate(patch_img1s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s,90), mx.image.imrotate(patch_img1s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s,135), mx.image.imrotate(patch_img1s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s,180), mx.image.imrotate(patch_img1s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s,225), mx.image.imrotate(patch_img1s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s,270), mx.image.imrotate(patch_img1s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s,315), mx.image.imrotate(patch_img1s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img1s,22.5), mx.image.imrotate(patch_img1s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img1s,67.5), mx.image.imrotate(patch_img1s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img1s,112.5), mx.image.imrotate(patch_img1s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img1s,157.5), mx.image.imrotate(patch_img1s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img1s,202.5), mx.image.imrotate(patch_img1s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img1s,247.5), mx.image.imrotate(patch_img1s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img1s,292.5), mx.image.imrotate(patch_img1s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img1s,337.5), mx.image.imrotate(patch_img1s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc1s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc1s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
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
                            # patch_img2s=nd.mean(nd.reshape(patch_img2s,(3,5,-1,64,64)),0)
                            # patch_img2s_2=nd.mean(nd.reshape(patch_img2s_2,(3,5,-1,64,64)),0)
                            # patch_img2s, patch_img2s_2, _ = self.centralize(patch_img2s, patch_img2s_2)
                            patch_img2s, patch_img2s_2, _ = self.centralize_kp_pairs_multiscale(patch_img2s, patch_img2s_2)
                            _, c1s, c2s,_= self.network(patch_img2s, patch_img2s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img2s,45), mx.image.imrotate(patch_img2s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img2s,90), mx.image.imrotate(patch_img2s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img2s,135), mx.image.imrotate(patch_img2s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img2s,180), mx.image.imrotate(patch_img2s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img2s,225), mx.image.imrotate(patch_img2s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img2s,270), mx.image.imrotate(patch_img2s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img2s,315), mx.image.imrotate(patch_img2s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img2s,22.5), mx.image.imrotate(patch_img2s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img2s,67.5), mx.image.imrotate(patch_img2s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img2s,112.5), mx.image.imrotate(patch_img2s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img2s,157.5), mx.image.imrotate(patch_img2s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img2s,202.5), mx.image.imrotate(patch_img2s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img2s,247.5), mx.image.imrotate(patch_img2s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img2s,292.5), mx.image.imrotate(patch_img2s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img2s,337.5), mx.image.imrotate(patch_img2s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc2s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc2s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
                            except:
                                pass
                    del patch_img1s,patch_img1s_2,patch_img2s,patch_img2s_2,c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16
                    del c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    ################################GPU
                    desc1s=nd.array(desc1s,ctx=img1s.context)#(5,196*8,N,K1)
                    desc2s=nd.array(desc2s,ctx=img1s.context)#(5,196*8,N,K2)
                    # desc1s=desc1s[0:5,:,:,:]
                    # desc2s=desc2s[0:5,:,:,:]
                    normalized_desc1s = nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    del desc1s,desc2s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(5,N,K1,K2)
                    del normalized_desc1s,normalized_desc2s
                    # sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K1,K2)
                    # sim_mat_12=nd.mean(sim_mats,0)
                    sim_mat_12=nd.squeeze((4*sim_mats[0:1,:,:,:]+1*sim_mats[1:2,:,:,:]+2*sim_mats[2:3,:,:,:]+1*sim_mats[3:4,:,:,:]+2*sim_mats[4:5,:,:,:]+2*sim_mats[5:6,:,:,:]+0.5*sim_mats[6:7,:,:,:]+sim_mats[7:8,:,:,:]+0.5*sim_mats[8:9,:,:,:]+sim_mats[9:10,:,:,:]+2*sim_mats[10:11,:,:,:]+0.5*sim_mats[11:12,:,:,:]+sim_mats[12:13,:,:,:]+0.5*sim_mats[13:14,:,:,:]+sim_mats[14:15,:,:,:])/20,axis=0)#(N,K1,K2)
                    
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    # pdb.set_trace()
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # mask_zone=dis<=(0.11**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # mask_zone=dis>=(0.05**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    mid_indices, mask12 = self.associate(sim_mat_12*mask_zone,orb2)#(N,K1,1)
                    max_indices,mask21 = self.associate(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K2,1)
                    indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K1
                    indices_2 = indices*mask12.squeeze(2)#N,K
                    # pdb.set_trace()
                    mask=nd.broadcast_equal(indices,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask2=nd.broadcast_equal(indices_2,(nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=img1s.context),0)+1)*(mask12*2-1).squeeze(2)-1)##N,K1
                    mask=mask*mask2==1##N,K
                    print(mask.sum())
                    mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb2,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K1,2)
                    coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                    coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                    # pdb.set_trace()
                    time3=time.time()
                    print(time3-time2)
                    

                    
                    for k in range (shape[0]):
                        kp1=[]
                        kp2=[]
                        im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                        plt.figure()
                        plt.imshow(im1)
                        count_pair=0
                        for i in range (coor1.shape[1]):
                            if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                count_pair=count_pair+1
                                kp1.append(coor1[k,i,:].asnumpy().tolist())
                                kp2.append(coor2[k,i,:].asnumpy().tolist())
                                plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                        if count_pair>0:
                            plt.title(str(name_num))
                            plt.savefig(savepath1+str(name_num)+'_'+str(count_pair)+'.jpg', dpi=600)
                        plt.close()
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                        if len(kp1)!=0:
                            name = ['X', 'Y']
                            kp1=np.asarray(kp1)
                            kp2=np.asarray(kp2)#.transpose(0,1)
                            outlmk1 = pd.DataFrame(columns=name, data=kp1[:,[1,0]])
                            outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                            # outlmk1.to_csv(savepath3+str(name_num+1)+'_2.csv')
                            outlmk2 = pd.DataFrame(columns=name, data=kp2[:,[1,0]])
                            outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
                            # outlmk2.to_csv(savepath3+str(name_num+1)+'_1.csv')
                    del sim_mat_12,sim_mat_21,dis,mask_zone,mid_indices,mask12,max_indices,mask21,indices,indices_2
                    del mask,mask2,mid_orb_warp
                    
                    flag_end=0
                    if orb2_ori.shape[1]>max_orb_num:
                        orb2=orb2_ori[:,max(int(orb2_ori.shape[1]/2),int(orb2_ori.shape[1])-max_orb_num):,:]
                        flag_end=1
                    if orb1_ori.shape[1]>max_orb_num:
                        orb1=orb1_ori[:,max(int(orb1_ori.shape[1]/2),int(orb1_ori.shape[1])-max_orb_num):,:]
                        flag_end=1
                    if flag_end==0:
                        return 0
                    print(orb1.shape[1])
                    print(orb2.shape[1])
                    time1=time.time()
                    desc1s=np.zeros([15,1568*2,shape[0],orb1.shape[1]])
                    desc2s=np.zeros([15,1568*2,shape[0],orb2.shape[1]])
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
                                                    nd.contrib.BilinearResize2D(img1s_1024[k:(k+1),:,max(int((kp1_y1*2-arange6)),0):min(int((kp1_y1*2+arange6)),int(shape[2]*2)),max(int((kp1_x1*2-arange6)),0):min(int((kp1_x1*2+arange6)),int(shape[2]*2))],height=64, width=64),dim=0)
                            except:
                                patch_img1s_2=patch_img1s
                            # patch_img1s=nd.mean(nd.reshape(patch_img1s,(3,5,-1,64,64)),0)
                            # patch_img1s_2=nd.mean(nd.reshape(patch_img1s_2,(3,5,-1,64,64)),0)
                            # patch_img1s, patch_img1s_2, _ = self.centralize(patch_img1s, patch_img1s_2)
                            patch_img1s, patch_img1s_2, _ = self.centralize_kp_pairs_multiscale(patch_img1s, patch_img1s_2)
                            _, c1s, c2s,_= self.network(patch_img1s, patch_img1s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s,45), mx.image.imrotate(patch_img1s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s,90), mx.image.imrotate(patch_img1s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s,135), mx.image.imrotate(patch_img1s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s,180), mx.image.imrotate(patch_img1s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s,225), mx.image.imrotate(patch_img1s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s,270), mx.image.imrotate(patch_img1s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s,315), mx.image.imrotate(patch_img1s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img1s,22.5), mx.image.imrotate(patch_img1s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img1s,67.5), mx.image.imrotate(patch_img1s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img1s,112.5), mx.image.imrotate(patch_img1s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img1s,157.5), mx.image.imrotate(patch_img1s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img1s,202.5), mx.image.imrotate(patch_img1s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img1s,247.5), mx.image.imrotate(patch_img1s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img1s,292.5), mx.image.imrotate(patch_img1s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img1s,337.5), mx.image.imrotate(patch_img1s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc1s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc1s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
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
                            # patch_img2s=nd.mean(nd.reshape(patch_img2s,(3,5,-1,64,64)),0)
                            # patch_img2s_2=nd.mean(nd.reshape(patch_img2s_2,(3,5,-1,64,64)),0)
                            # patch_img2s, patch_img2s_2, _ = self.centralize(patch_img2s, patch_img2s_2)
                            patch_img2s, patch_img2s_2, _ = self.centralize_kp_pairs_multiscale(patch_img2s, patch_img2s_2)
                            _, c1s, c2s,_= self.network(patch_img2s, patch_img2s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img2s,45), mx.image.imrotate(patch_img2s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img2s,90), mx.image.imrotate(patch_img2s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img2s,135), mx.image.imrotate(patch_img2s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img2s,180), mx.image.imrotate(patch_img2s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img2s,225), mx.image.imrotate(patch_img2s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img2s,270), mx.image.imrotate(patch_img2s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img2s,315), mx.image.imrotate(patch_img2s_2,315))#5,196,1,1
                            
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img2s,22.5), mx.image.imrotate(patch_img2s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img2s,67.5), mx.image.imrotate(patch_img2s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img2s,112.5), mx.image.imrotate(patch_img2s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img2s,157.5), mx.image.imrotate(patch_img2s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img2s,202.5), mx.image.imrotate(patch_img2s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img2s,247.5), mx.image.imrotate(patch_img2s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img2s,292.5), mx.image.imrotate(patch_img2s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img2s,337.5), mx.image.imrotate(patch_img2s_2,337.5))#5,196,1,1
                            
                            c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                            c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                            c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                            c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                            c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                            c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                            c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                            
                            c1s_9, c2s_9=c1s_9.squeeze().asnumpy(),c2s_9.squeeze().asnumpy()
                            c1s_10, c2s_10=c1s_10.squeeze().asnumpy(),c2s_10.squeeze().asnumpy()
                            c1s_11, c2s_11=c1s_11.squeeze().asnumpy(),c2s_11.squeeze().asnumpy()
                            c1s_12, c2s_12=c1s_12.squeeze().asnumpy(),c2s_12.squeeze().asnumpy()
                            c1s_13, c2s_13=c1s_13.squeeze().asnumpy(),c2s_13.squeeze().asnumpy()
                            c1s_14, c2s_14=c1s_14.squeeze().asnumpy(),c2s_14.squeeze().asnumpy()
                            c1s_15, c2s_15=c1s_15.squeeze().asnumpy(),c2s_15.squeeze().asnumpy()
                            c1s_16, c2s_16=c1s_16.squeeze().asnumpy(),c2s_16.squeeze().asnumpy()
                            
                            desc2s[:,:,k,2*i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16),1)
                            try:
                                desc2s[:,:,k,2*i+1]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16),1)#(5,196*8,1,1)
                            except:
                                pass
                    del patch_img1s,patch_img1s_2,patch_img2s,patch_img2s_2,c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8,c1s_9,c1s_10,c1s_11,c1s_12,c1s_13,c1s_14,c1s_15,c1s_16
                    del c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8,c2s_9,c2s_10,c2s_11,c2s_12,c2s_13,c2s_14,c2s_15,c2s_16
                    time2=time.time()
                    print(time2-time1)
                    print('kp pairing')
                    ################################GPU
                    desc1s=nd.array(desc1s,ctx=img1s.context)#(5,196*8,N,K1)
                    desc2s=nd.array(desc2s,ctx=img1s.context)#(5,196*8,N,K2)
                    normalized_desc1s = nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                    del desc1s,desc2s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(5,N,K1,K2)
                    del normalized_desc1s,normalized_desc2s
                    # sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K1,K2)
                    # sim_mat_12=nd.mean(sim_mats,0)
                    sim_mat_12=nd.squeeze((4*sim_mats[0:1,:,:,:]+1*sim_mats[1:2,:,:,:]+2*sim_mats[2:3,:,:,:]+1*sim_mats[3:4,:,:,:]+2*sim_mats[4:5,:,:,:]+2*sim_mats[5:6,:,:,:]+0.5*sim_mats[6:7,:,:,:]+sim_mats[7:8,:,:,:]+0.5*sim_mats[8:9,:,:,:]+sim_mats[9:10,:,:,:]+2*sim_mats[10:11,:,:,:]+0.5*sim_mats[11:12,:,:,:]+sim_mats[12:13,:,:,:]+0.5*sim_mats[13:14,:,:,:]+sim_mats[14:15,:,:,:])/20,axis=0)#(N,K1,K2)
                    
                    del sim_mats
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # mask_zone=dis<=(0.11**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    # mask_zone=dis>=(0.05**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
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
                   
                    time3=time.time()
                    print(time3-time2)
                    del sim_mat_12,sim_mat_21,dis,mask_zone,mid_indices,mask12,max_indices,mask21,indices,indices_2
                    del mask,mask2,mid_orb_warp

                    
                    for k in range (shape[0]):
                        kp1=[]
                        kp2=[]
                        im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),img2s[k, 0, :, :].asnumpy())
                        plt.figure()
                        plt.imshow(im1)
                        count_pair=0
                        for i in range (coor1.shape[1]):
                            if not (coor1[k,i,1].asnumpy()==0 or coor1[k,i,0].asnumpy()==0):
                                count_pair=count_pair+1
                                kp1.append(coor1[k,i,:].asnumpy().tolist())
                                kp2.append(coor2[k,i,:].asnumpy().tolist())
                                plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                        if count_pair>0:
                            plt.title(str(name_num))
                            plt.savefig(savepath1+str(name_num)+'_'+str(count_pair)+'_2.jpg', dpi=600)
                        plt.close()
                        try:
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_1.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp1=lmk_temp.tolist()
                            lmk_temp = pd.read_csv(os.path.join(savepath3, str(name_num)+'_2.csv'))
                            lmk_temp = np.array(lmk_temp)
                            lmk_temp = lmk_temp[:, [2, 1]]
                            lmk_temp2=lmk_temp.tolist()
                        except:
                            pass
                        else:
                            kp1.extend(lmk_temp1)
                            kp2.extend(lmk_temp2)
                        if len(kp1)!=0:
                            name = ['X', 'Y']
                            kp1=np.asarray(kp1)
                            kp2=np.asarray(kp2)#.transpose(0,1)
                            outlmk1 = pd.DataFrame(columns=name, data=kp1[:,[1,0]])
                            outlmk1.to_csv(savepath3+str(name_num)+'_1.csv')
                            outlmk2 = pd.DataFrame(columns=name, data=kp2[:,[1,0]])
                            outlmk2.to_csv(savepath3+str(name_num)+'_2.csv')
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def rebuttle_kp_pairs_multiscale(self, dist_weight, img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s,name_num):
        img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        if 1:
        # with autograd.record():
            for img1s, img2s,img1s_256, img2s_256,img1s_1024, img2s_1024,orb1,orb2 in zip(img1, img2,img1_256, img2_256,img1_1024, img2_1024,orb1s,orb2s):
                if 1:
                    shape = img1s.shape
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/LFS_SFG_multiscale_kps1024_0.97_0.9/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite1_multiscale_kps1024_0.99_0.85/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite2_multiscale_kps1024_0.99_0.85/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite2_multiscale_kps1024_0.99_0.85/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite3_multiscale_kps1024_0.995_0.75/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite3_multiscale_kps1024_0.995_0.75/"
                    savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/baseline_ite2_multiscale_kps1024_0.97_0.9/"
                    savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/baseline_ite1_multiscale_kps1024_0.99_0.85/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85/"
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
                    
                    # orb1=orb1[:,np.where(orb1[0,:,0]>600)[0],:]
                    # orb2=orb2[:,np.where(orb2[0,:,0]>600)[0],:]
                    
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
                    ################################GPU
                    normalized_desc1s=nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(15,N,K1,196*16)
                    normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(15,N,196*16,K2)
                    del desc1s,desc2s
                    sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(15,N,K1,K2)
                    del normalized_desc1s
                    # sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K1,K2)
                    sim_mat_12=nd.mean(sim_mats,0)#(N,K1,K2)
                    # sim_mat_12=nd.squeeze((4*sim_mats[0:1,:,:,:]+1*sim_mats[1:2,:,:,:]+2*sim_mats[2:3,:,:,:]+1*sim_mats[3:4,:,:,:]+2*sim_mats[4:5,:,:,:]+2*sim_mats[5:6,:,:,:]\
                                # +0.5*sim_mats[6:7,:,:,:]+sim_mats[7:8,:,:,:]+0.5*sim_mats[8:9,:,:,:]+sim_mats[9:10,:,:,:]+2*sim_mats[10:11,:,:,:]+0.5*sim_mats[11:12,:,:,:]\
                                # +sim_mats[12:13,:,:,:]+0.5*sim_mats[13:14,:,:,:]+sim_mats[14:15,:,:,:])/20,axis=0)#(N,K1,K2)
                    sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K2,K1)
                    ####orb1(N,K1,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                    dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb2*orb2,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb2,1,2)))#N,K,K
                    
                    mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                    
                    # mask_zone1=dis>=(0.04**2)*(shape[2]**2)*2#0.028 0.015 0.04
                    # mask_zone2=dis<(0.08**2)*(shape[2]**2)*2#0.015 0.04
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
                        # coor1_klist=coor1[k,:,:].asnumpy().tolist()
                        # coor2_klist=coor2[k,:,:].asnumpy().tolist()
                        # index1_nonzero=[indextemp for indextemp,kpstemp in enumerate(coor1_klist) if kpstemp[0]!=0 and kpstemp[1]!=0]
                        # index2_nonzero=[indextemp for indextemp,kpstemp in enumerate(coor2_klist) if kpstemp[0]!=0 and kpstemp[1]!=0]
                        # index_nonzero=[]
                        # for indextemp in index1_nonzero:
                            # if indextemp in index2_nonzero:
                                # index_nonzero.append(indextemp)
                        # pdb.set_trace()
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
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/LFS_SFG_multiscale_kps1024_0.97_0.9/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/LFS_SFG_multiscale_kps1024_0.97_0.9/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite1_multiscale_kps1024_0.99_0.85/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite1_multiscale_kps1024_0.99_0.85/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite2_multiscale_kps1024_0.99_0.85/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite2_multiscale_kps1024_0.99_0.85/"
                    # savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/PCG_ite3_multiscale_kps1024_0.995_0.75/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/PCG_ite3_multiscale_kps1024_0.995_0.75/"
                    savepath1="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/baseline_ite2_multiscale_kps1024_0.97_0.9/"
                    savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9/"
                    # savepath3="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite1_multiscale_kps1024_0.99_0.85/"
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
                    
                    
                    # orb1=orb1[:,np.where(orb1[0,:,0]>600)[0],:]
                    # orb2=orb2[:,np.where(orb2[0,:,0]>600)[0],:]
                    
                    
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
                    ################################GPU
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
                    # mask_zone2=dis<(0.08**2)*(shape[2]**2)*2#0.015 0.04
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
                        # coor1_klist=coor1[k,:,:].asnumpy().tolist()
                        # coor2_klist=coor2[k,:,:].asnumpy().tolist()
                        # index1_nonzero=[indextemp for indextemp,kpstemp in enumerate(coor1_klist) if kpstemp[0]!=0 and kpstemp[1]!=0]
                        # index2_nonzero=[indextemp for indextemp,kpstemp in enumerate(coor2_klist) if kpstemp[0]!=0 and kpstemp[1]!=0]
                        # index_nonzero=[]
                        # for indextemp in index1_nonzero:
                            # if indextemp in index2_nonzero:
                                # index_nonzero.append(indextemp)
                        if index_nonzero.shape[0]>0:
                            # pdb.set_trace()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def associate_single(self, sim_mat,fkp):
        #############sim_mat:(N,K1,K2)  fkp(N,K2,2)    
        # pdb.set_trace()
        indice = nd.stop_gradient(nd.topk(sim_mat, axis=2, k=2, ret_typ='indices'))#(N,K1,2)
        fkp_ref=nd.stop_gradient(nd.diag(nd.gather_nd(nd.swapaxes(fkp,0,1),nd.transpose(nd.slice_axis(indice,axis=2,begin=0,end=1),axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)))#(K,2,N)#(N,K,N,2)
        d_temp=nd.stop_gradient(nd.abs(nd.sum(fkp_ref*fkp_ref,axis=2,keepdims=True)+nd.sum(fkp*fkp,axis=2,keepdims=True).transpose((0,2,1))-2*nd.batch_dot(fkp_ref,fkp.transpose((0,2,1)))))#N,K,K
        mask_nms1=d_temp>=(0.004**2)*(1024**2)*2#0.005
        mask_nms2=d_temp==0
        mask_nms=nd.stop_gradient(((mask_nms1+mask_nms2)>=1))
        sim = nd.stop_gradient(nd.topk(sim_mat*mask_nms,axis=2, k=2, ret_typ='value'))
        mask1=nd.stop_gradient(nd.broadcast_lesser(nd.slice_axis(sim,axis=2,begin=1,end=2),(nd.slice_axis(sim,axis=2,begin=0,end=1)*0.98)))
        mask2=nd.stop_gradient(nd.slice_axis(sim,axis=2,begin=0,end=1)>0.95)#.85#.9
        mask=nd.stop_gradient(mask1*mask2==1)
        return indice[:,:,0:1],mask#(N,K,1)
    
    
    
    
    
    
    def associate(self, sim_mat,fkp):
        #############sim_mat:(N,K1,K2)  fkp(N,K2,2)    
        # pdb.set_trace()
        indice = nd.stop_gradient(nd.topk(sim_mat, axis=2, k=2, ret_typ='indices'))#(N,K1,2)
        fkp_ref=nd.stop_gradient(nd.diag(nd.gather_nd(nd.swapaxes(fkp,0,1),nd.transpose(nd.slice_axis(indice,axis=2,begin=0,end=1),axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)))#(K,2,N)#(N,K,N,2)
        d_temp=nd.stop_gradient(nd.abs(nd.sum(fkp_ref*fkp_ref,axis=2,keepdims=True)+nd.sum(fkp*fkp,axis=2,keepdims=True).transpose((0,2,1))-2*nd.batch_dot(fkp_ref,fkp.transpose((0,2,1)))))#N,K,K
        mask_nms1=d_temp>=(0.004**2)*(1024**2)*2#0.005
        mask_nms2=d_temp==0
        # pdb.set_trace()#########判断d_temp==0是否存在
        mask_nms=nd.stop_gradient(((mask_nms1+mask_nms2)>=1))
        sim = nd.stop_gradient(nd.topk(sim_mat*mask_nms,axis=2, k=2, ret_typ='value'))
        mask1=nd.stop_gradient(nd.broadcast_lesser(nd.slice_axis(sim,axis=2,begin=1,end=2),(nd.slice_axis(sim,axis=2,begin=0,end=1)*0.97)))#0.97
        mask2=nd.stop_gradient(nd.slice_axis(sim,axis=2,begin=0,end=1)>0.9)#0.9#.85#.9
        mask=nd.stop_gradient(mask1*mask2==1)
        return indice[:,:,0:1],mask#(N,K,1)
    
    def associate_numpy(self, sim_mat,fkp):
        #############sim_mat:(N,K,K)  fkp(N,K,2)
        # pdb.set_trace()
        indice = np.argpartition(sim_mat, -2, 2)[:,:,-2:]#(N,K,2)
        fkp_ref=np.diagonal(np.take(fkp,indice[:,:,-1],axis=1),axis1=0,axis2=1).transpose((2,0,1))#(N,K,2)#(K,2,N)#(N,K,N,2)
        d_temp=np.absolute(np.sum(fkp_ref*fkp_ref,axis=2,keepdims=True)+np.sum(fkp*fkp,axis=2,keepdims=True).transpose((0,2,1))-2*np.matmul(fkp_ref,fkp.transpose((0,2,1))))#N,K,K
        mask_nms1=d_temp>=(0.004**2)*(1024**2)*2#0.005
        mask_nms2=d_temp==0
        mask_nms=((mask_nms1+mask_nms2)>=1)
        # sim = nd.topk(sim_mat*mask_nms,axis=2, k=2, ret_typ='value')
        sim = np.partition(sim_mat*mask_nms,-2, 2)[:,:,-2:]#(N,K,2)
        # mask1=nd.broadcast_lesser(nd.slice_axis(sim,axis=2,begin=1,end=2),(nd.slice_axis(sim,axis=2,begin=0,end=1)*0.98))
        mask1=sim[:,:,-2:-1]<sim[:,:,-1:]*0.98#0.98
        mask2=sim[:,:,-1:]>0.9#.85#.9
        mask=mask1*mask2==1
        return indice[:,:,1:2],mask#(N,K,1)
    
    
    
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
    def landmark_dist_for_4_indicators_original(self, lmk1s, lmk2s, flows):
        percent=0.75#0.1
        lmk1=lmk1s[:,:,0:2]
        lmk2=lmk2s[:,:,0:2]
        shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
        lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
        
        for flow in flows:
            batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
            batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
            warped_lmk = lmk1+ nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
        warped_lmk_original=(warped_lmk- shape[0]/2)/lmk2s[:,:,5:6]+shape[0]/2
        warped_lmk_original_copy = copy.deepcopy(warped_lmk_original)
        warped_lmk_original[:,:,0]=warped_lmk_original_copy[:,:,1]*lmk2s[:,:,6]+warped_lmk_original_copy[:,:,0]*(1-lmk2s[:,:,6])
        warped_lmk_original[:,:,1]=(-warped_lmk_original_copy[:,:,0]+shape[0])*lmk2s[:,:,6]+warped_lmk_original_copy[:,:,1]*(1-lmk2s[:,:,6])
        warped_lmk_original=warped_lmk_original*(lmk2s[:,:,7:9].max(2).expand_dims(axis=2))/shape[0]*lmk2s[:,:,5:6]
        lmk2_original=(lmk2- shape[0]/2)/lmk2s[:,:,5:6]+shape[0]/2
        lmk2_original_copy = copy.deepcopy(lmk2_original)
        lmk2_original[:,:,0]=lmk2_original_copy[:,:,1]*lmk2s[:,:,6]+lmk2_original_copy[:,:,0]*(1-lmk2s[:,:,6])
        lmk2_original[:,:,1]=(-lmk2_original_copy[:,:,0]+shape[0])*lmk2s[:,:,6]+lmk2_original_copy[:,:,1]*(1-lmk2s[:,:,6])
        lmk2_original=lmk2_original*(lmk2s[:,:,7:9].max(2).expand_dims(axis=2))/shape[0]*lmk2s[:,:,5:6]
        lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk_original - lmk2_original), axis=-1) * lmk_mask)
        lmk_dist_valid=lmk_dist[0,nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices').squeeze()]
        lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value',is_ascend=1)##########从小到大排列
        if int(lmk_mask.sum().asnumpy()[0])%2==0:
            lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
        else:
            lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
        lmk_dist_mean=lmk_dist_valid.mean()
        
        
        intpart=int(np.floor((int(lmk_mask.sum().asnumpy()[0])-1)*percent))
        doublepart=(int(lmk_mask.sum().asnumpy()[0])-1)*percent-intpart
        lmk_dist_percentile=(1-doublepart)*lmk_dist_valid_sorted[intpart]+doublepart*lmk_dist_valid_sorted[intpart+1]

        lmk_dist_ori = nd.sqrt(nd.sum(nd.square(lmk1 - lmk2), axis=-1))
        lmk_mask_large1 = lmk_mask*(lmk_dist_ori>18)##80
        lmk_mask_large2 = lmk_mask*(lmk_dist_ori>22)##85
        lmk_mask_large3 = lmk_mask*(lmk_dist_ori>30)##90
        lmk_mask_large4 = lmk_mask*(lmk_dist_ori>43)##95
        lmk_mask_large5 = lmk_mask*(lmk_dist_ori>58)##98
        # if (lmk_dist_ori>58).sum().asnumpy()[0]>0:
            # pdb.set_trace()
        lmk_dist_large_mean1=lmk_dist[0,nd.topk(lmk_mask_large1, axis=1, k=int(lmk_mask_large1.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large1.sum().asnumpy()[0]>0)
        lmk_dist_large_mean2=lmk_dist[0,nd.topk(lmk_mask_large2, axis=1, k=int(lmk_mask_large2.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large2.sum().asnumpy()[0]>0)
        lmk_dist_large_mean3=lmk_dist[0,nd.topk(lmk_mask_large3, axis=1, k=int(lmk_mask_large3.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large3.sum().asnumpy()[0]>0)
        lmk_dist_large_mean4=lmk_dist[0,nd.topk(lmk_mask_large4, axis=1, k=int(lmk_mask_large4.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large4.sum().asnumpy()[0]>0)
        lmk_dist_large_mean5=lmk_dist[0,nd.topk(lmk_mask_large5, axis=1, k=int(lmk_mask_large5.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large5.sum().asnumpy()[0]>0)
        return lmk_dist_mean / nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)), lmk_dist_median/nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                , lmk_dist_percentile/nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean1/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                ,lmk_dist_large_mean2/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean3/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                ,lmk_dist_large_mean4/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean5/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),\
                warped_lmk, lmk2,lmk_dist_ori
        
        
        
    def landmark_dist_for_4_indicators(self, lmk1, lmk2, flows):
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5)
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk_dist_valid=lmk_dist[0,valid_index.squeeze()]
            lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value')
            if int(lmk_mask.sum().asnumpy()[0])%2==0:
                lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
            else:
                lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
            lmk_dist_mean=lmk_dist_valid.mean()
            lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)#mean rtre######问题在于点的数目取mean,非有效值和有效值均加了1e-5，导致结果有一定误差
            lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*(np.shape(lmk2)[1]) # 消除当kp数目为0的时候的影响,考虑到batch可能大于1
            lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            # pdb.set_trace()
            return lmk_dist / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),warped_lmk, lmk2
            # return lmk_dist_mean / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),warped_lmk, lmk2
        else:
            return 0, [], []
    def landmark_dist_for_6_indicators_ANHIR_train(self, lmk1, lmk2, flows):
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5)
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            # pdb.set_trace()
            lmk_dist_valid=lmk_dist[0,valid_index.squeeze()]
            lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value')
            # pdb.set_trace()
            if int(lmk_mask.sum().asnumpy()[0])%2==0:
                lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
            else:
                lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
            lmk_dist_mean=lmk_dist_valid.mean()
            lmk_dist_max=lmk_dist_valid_sorted[0]
            return lmk_dist_mean / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),lmk_dist_max/(shape[0]*1.414),warped_lmk, lmk2
        else:
            return 0, [], []
    def landmark_dist_for_large_displacement(self, lmk1, lmk2, flows):
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            # pdb.set_trace()
            lmk_dist_ori = nd.sqrt(nd.sum(nd.square(lmk1 - lmk2), axis=-1))
            lmk_mask = ((1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5)*(lmk_dist_ori>8)#######初始距离大于8的认为是大位移
            if lmk_mask.sum().asnumpy()[0]<1:
                return lmk_mask.sum()[0], lmk_mask.sum()[0],[],[],lmk_mask.sum()[0]
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5)
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk_dist_valid=lmk_dist[0,valid_index.squeeze()]
            lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value')
            if int(lmk_mask.sum().asnumpy()[0])%2==0:
                lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
            else:
                lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
            lmk_dist_mean=lmk_dist_valid.mean()
            lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)#mean rtre######问题在于点的数目取mean,非有效值和有效值均加了1e-5，导致结果有一定误差
            lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*(np.shape(lmk2)[1]) # 消除当kp数目为0的时候的影响,考虑到batch可能大于1
            lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            # pdb.set_trace()
            return lmk_dist / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),warped_lmk, lmk2,lmk_mask.sum()[0]
            # return lmk_dist_mean / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),warped_lmk, lmk2
        else:
            return 0, [], []
    
    def landmark_dist_for_90th_percentile(self, lmk1, lmk2, flows):
        percent=0.25#0.1
        # pdb.set_trace()
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5)
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk_dist_valid=lmk_dist[0,valid_index.squeeze()]
            lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value')
            intpart=int(np.floor((int(lmk_mask.sum().asnumpy()[0])-1)*percent))
            doublepart=(int(lmk_mask.sum().asnumpy()[0])-1)*percent-intpart
            lmk_dist_percentile=(1-doublepart)*lmk_dist_valid_sorted[intpart]+doublepart*lmk_dist_valid_sorted[intpart+1]
            lmk_dist_mean=lmk_dist_valid.mean()
            lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)#mean rtre######问题在于点的数目取mean,非有效值和有效值均加了1e-5，导致结果有一定误差
            lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*(np.shape(lmk2)[1]) # 消除当kp数目为0的时候的影响,考虑到batch可能大于1
            lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            # pdb.set_trace()
            return lmk_dist / (shape[0]*1.414), lmk_dist_percentile/(shape[0]*1.414),warped_lmk, lmk2
            # return lmk_dist_mean / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),warped_lmk, lmk2
        else:
            return 0, [], []
    def landmark_dist_for_submit(self, lmk1, flows):
        if np.shape(lmk1)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            return warped_lmk
        else:
            return 0
    def landmark_dist_for_submit_train(self, lmk1,lmk2, flows):
        if np.shape(lmk1)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask)
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk_dist_valid=lmk_dist[0,valid_index.squeeze()]
            lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value')
            if int(lmk_mask.sum().asnumpy()[0])%2==0:
                lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
            else:
                lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
            lmk_dist_mean=lmk_dist_valid.mean()
            lmk_dist_max=lmk_dist_valid_sorted[0]
            return lmk_dist_mean / (shape[0]*1.414), lmk_dist_median/(shape[0]*1.414),lmk_dist_max/(shape[0]*1.414),warped_lmk
            
            return warped_lmk
        else:
            return 0
    
    # def landmark_dist_validate(self, lmk1, lmk2, flows):
        # if np.shape(lmk2)[0] > 0:
            # shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            # lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5
            # for flow in flows:
                # batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
                # batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                # warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
            # lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)#mean rtre######问题在于点的数目取mean
            # lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*(np.shape(lmk2)[1]) # 消除当kp数目为0的时候的影响,考虑到batch可能大于1
            # lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            # return lmk_dist / (shape[0]*1.414), warped_lmk, lmk2
        # else:
            # return 0, [], []
    def landmark_dist_v(self, lmk1, lmk2, flows):
        lmknew = np.zeros((np.shape(lmk1)[0], np.shape(lmk1)[1], np.shape(lmk1)[2]))
        lmk2new = np.zeros((np.shape(lmk2)[0], np.shape(lmk2)[1], np.shape(lmk2)[2]))
        if np.shape(lmk2)[0] > 0:
            flow_len = len(flows)
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)
            lmk_dist_all = nd.ones((np.shape(lmk1)[0],), ctx=flows[0].context)
            lmk_dist_all2 = []
            for k in range(0, np.shape(lmk1)[0]):
                # old lmk_mask is when lmk1 and lmk2 are all not 0, 不知为何，会出现非零的补充。所以改为和200项最后一项相同的就不要
                lmk1n = lmk1[k]
                lmk1n = lmk1n.reshape(1, np.shape(lmk1n)[0], np.shape(lmk1n)[1])
                lmk2n = lmk2[k]
                lmk2n = lmk2n.reshape(1, np.shape(lmk2n)[0], np.shape(lmk2n)[1])
                lmk_mask = (1 - (lmk1n[0, :, 0] * lmk1n[0, :, 1] == lmk1n[0][199][0] * lmk1n[0][199][1])) * (1 - (lmk2n[0, :, 0] * lmk2n[0, :, 1] == lmk2n[0][199][0] * lmk2n[0][199][1])) > 0.5
                mask_num = np.sum(lmk_mask)  # gl resuse lmk_mask
                mask_num = int(mask_num.asnumpy())  # gl resuse lmk_mask
                lmk1n = lmk1n[:, :mask_num, :]  # gl resuse lmk_mask
                lmk2n = lmk2n[:, :mask_num, :]  # gl resuse lmk_mask
                for flow in flows:
                    flow = flow[k]
                    flow = flow.reshape(1, np.shape(flow)[0], np.shape(flow)[1], np.shape(flow)[2])
                    batch_lmk = lmk1n / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1
                    batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                    warped_lmk = lmk1n + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))
                # start: median rTRE
                lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2n), axis=-1) + 1e-5)
                lmk_dist_numpy = []
                for m in range(0, np.shape(lmk_dist)[1]):
                    lmk_dist_numpy.append(lmk_dist[0, m].asnumpy())
                if np.shape(lmk_dist)[1] % 2 == 0:
                    med = lmk_dist_numpy.index(np.median(lmk_dist_numpy[1:]))
                else:
                    med = lmk_dist_numpy.index(np.median(lmk_dist_numpy))
                lmk_dist_median = lmk_dist[0,med]
                lmk_dist_all[k]=lmk_dist_median.asnumpy()
                # end:median rTRE
                # # start: mean rTRE
                # lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2n), axis=-1) + 1e-5), axis=-1)
                # lmk_dist_all[k] = lmk_dist.asnumpy()
                # # end: mean rTRE
                lmk2new[k, :mask_num, :] = lmk2n.asnumpy()
                lmknew[k, :mask_num, :] = warped_lmk.asnumpy()

            return lmk_dist_all / (shape[0]*1.414), lmknew, lmk2new
        else:
            return 0, [], []
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
    def Get_Jac(self,displacement):
        '''
        the expected input: displacement of shape(batch, H, W, D, channel),
        obtained in TensorFlow.
        '''
        D_y = (displacement[:, 1:, :-1,  :] - displacement[:, :-1, :-1,  :])
        D_x = (displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1,  :])
        #D_z = (displacement[:, :-1, :-1,  :] - displacement[:, :-1, :-1,  :])

        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = (D_y[..., 0] ) * (D_x[..., 1] )

        D = D1 - D2

        return D
    
    
    def landmark_dist_for_4_indicators_original_ACROBAT(self, lmk1s, lmk2s, flows):
        percent=0.75#0.1
        percent2=0.9#0.1
        lmk1=lmk1s[:,:,0:2]
        lmk2=lmk2s[:,:,0:2]
        shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
        # lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
        lmk_mask = ((1 - nd.prod(lmk1 <= 0, axis=-1)) > 0.5)*((1 - nd.prod(lmk2 <= 0, axis=-1)) > 0.5)
        for flow in flows:
            batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
            batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
            warped_lmk0 = lmk1+ nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
        lmk_dist0 = nd.sqrt(nd.sum(nd.square(warped_lmk0 - lmk2), axis=-1) * lmk_mask)
        warped_lmk=copy.deepcopy(warped_lmk0)
        warped_lmk[:,:,0]=(warped_lmk[:,:,0]/(shape[0]-1)*(lmk2s[:,:,4]-lmk2s[:,:,3]+lmk2s[:,:,11]+lmk2s[:,:,10])+lmk2s[:,:,3]-1-lmk2s[:,:,10])*lmk2s[:,:,2]
        warped_lmk[:,:,1]=(warped_lmk[:,:,1]/(shape[0]-1)*(lmk2s[:,:,6]-lmk2s[:,:,5]+lmk2s[:,:,9]+lmk2s[:,:,8])+lmk2s[:,:,5]-1-lmk2s[:,:,8])*lmk2s[:,:,2]
        lmk2[:,:,0]=(lmk2[:,:,0]/(shape[0]-1)*(lmk2s[:,:,4]-lmk2s[:,:,3]+lmk2s[:,:,11]+lmk2s[:,:,10])+lmk2s[:,:,3]-1-lmk2s[:,:,10])*lmk2s[:,:,2]
        lmk2[:,:,1]=(lmk2[:,:,1]/(shape[0]-1)*(lmk2s[:,:,6]-lmk2s[:,:,5]+lmk2s[:,:,9]+lmk2s[:,:,8])+lmk2s[:,:,5]-1-lmk2s[:,:,8])*lmk2s[:,:,2]
        lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask)
        # print(lmk_dist)
        # pdb.set_trace()
        lmk_dist_valid=lmk_dist[0,nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices').squeeze()]
        lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value',is_ascend=1)##########从小到大排列
        if int(lmk_mask.sum().asnumpy()[0])%2==0:
            lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
        else:
            lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
        lmk_dist_mean=lmk_dist_valid.mean()
        
        intpart=int(np.floor((int(lmk_mask.sum().asnumpy()[0])-1)*percent))
        doublepart=(int(lmk_mask.sum().asnumpy()[0])-1)*percent-intpart
        try:
            lmk_dist_percentile=(1-doublepart)*lmk_dist_valid_sorted[intpart]+doublepart*lmk_dist_valid_sorted[intpart+1]
        except:
            if lmk_dist_valid_sorted.shape[0]<=intpart+1:
                lmk_dist_percentile=lmk_dist_valid_sorted[-1]
            else:
                print(intpart)
                print(lmk_mask.sum().asnumpy()[0])
                pdb.set_trace()
        intpart2=int(np.floor((int(lmk_mask.sum().asnumpy()[0])-1)*percent2))
        doublepart2=(int(lmk_mask.sum().asnumpy()[0])-1)*percent2-intpart2
        try:
            lmk_dist_percentile2=(1-doublepart2)*lmk_dist_valid_sorted[intpart2]+doublepart2*lmk_dist_valid_sorted[intpart2+1]
        except:
            if lmk_dist_valid_sorted.shape[0]<=intpart2+1:
                lmk_dist_percentile2=lmk_dist_valid_sorted[-1]
            else:
                print(intpart2)
                print(lmk_mask.sum().asnumpy()[0])
                pdb.set_trace()
        # pdb.set_trace()
        lmk_dist_ori = nd.sqrt(nd.sum(nd.square(lmk1 - lmk2s[:,:,0:2]), axis=-1)* lmk_mask)
        lmk_dist_ori_valid = lmk_dist_ori[0,nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices').squeeze()]
        # lmk_mask_large1 = lmk_mask*(lmk_dist_ori>5.744401454925537/512*shape[0])##80
        # lmk_mask_large2 = lmk_mask*(lmk_dist_ori>6.859210968017578/512*shape[0])##85
        # lmk_mask_large3 = lmk_mask*(lmk_dist_ori>8.676155090332031/512*shape[0])##90
        # lmk_mask_large4 = lmk_mask*(lmk_dist_ori>12.046236991882324/512*shape[0])##95
        # lmk_mask_large5 = lmk_mask*(lmk_dist_ori>16.593781661987308/512*shape[0])##98
        lmk_mask_large1 = lmk_mask*(lmk_dist_ori>5.6506268501281784/512*shape[0])##80#########delete_kps
        lmk_mask_large2 = lmk_mask*(lmk_dist_ori>6.732109117507934/512*shape[0])##85
        lmk_mask_large3 = lmk_mask*(lmk_dist_ori>8.467962741851807/512*shape[0])##90
        lmk_mask_large4 = lmk_mask*(lmk_dist_ori>11.667794275283802/512*shape[0])##95
        lmk_mask_large5 = lmk_mask*(lmk_dist_ori>16.48706272125243/512*shape[0])##98
        lmk_dist_large_mean1=lmk_dist[0,nd.topk(lmk_mask_large1, axis=1, k=int(lmk_mask_large1.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large1.sum().asnumpy()[0]>0)
        lmk_dist_large_mean2=lmk_dist[0,nd.topk(lmk_mask_large2, axis=1, k=int(lmk_mask_large2.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large2.sum().asnumpy()[0]>0)
        lmk_dist_large_mean3=lmk_dist[0,nd.topk(lmk_mask_large3, axis=1, k=int(lmk_mask_large3.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large3.sum().asnumpy()[0]>0)
        lmk_dist_large_mean4=lmk_dist[0,nd.topk(lmk_mask_large4, axis=1, k=int(lmk_mask_large4.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large4.sum().asnumpy()[0]>0)
        lmk_dist_large_mean5=lmk_dist[0,nd.topk(lmk_mask_large5, axis=1, k=int(lmk_mask_large5.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large5.sum().asnumpy()[0]>0)
        return lmk_dist_mean, lmk_dist_median, lmk_dist_percentile,lmk_dist_percentile2,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                lmk_dist_large_mean4,lmk_dist_large_mean5,warped_lmk0, lmk2,lmk_dist_ori,lmk_dist_valid
                # lmk_dist_large_mean4,lmk_dist_large_mean5,warped_lmk0, lmk2,lmk_dist_ori,lmk_dist_ori_valid,lmk_dist_valid

        # return lmk_dist_mean, lmk_dist_median,lmk_dist_percentile2,warped_lmk, lmk2,lmk_dist_ori,lmk_dist_valid
    
    def warp_landmark_ACROBAT_for_submit(self, lmk1s, lmk2s, flows):
        percent=0.9
        lmk1=lmk1s[:,:,0:2]
        lmk2=lmk2s[:,:,0:2]
        shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
        # lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
        lmk_mask = (1 - nd.prod(lmk2 <= 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
        # pdb.set_trace()
        for flow in flows:
            batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1#坐标归一化到[-1,1]
            batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
            warped_lmk = lmk1+ nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose((0, 2, 1))#batch_lmk.flip(axis=1)是把行列坐标变为x,y坐标
        warped_lmk[:,:,0]=(warped_lmk[:,:,0]/(shape[0]-1)*(lmk2s[:,:,4]-lmk2s[:,:,3]+lmk2s[:,:,11]+lmk2s[:,:,10])+lmk2s[:,:,3]-1-lmk2s[:,:,10])*lmk2s[:,:,2]
        warped_lmk[:,:,1]=(warped_lmk[:,:,1]/(shape[0]-1)*(lmk2s[:,:,6]-lmk2s[:,:,5]+lmk2s[:,:,9]+lmk2s[:,:,8])+lmk2s[:,:,5]-1-lmk2s[:,:,8])*lmk2s[:,:,2]
        lmk2[:,:,0]=(lmk2[:,:,0]/(shape[0]-1)*(lmk2s[:,:,4]-lmk2s[:,:,3]+lmk2s[:,:,11]+lmk2s[:,:,10])+lmk2s[:,:,3]-1-lmk2s[:,:,10])*lmk2s[:,:,2]
        lmk2[:,:,1]=(lmk2[:,:,1]/(shape[0]-1)*(lmk2s[:,:,6]-lmk2s[:,:,5]+lmk2s[:,:,9]+lmk2s[:,:,8])+lmk2s[:,:,5]-1-lmk2s[:,:,8])*lmk2s[:,:,2]
        lmk_dist = nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask)
        lmk_dist_valid=lmk_dist[0,nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices').squeeze()]
        lmk_dist_valid_sorted=nd.topk(lmk_dist_valid, axis=0, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='value',is_ascend=1)##########从小到大排列
        if int(lmk_mask.sum().asnumpy()[0])%2==0:
            lmk_dist_median=(lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)]+lmk_dist_valid_sorted[int(lmk_mask.sum().asnumpy()[0]/2)-1])/2
        else:
            lmk_dist_median=lmk_dist_valid_sorted[int((lmk_mask.sum().asnumpy()[0]-1)/2)]
        lmk_dist_mean=lmk_dist_valid.mean()

        intpart=int(np.floor((int(lmk_mask.sum().asnumpy()[0])-1)*percent))
        doublepart=(int(lmk_mask.sum().asnumpy()[0])-1)*percent-intpart
        try:
            lmk_dist_percentile=(1-doublepart)*lmk_dist_valid_sorted[intpart]+doublepart*lmk_dist_valid_sorted[intpart+1]
        except:
            if lmk_dist_valid_sorted.shape[0]<=intpart+1:
                lmk_dist_percentile=lmk_dist_valid_sorted[-1]
            else:
                print(intpart)
                print(lmk_mask.sum().asnumpy()[0])
                pdb.set_trace()
        if np.abs(lmk_dist_percentile.asnumpy().mean()-np.percentile(lmk_dist_valid_sorted.asnumpy(),90))>0.01:
            pdb.set_trace()
        
        
        lmk_dist_ori = nd.sqrt(nd.sum(nd.square(lmk1 - lmk2), axis=-1))
        # lmk_mask_large1 = lmk_mask*(lmk_dist_ori>18)##80
        # lmk_mask_large2 = lmk_mask*(lmk_dist_ori>22)##85
        # lmk_mask_large3 = lmk_mask*(lmk_dist_ori>30)##90
        # lmk_mask_large4 = lmk_mask*(lmk_dist_ori>43)##95
        # lmk_mask_large5 = lmk_mask*(lmk_dist_ori>58)##98
        # lmk_dist_large_mean1=lmk_dist[0,nd.topk(lmk_mask_large1, axis=1, k=int(lmk_mask_large1.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large1.sum().asnumpy()[0]>0)
        # lmk_dist_large_mean2=lmk_dist[0,nd.topk(lmk_mask_large2, axis=1, k=int(lmk_mask_large2.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large2.sum().asnumpy()[0]>0)
        # lmk_dist_large_mean3=lmk_dist[0,nd.topk(lmk_mask_large3, axis=1, k=int(lmk_mask_large3.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large3.sum().asnumpy()[0]>0)
        # lmk_dist_large_mean4=lmk_dist[0,nd.topk(lmk_mask_large4, axis=1, k=int(lmk_mask_large4.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large4.sum().asnumpy()[0]>0)
        # lmk_dist_large_mean5=lmk_dist[0,nd.topk(lmk_mask_large5, axis=1, k=int(lmk_mask_large5.sum().asnumpy()[0]), ret_typ='indices').squeeze()].mean()*int(lmk_mask_large5.sum().asnumpy()[0]>0)
        # return lmk_dist_mean / nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)), lmk_dist_median/nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                # , lmk_dist_percentile/nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean1/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                # ,lmk_dist_large_mean2/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean3/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1))\
                # ,lmk_dist_large_mean4/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),lmk_dist_large_mean5/ nd.sqrt(nd.sum(nd.square(lmk2s[:,0,9:11]), axis=-1)),\
                # warped_lmk, lmk2,lmk_dist_ori,lmk_dist_valid
        return lmk_dist_mean, lmk_dist_median,lmk_dist_percentile,warped_lmk, lmk2,lmk_dist_ori,lmk_dist_valid
    
    def validate_ACROBAT(self, data):
        results = []
        raws=[]
        evas = []
        evas_mask = []
        size = len(data)
        bs = len(self.ctx)
        output_cnt = 0
        dist_loss_means=[]
        dist_loss_medians=[]
        dist_loss_75ths=[]
        dist_loss_90ths=[]
        lmk_dist_large_mean1s=[]
        lmk_dist_large_mean2s=[]
        lmk_dist_large_mean3s=[]
        lmk_dist_large_mean4s=[]
        lmk_dist_large_mean5s=[]
        MIs=[]
        lmk_dist_oris=[]
        matrix = []
        for j in range(0, size, bs):
        # for j in range(80, 81, bs):
            batch_data = data[j: j + bs]
            ctx = self.ctx[ : min(len(batch_data), len(self.ctx))]
            nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split = False) for i in range(len(batch_data[0]))]
            for img1, img2, lmk1, lmk2 in zip(*nd_data):
                img1, img2 = img1 / 255.0, img2 / 255.0
                img1, img2, rgb_mean = self.centralize(img1, img2)
                pred, _,_,_ = self.network(img1, img2)
                shape = img1.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                flows = []
                flows.append(flow)
                warp = self.reconstruction(img2, flow)
                dist_loss_mean, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                lmk_dist_large_mean4,lmk_dist_large_mean5,_, _,lmk_dist_ori,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1, lmk2, flows)
                dist_loss_means.append(dist_loss_mean.asnumpy())
                dist_loss_medians.append(dist_loss_median.asnumpy())
                dist_loss_75ths.append(dist_loss_75th.asnumpy())
                dist_loss_90ths.append(dist_loss_90th.asnumpy())
                
                if lmk_dist_large_mean1>0:
                    lmk_dist_large_mean1s.append(lmk_dist_large_mean1.asnumpy())
                if lmk_dist_large_mean2>0:
                    lmk_dist_large_mean2s.append(lmk_dist_large_mean2.asnumpy())
                if lmk_dist_large_mean3>0:
                    lmk_dist_large_mean3s.append(lmk_dist_large_mean3.asnumpy())
                if lmk_dist_large_mean4>0:
                    lmk_dist_large_mean4s.append(lmk_dist_large_mean4.asnumpy())
                if lmk_dist_large_mean5>0:
                    lmk_dist_large_mean5s.append(lmk_dist_large_mean5.asnumpy())
                MI=MILoss_wxy(img1[0,0,:,:].asnumpy(),warp[0,0,:,:].asnumpy())
                MIs.append(MI)
                # lmk_dist_oris.append(lmk_dist_ori.asnumpy().squeeze().tolist())
        # lmk_dist_oris=np.array(lmk_dist_oris).reshape(-1)
        # lmk_dist_oris=lmk_dist_oris[np.where(lmk_dist_oris)[0]]
        # pdb.set_trace()
        return np.mean(dist_loss_means),np.std(dist_loss_means),np.median(dist_loss_means),np.percentile(dist_loss_means,75),np.percentile(dist_loss_means,90)\
               ,np.mean(dist_loss_medians),np.std(dist_loss_medians),np.median(dist_loss_medians),np.mean(dist_loss_75ths),np.std(dist_loss_75ths)\
               ,np.percentile(dist_loss_75ths,75),np.mean(dist_loss_90ths),np.std(dist_loss_90ths),np.percentile(dist_loss_90ths,90),np.mean(lmk_dist_large_mean1s)\
               ,np.std(lmk_dist_large_mean1s),np.mean(lmk_dist_large_mean2s),np.std(lmk_dist_large_mean2s),np.mean(lmk_dist_large_mean3s),np.std(lmk_dist_large_mean3s)\
               ,np.mean(lmk_dist_large_mean4s),np.std(lmk_dist_large_mean4s),np.mean(lmk_dist_large_mean5s),np.std(lmk_dist_large_mean5s),np.mean(MIs),np.std(MIs),np.median(dist_loss_75ths),np.median(dist_loss_90ths)
    def validate_ACROBAT_large(self, data):
        results = []
        raws=[]
        evas = []
        evas_mask = []
        size = len(data)
        bs = len(self.ctx)
        output_cnt = 0
        dist_loss_means=[]
        dist_loss_meansi=[]
        dist_loss_medians=[]
        dist_loss_75ths=[]
        dist_loss_90ths=[]
        lmk_dist_large_mean1s=[]
        lmk_dist_large_mean2s=[]
        lmk_dist_large_mean3s=[]
        lmk_dist_large_mean4s=[]
        lmk_dist_large_mean5s=[]
        MIs=[]
        lmk_dist_oris=[]
        matrix = []
        for j in range(0, size, bs):
        # for j in range(80, 81, bs):
            batch_data = data[j: j + bs]
            ctx = self.ctx[ : min(len(batch_data), len(self.ctx))]
            nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split = False) for i in range(len(batch_data[0]))]
            for img1, img2, lmk1, lmk2 in zip(*nd_data):
                img1, img2 = img1 / 255.0, img2 / 255.0
                img1, img2, rgb_mean = self.centralize(img1, img2)
                pred, _,_,_ = self.network(img1, img2)
                shape = img1.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                flows = []
                flows.append(flow)
                warp = self.reconstruction(img2, flow)
                if j!=80:
                    dist_loss_mean, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                    lmk_dist_large_mean4,lmk_dist_large_mean5,_, _,lmk_dist_ori,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1, lmk2, flows)
                else:
                    lmk1d=copy.deepcopy(lmk1)
                    lmk2d=copy.deepcopy(lmk2)
                    lmk1d[:,53:60,:]=lmk1[:,61,:]
                    lmk2d[:,53:60,:]=lmk2[:,61,:]
                    dist_loss_mean, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                    lmk_dist_large_mean4,lmk_dist_large_mean5,_, _,lmk_dist_ori,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1d, lmk2d, flows)
                dist_loss_means.append(dist_loss_mean.asnumpy())
                dist_loss_medians.append(dist_loss_median.asnumpy())
                dist_loss_75ths.append(dist_loss_75th.asnumpy())
                dist_loss_90ths.append(dist_loss_90th.asnumpy())
                
                if lmk_dist_large_mean1>0:
                    lmk_dist_large_mean1s.append(lmk_dist_large_mean1.asnumpy())
                if lmk_dist_large_mean2>0:
                    lmk_dist_large_mean2s.append(lmk_dist_large_mean2.asnumpy())
                if lmk_dist_large_mean3>0:
                    lmk_dist_large_mean3s.append(lmk_dist_large_mean3.asnumpy())
                if lmk_dist_large_mean4>0:
                    lmk_dist_large_mean4s.append(lmk_dist_large_mean4.asnumpy())
                if lmk_dist_large_mean5>0:
                    lmk_dist_large_mean5s.append(lmk_dist_large_mean5.asnumpy())
                MI=MILoss_wxy(img1[0,0,:,:].asnumpy(),warp[0,0,:,:].asnumpy())
                MIs.append(MI)
                
                
                if j==80:
                    dist_loss_mean57, _,_,_,_,_,_,_,_,_, _,_,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1, lmk2, flows)
                
                
                # lmk_dist_oris.append(lmk_dist_ori.asnumpy().squeeze().tolist())
        # lmk_dist_oris=np.array(lmk_dist_oris).reshape(-1)
        # lmk_dist_oris=lmk_dist_oris[np.where(lmk_dist_oris)[0]]
        # pdb.set_trace()
        return dist_loss_mean57.asnumpy(),np.mean(dist_loss_means),np.std(dist_loss_means),np.median(dist_loss_means),np.percentile(dist_loss_means,75),np.percentile(dist_loss_means,90)\
               ,np.mean(dist_loss_medians),np.std(dist_loss_medians),np.median(dist_loss_medians),np.mean(dist_loss_75ths),np.std(dist_loss_75ths)\
               ,np.percentile(dist_loss_75ths,75),np.mean(dist_loss_90ths),np.std(dist_loss_90ths),np.percentile(dist_loss_90ths,90),np.mean(lmk_dist_large_mean1s)\
               ,np.std(lmk_dist_large_mean1s),np.mean(lmk_dist_large_mean2s),np.std(lmk_dist_large_mean2s),np.mean(lmk_dist_large_mean3s),np.std(lmk_dist_large_mean3s)\
               ,np.mean(lmk_dist_large_mean4s),np.std(lmk_dist_large_mean4s),np.mean(lmk_dist_large_mean5s),np.std(lmk_dist_large_mean5s),np.mean(MIs),np.std(MIs),np.median(dist_loss_75ths),np.median(dist_loss_90ths)
    def validate_Ttest(self, data):
        results = []
        raws=[]
        evas = []
        evas_mask = []
        size = len(data)
        bs = len(self.ctx)
        output_cnt = 0
        lmk_dist_valids=[]
        lmk_dist_oris=[]
        lmk_dist_oris_valid=[]
        for j in range(0, size, bs):
        # for j in range(80, 81, bs):
            batch_data = data[j: j + bs]
            ctx = self.ctx[ : min(len(batch_data), len(self.ctx))]
            nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split = False) for i in range(len(batch_data[0]))]
            for img1, img2, lmk1, lmk2 in zip(*nd_data):
                img1, img2 = img1 / 255.0, img2 / 255.0
                img1, img2, rgb_mean = self.centralize(img1, img2)
                pred, _,_,_ = self.network(img1, img2)
                
                shape = img1.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                flows = []
                flows.append(flow)
                dist_loss_mean, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                lmk_dist_large_mean4,lmk_dist_large_mean5,_, _,lmk_dist_ori,lmk_dist_ori_valid,lmk_dist_valid = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1, lmk2, flows)
                try:
                    lmk_dist_valids.extend(lmk_dist_valid.asnumpy().squeeze().tolist())
                except:
                    lmk_dist_valids.extend(lmk_dist_valid.asnumpy().tolist())
                lmk_dist_oris.extend(lmk_dist_ori.asnumpy().squeeze().tolist())
                try:
                    lmk_dist_oris_valid.extend(lmk_dist_ori_valid.asnumpy().squeeze().tolist())
                except:
                    lmk_dist_oris_valid.extend(lmk_dist_ori_valid.asnumpy().tolist())
        return np.array(lmk_dist_valids),np.array(lmk_dist_oris),np.array(lmk_dist_oris_valid)
        
    def validate_ACROBAT_for_submit(self, data,valid_pairs,checkpoint):
        results = []
        raws=[]
        evas = []
        evas_mask = []
        size = len(data)
        
        bs = len(self.ctx)
        output_cnt = 0
        dist_loss_means=[]
        dist_loss_medians=[]
        dist_loss_75ths=[]
        matrix = []
        for j in range(0, size, bs):
            batch_data = data[j: j + bs]
            ctx = self.ctx[ : min(len(batch_data), len(self.ctx))]
            nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split = False) for i in range(len(batch_data[0]))]
            for img1, img2, lmk1, lmk2 in zip(*nd_data):
                img1, img2 = img1 / 255.0, img2 / 255.0
                img1, img2, rgb_mean = self.centralize(img1, img2)
                pred, _,_,_ = self.network(img1, img2)
                shape = img1.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                flows = []
                flows.append(flow)
                
                dist_loss_mean, dist_loss_median,dist_loss_75th,warped_lmk, _,_,_ = self.warp_landmark_ACROBAT_for_submit(lmk1, lmk2, flows)
                dist_loss_means.append(dist_loss_mean.asnumpy())
                dist_loss_medians.append(dist_loss_median.asnumpy())
                dist_loss_75ths.append(dist_loss_75th.asnumpy())
                lmk1_temp=lmk1[0,:,0:2]
                lmk2_temp=lmk2[0,:,0:2]
                lmk_mask = (1 - nd.prod(lmk1_temp <= 0, axis=-1)) > 0.5#####添加掩膜，去除lmk1中的0值
                if j==0:
                    csvpath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/kps_for_submit/annotated_kps_validation.csv"
                else:
                    csvpath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/kps_for_submit/submit_"+checkpoint+'_0127-python.xlsx'
                csvdata2 =np.array(pd.read_excel(csvpath,header=None,index_col=None))
                num=valid_pairs[j][0].split('_')[0]
                # pdb.set_trace()
                csvdata2[np.where(csvdata2[:,1]==int(num))[0],10]=warped_lmk[0,0:int(lmk_mask.sum().asnumpy()[0]),1].asnumpy()
                csvdata2[np.where(csvdata2[:,1]==int(num))[0],11]=warped_lmk[0,0:int(lmk_mask.sum().asnumpy()[0]),0].asnumpy()
                fwrite = pd.DataFrame(csvdata2)
                fwrite.to_excel("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/kps_for_submit/submit_"+checkpoint+'_0127-python.xlsx', index=False,header=False)
                
        return np.mean(dist_loss_means),np.std(dist_loss_means),np.median(dist_loss_means), np.median(dist_loss_75ths), np.mean(dist_loss_75ths),np.std(dist_loss_75ths)
    
    
    
    
    
    def validate_for_4_indicators_original(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count):
        raws = []
        dist_mean = []
        dist_median = []
        dist_75th = []
        jacobi_means = []
        jacobi_stds = []
        jacobi_foldings = []
        lmk_dist_large_mean1s = []
        lmk_dist_large_mean2s = []
        lmk_dist_large_mean3s = []
        lmk_dist_large_mean4s = []
        lmk_dist_large_mean5s = []
        # nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
        
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1, lmk2 in zip(img1s, img2s, lmk1s, lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            # pdb.set_trace()
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean().asnumpy())
            # pdb.set_trace()
            dist_loss_mean, dist_loss_median,dist_loss_75th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,lmk_dist_large_mean4,lmk_dist_large_mean5\
                                                ,warped_lmk, lmk2new,lmk_dist_ori = self.landmark_dist_for_4_indicators_original(lmk1, lmk2, flows)
            # dist_mean.append()
            # dist_median.append()
            # dist_75th.append()
            
            jacobi=self.Get_Jac(nd.transpose(flow,(0,2,3,1))).asnumpy()
            jacobi_means.append(np.mean(jacobi))
            jacobi_stds.append(np.std(jacobi))
            jacobi_foldings.append(np.sum(jacobi<0)/(jacobi.shape[0]*jacobi.shape[1]*jacobi.shape[2]))
            # lmk_dist_large_mean1s.append(lmk_dist_large_mean1)
            # lmk_dist_large_mean2s.append(lmk_dist_large_mean2)
            # lmk_dist_large_mean3s.append(lmk_dist_large_mean3)
            # lmk_dist_large_mean4s.append(lmk_dist_large_mean4)
            # lmk_dist_large_mean5s.append(lmk_dist_large_mean5)
            
            # if count==46:
            # pdb.set_trace()
            
            # ##########################################plot the warped images with the warped lmk
            # pred, _,_,_ = self.network(img2, img1)
            # flow = self.upsampler(pred[-1])
            # if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                # flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            # warp = self.reconstruction(img1, flow)
            # im1=self.appendimages(warp[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # plt.scatter(warped_lmk[0,:,1].asnumpy(),warped_lmk[0,:,0].asnumpy(), color='red', alpha=0.2)
            # plt.scatter(lmk2[0,:,1].asnumpy()+img1.shape[3],lmk2[0,:,0].asnumpy(), color='red', alpha=0.2)
            # # plt.plot([warped_lmk[0,:,1].asnumpy()[0],lmk2[0,:,1].asnumpy()[0]+512],[warped_lmk[0,:,0].asnumpy()[0],lmk2[0,:,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean*dist_weight).asnumpy()[0]))
            # plt.savefig('/ssd1/wxy/association/Maskflownet_association_1024/images/rebuttle_S_SFG/'+str(count)+'_'+str((dist_loss_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            # # pdb.set_trace()
            # # if count==78:
                # # pdb.set_trace()
            
        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warp,flows
        return dist_loss_mean.asnumpy()*dist_weight,dist_loss_median.asnumpy()*dist_weight,dist_loss_75th.asnumpy()*dist_weight,lmk_dist_large_mean1.asnumpy()*dist_weight,\
                lmk_dist_large_mean2.asnumpy()*dist_weight,lmk_dist_large_mean3.asnumpy()*dist_weight,lmk_dist_large_mean4.asnumpy()*dist_weight,lmk_dist_large_mean5.asnumpy()*dist_weight,\
            np.mean(jacobi),np.std(jacobi),np.sum(jacobi<0)/(jacobi.shape[0]*jacobi.shape[1]*jacobi.shape[2]),lmk_dist_ori.asnumpy()#np.median(results_median)
    
    
    def validate_for_4_indicators(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count):
        raws = []
        dist_mean = []
        dist_median = []
        # nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
        
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1, lmk2 in zip(img1s, img2s, lmk1s, lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            # pdb.set_trace()
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean())
            # pdb.set_trace()
            dist_loss_mean, dist_loss_median,warped_lmk, lmk2new = self.landmark_dist_for_4_indicators(lmk1, lmk2, flows)
            dist_mean.append(dist_loss_mean.asnumpy()*dist_weight)
            dist_median.append(dist_loss_median.asnumpy()*dist_weight)
            
            # ##########################################plot the warped images with the warped lmk
            # pred, _,_,_ = self.network(img2, img1)
            # flow = self.upsampler(pred[-1])
            # if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                # flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            # warp = self.reconstruction(img1, flow)
            # im1=self.appendimages(warp[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # for i in range (6):
                # plt.plot([warped_lmk[0,i,1].asnumpy()[0],lmk2[0,i,1].asnumpy()[0]+512],[warped_lmk[0,i,0].asnumpy()[0],lmk2[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean*dist_weight).asnumpy()[0]))
            # plt.savefig('/data/wxy/association/Association/images/validate_analyse_7caJan21_5760/'+str(count)+'_'+str((dist_loss_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            # # if count==78:
                # # pdb.set_trace()
            
        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warp,flows
        rawmean = []
        for raw in raws:
            raw = raw.asnumpy()
            rawmean.append(raw)
        return dist_mean,dist_median#np.median(results_median)
    
    
    def validate_for_90th_percentile(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count):
        raws = []
        dist_mean = []
        dist_median = []
        # nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
        
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1, lmk2 in zip(img1s, img2s, lmk1s, lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            shape = img1.shape
            pred, _,_,_ = self.network(img1, img2)
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean())
            
            
            dist_loss_mean, dist_loss_median,warped_lmk, lmk2new = self.landmark_dist_for_90th_percentile(lmk1, lmk2, flows)
            dist_mean.append(dist_loss_mean.asnumpy()*dist_weight)
            dist_median.append(dist_loss_median.asnumpy()*dist_weight)
            
            # ##########################################plot the warped images with the warped lmk
            # pred, _,_,_ = self.network(img2, img1)
            # flow = self.upsampler(pred[-1])
            # if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                # flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            # warp = self.reconstruction(img1, flow)
            # im1=self.appendimages(warp[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # for i in range (6):
                # plt.plot([warped_lmk[0,i,1].asnumpy()[0],lmk2[0,i,1].asnumpy()[0]+512],[warped_lmk[0,i,0].asnumpy()[0],lmk2[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean*dist_weight).asnumpy()[0]))
            # plt.savefig('/data/wxy/association/Association/images/validate_analyse_7caJan21_5760/'+str(count)+'_'+str((dist_loss_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            # # if count==78:
                # # pdb.set_trace()
            
        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warp,flows
        rawmean = []
        for raw in raws:
            raw = raw.asnumpy()
            rawmean.append(raw)
        return dist_mean,dist_median#np.median(results_median)
    
    
    def validate_for_large_displacement(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count):
        raws = []
        dist_mean = []
        dist_median = []
        # nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
        
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1, lmk2 in zip(img1s, img2s, lmk1s, lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            # pdb.set_trace()
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean())
            # pdb.set_trace()
            dist_loss_mean, dist_loss_median,warped_lmk, lmk2new = self.landmark_dist_for_large_displacement(lmk1, lmk2, flows)
            dist_mean.append(dist_loss_mean*dist_weight)
            dist_median.append(dist_loss_median*dist_weight)
            
        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warp,flows
        rawmean = []
        for raw in raws:
            raw = raw.asnumpy()
            rawmean.append(raw)
        distmean = []
        distmesian = []
        for distm in dist_mean:
            distm = distm.asnumpy()
            distmean.append(distm)
        for distm in dist_median:
            distm = distm.asnumpy()
            distmesian.append(distm)
        return distmean,distmesian#np.median(results_median)
    
    
    
    def validate_for_submit(self,dist_weight,img1s,img2s,lmk1s,count,checkpoint):
        raws = []
        dist_mean = []
        dist_median = []
        # savepath3="/data/wxy/association/Maskflownet_association_1024/kps_for_submit/"+checkpoint+'_without_dist/'
        savepath3="/data/wxy/association/Maskflownet_association_1024/kps_for_submit/"+'refine_without_dist/'
        if not os.path.exists(savepath3):
            os.mkdir(savepath3)
        img1s, img2s, lmk1s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s))
        for img1, img2, lmk1 in zip(img1s, img2s, lmk1s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            
            flows = []
            flows.append(flow)
            warped_lmk= self.landmark_dist_for_submit(lmk1, flows)
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1)) > 0.5#####添加掩膜，去除lmk中的0值
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk1_valid=lmk1[0,valid_index.squeeze()]
            warped_lmk_valid=warped_lmk[0,valid_index.squeeze()]

            pred2, _,_,_ = self.network(img2, img1)
            flow2 = self.upsampler(pred2[-1])
            if shape[2] != flow2.shape[2] or shape[3] != flow2.shape[3]:
                flow2 = nd.contrib.BilinearResize2D(flow2, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow2.shape[d] for d in (2, 3)], ctx=flow2.context).reshape((1, 2, 1, 1))
            warp2 = self.reconstruction(img1, flow2)
            # plt.figure()
            # im1=self.appendimages(img1[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.imshow(im1)
            # for i in range (lmk1_valid.shape[0]):
                # plt.plot([lmk1_valid[i,1].asnumpy()[0],warped_lmk_valid[i,1].asnumpy()[0]+1024],[lmk1_valid[i,0].asnumpy()[0],warped_lmk_valid[i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.savefig(savepath3+str(count)+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            skimage.io.imsave(savepath3+str(count)+'_1.jpg',((img1+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            skimage.io.imsave(savepath3+str(count)+'_2.jpg',((img2+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            skimage.io.imsave(savepath3+str(count)+'_1_warpped.jpg',((warp2+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            name = ['X', 'Y']
            warped_lmk=warped_lmk.asnumpy().squeeze()
            lmk1=lmk1.asnumpy().squeeze()
            outlmk1 = pd.DataFrame(columns=name, data=warped_lmk[:,[1,0]])
            outlmk1.to_csv(savepath3+str(count)+'_1_warpped.csv')
            outlmk1 = pd.DataFrame(columns=name, data=lmk1[:,[1,0]])
            outlmk1.to_csv(savepath3+str(count)+'_1.csv')
        del img1, img2, lmk1,rgb_mean,img1s, img2s, lmk1s, pred, flow,warp,flows
        return 0
    def validate_for_submit_evaluation_time(self,dist_weight,img1s,img2s,lmk1s,count,checkpoint):
        img1s, img2s, lmk1s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s))
        for img1, img2, lmk1 in zip(img1s, img2s, lmk1s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
        return 0
    def validate_for_submit_train(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count,checkpoint):
        raws = []
        dist_mean = []
        dist_median = []
        # savepath3="/data/wxy/association/Maskflownet_association_1024/kps_for_submit/"+checkpoint+'_without_dist/'
        savepath3="/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path2/final_path/"
        if not os.path.exists(savepath3):
            os.mkdir(savepath3)
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1,lmk2 in zip(img1s, img2s, lmk1s,lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            
            flows = []
            flows.append(flow)
            lmk_mean,lmk_median,lmk_max,warped_lmk= self.landmark_dist_for_submit_train(lmk1,lmk2, flows)
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5
            valid_index=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices')
            lmk1_valid=lmk1[0,valid_index.squeeze()]
            warped_lmk_valid=warped_lmk[0,valid_index.squeeze()]
            lmk2_valid=lmk2[0,valid_index.squeeze()]
            
            pred2, _,_,_ = self.network(img2, img1)
            flow2 = self.upsampler(pred2[-1])
            if shape[2] != flow2.shape[2] or shape[3] != flow2.shape[3]:
                flow2 = nd.contrib.BilinearResize2D(flow2, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow2.shape[d] for d in (2, 3)], ctx=flow2.context).reshape((1, 2, 1, 1))
            warp2 = self.reconstruction(img1, flow2)
            # plt.figure()
            # im1=self.appendimages(img1[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.imshow(im1)
            # for i in range (lmk1_valid.shape[0]):
                # plt.plot([lmk1_valid[i,1].asnumpy()[0],warped_lmk_valid[i,1].asnumpy()[0]+1024],[lmk1_valid[i,0].asnumpy()[0],warped_lmk_valid[i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((lmk_mean*dist_weight).asnumpy()[0]))
            # plt.savefig(savepath3+str(count)+'_'+str((lmk_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            skimage.io.imsave(savepath3+str(count)+'_1.jpg',((img1+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            skimage.io.imsave(savepath3+str(count)+'_2.jpg',((img2+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            skimage.io.imsave(savepath3+str(count)+'_1_warpped.jpg',((warp2+rgb_mean)[0,0,:,:]*255).squeeze().asnumpy().astype(np.uint8))
            name = ['X', 'Y']
            warped_lmk=warped_lmk.asnumpy().squeeze()
            lmk1=lmk1.asnumpy().squeeze()
            lmk2=lmk2.asnumpy().squeeze()
            outlmk1 = pd.DataFrame(columns=name, data=warped_lmk[:,[1,0]])
            outlmk1.to_csv(savepath3+str(count)+'_1_warpped.csv')
            outlmk1 = pd.DataFrame(columns=name, data=lmk1[:,[1,0]])
            outlmk1.to_csv(savepath3+str(count)+'_1.csv')
            outlmk1 = pd.DataFrame(columns=name, data=lmk2[:,[1,0]])
            outlmk1.to_csv(savepath3+str(count)+'_2.csv')
        del img1, img2, lmk1,rgb_mean,img1s, img2s, lmk1s, pred, flow,warp,flows
        return 0
    def validate_for_ANHIR_train_evaluation(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count,checkpoint):
        dist_mean = []
        dist_median = []
        dist_max = []
        # pdb.set_trace()
        try:
            img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        except:
            img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, [self.ctx[0]]),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1,lmk2 in zip(img1s, img2s, lmk1s,lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            shape = img1.shape
            pred, _,_,_ = self.network(img1, img2)
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            pred2, _,_,_ = self.network(img2, img1)
            flow2 = self.upsampler(pred2[-1])
            if shape[2] != flow2.shape[2] or shape[3] != flow2.shape[3]:
                flow2 = nd.contrib.BilinearResize2D(flow2, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow2.shape[d] for d in (2, 3)], ctx=flow2.context).reshape((1, 2, 1, 1))
            warp2 = self.reconstruction(img1, flow2)
            flows = []
            flows.append(flow)
            dist_loss_mean, dist_loss_median, dist_loss_max,warped_lmk, lmk2new = self.landmark_dist_for_6_indicators_ANHIR_train(lmk1, lmk2, flows)
            dist_mean.append(dist_loss_mean*dist_weight)
            dist_median.append(dist_loss_median*dist_weight)
            dist_max.append(dist_loss_max*dist_weight)
        # savepath='/data/wxy/association/Maskflownet_association_1024/training_visualization/'+checkpoint+'/'
        # if not os.path.exists(savepath):
            # os.mkdir(savepath)
        # if 1:#count in [121,123]:
            # plt.figure()
            # im1=self.appendimages(warp2[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.imshow(im1)
            # for i in range (warped_lmk.shape[1]):
                # plt.plot([warped_lmk[0,i,1].asnumpy()[0],lmk2[0,i,1].asnumpy()[0]+1024],[warped_lmk[0,i,0].asnumpy()[0],lmk2[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean*dist_weight).asnumpy()[0]))
            # plt.savefig(savepath+str(count)+'_'+str((dist_loss_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
        distmean = []
        distmesian = []
        distmax = []
        for distm in dist_mean:
            distm = distm.asnumpy()
            distmean.append(distm)
        for distm in dist_median:
            distm = distm.asnumpy()
            distmesian.append(distm)
        for distm in dist_max:
            distm = distm.asnumpy()
            distmax.append(distm)
        return distmean,distmesian,distmax
    def validate_for_jacobi(self,dist_weight,img1s,img2s,lmk1s,count):
        raws = []
        dist_mean = []
        dist_median = []
        savepath3="/data/wxy/association/Maskflownet_association/flows/73bDec29/"
        if not os.path.exists(savepath3):
            os.mkdir(savepath3)
        img1s, img2s, lmk1s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s))
        for img1, img2, lmk1 in zip(img1s, img2s, lmk1s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            np.save(savepath3+str(count)+'.npy',flow.asnumpy())
        return 0
    
    
    
    
    def validate_for_image_visualization(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count,checkpoint):
        prep_path3 = "/data/wxy/association/figs_for_paper/512_color_after_affine_matrix_256da4/"
        im_temp1 = skimage.io.imread(os.path.join(prep_path3, str(count)+'_1.jpg'), as_gray=False)
        im_temp2 = skimage.io.imread(os.path.join(prep_path3, str(count)+'_2.jpg'), as_gray=False)
        im_temp1=np.expand_dims(np.transpose(im_temp1,(2,0,1)),axis=0)
        im_temp2=np.expand_dims(np.transpose(im_temp2,(2,0,1)),axis=0)
        savepath='/data/wxy/association/Maskflownet_association_1024/visualization/figs_in_paper/'+checkpoint+'_without_flow_mask/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        raws = []
        dist_mean = []
        img1s, img2s, lmk1s,lmk2s,im_temp1,im_temp2 = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s,im_temp1,im_temp2))
        for img1, img2, lmk1, lmk2,im_temp1,im_temp2 in zip(img1s, img2s, lmk1s, lmk2s,im_temp1,im_temp2):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            grid_x,grid_y=np.meshgrid(np.arange(512),np.arange(512))
            grid_flow=nd.array(np.expand_dims(np.concatenate((np.expand_dims(grid_y,0),np.expand_dims(grid_x,0)),0),0),ctx=img1.context)
            mask1=img1s[0]>0
            mask2=img2s[0]>0
            
            pred, _,_,_ = self.network(img2, img1)
            flow = self.upsampler(pred[-1])
            mask2_close = mask2[0:1,0:1,:,:]
            # mask2_close = nd.expand_dims(nd.expand_dims(nd.array(cv2.morphologyEx(mask2[0,0,:,:].asnumpy(), cv2.MORPH_CLOSE, np.ones((20,20))),ctx=flow.context),0),0)
            # mask_flow1to2=mask2_close*flow-grid_flow*(1-mask2_close)
            mask_flow1to2=flow
            # warped_img1 = self.reconstruction(img1s[0], mask_flow1to2)#+(1-mask2_close)*img1s[0]
            warped_img1 = self.reconstruction(im_temp1, mask_flow1to2)#+(1-mask2_close)*img1s[0]
            # warped_img1 = self.reconstruction(img1s[0], flow)#+(1-mask2_close)*img1s[0]
            dist_loss_mean_2to1, warped_lmk2, _ = self.landmark_dist(lmk2, lmk1, [flow])
            
            
            
            pred, _,_,_ = self.network(img1, img2)
            flow = self.upsampler(pred[-1])
            mask1_close = mask1[0:1,0:1,:,:]
            # mask1_close = nd.expand_dims(nd.expand_dims(nd.array(cv2.morphologyEx(mask1[0,0,:,:].asnumpy(), cv2.MORPH_CLOSE, np.ones((20,20))),ctx=flow.context),0),0)
            # mask_flow2to1=mask1_close*flow-grid_flow*(1-mask1_close)
            mask_flow2to1=flow
            # warped_img2 = self.reconstruction(img2s[0], mask_flow2to1)#+(1-mask1_close)*img2s[0]
            warped_img2 = self.reconstruction(im_temp2, mask_flow2to1)#+(1-mask1_close)*img2s[0]
            # warped_img2 = self.reconstruction(img2s[0], flow)#+(1-mask1_close)*img2s[0]
            dist_loss_mean1to2, warped_lmk1, _ = self.landmark_dist(lmk1, lmk2, [flow])
            
            
            # pdb.set_trace()
            
            
            # flow_numpy=flow[0,0,:,:].asnumpy()
            # y, x = np.mgrid[40/2:512:40, 40/2:512:40].reshape(2, -1).astype(int)
            # fx = flow_numpy[y, x].T / flow_numpy[y, x].max() * 40 // 2
            # fy = flow_numpy[y, x].T / flow_numpy[y, x].max() * 40 // 2
            # lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
            # lines = np.int32(lines)
            # # 创建图像并绘制
            # vis = img1[0,0,:,:].asnumpy()  #cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            # for (x1, y1), (x2, y2) in lines:
              # cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
              # cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)
            # plt.figure()
            # plt.imshow(vis)
            # plt.show()
            # pdb.set_trace()
            
            skimage.io.imsave(savepath+str(count)+'_1_warpped.jpg',np.uint8(np.transpose(warped_img1.squeeze().asnumpy(),(1,2,0))))
            skimage.io.imsave(savepath+str(count)+'_2_warpped.jpg',np.uint8(np.transpose(warped_img2.squeeze().asnumpy(),(1,2,0))))
            scio.savemat(savepath+str(count)+'.mat',{'lmk1':lmk1.squeeze().asnumpy(),'lmk2':lmk2.squeeze().asnumpy(),'warped_lmk1':warped_lmk1.squeeze().asnumpy(),\
                        'warped_lmk2':warped_lmk2.squeeze().asnumpy(),'dist12':dist_loss_mean1to2.asnumpy(),'dist21':dist_loss_mean_2to1.asnumpy()})
            
            
            
            
            # im1=self.appendimages(warped_img1[0, 0, :, :].squeeze().asnumpy(),img2s[0][0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # # for i in range (6):
                # # plt.plot([warped_lmk2[0,i,1].asnumpy()[0],lmk2[0,i,1].asnumpy()[0]+512],[warped_lmk2[0,i,0].asnumpy()[0],lmk2[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean1to2*dist_weight).asnumpy()[0]))
            # plt.savefig(savepath+str(count)+'_'+str((dist_loss_mean1to2*dist_weight).asnumpy()[0])+'_warpped_img1_withoutclose.jpg', dpi=600)
            # plt.close()
            
            
            # im1=self.appendimages(img1s[0][0, 0, :, :].squeeze().asnumpy(),warped_img2[0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # # for i in range (6):
                # # plt.plot([lmk1[0,i,1].asnumpy()[0],warped_lmk1[0,i,1].asnumpy()[0]+512],[lmk1[0,i,0].asnumpy()[0],warped_lmk1[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean_2to1*dist_weight).asnumpy()[0]))
            # plt.savefig(savepath+str(count)+'_'+str((dist_loss_mean_2to1*dist_weight).asnumpy()[0])+'_warpped_img2_withoutclose.jpg', dpi=600)
            # plt.close()
            

        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warped_img2,warped_img1
        return 0
    
    
    def validate_for_large_displacement_image_visualization(self,data,pairs,checkpoint):
        savepath='/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/visualization_'+checkpoint+'_all_evaluation/'
        # savepath='/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/visualization_large_displacement_57/'
        # savepath='/ssd1/wxy/association/Maskflownet_association_1024/rebuttle_images/large_displacement_123/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        colorpath="/ssd2/wxy/IPCG_Acrobat/Affine_transformation/images/warp_results390Jan02-2356_32444_valid_final_4096_kps_add1_0127-python-color/"
        # for j in range(0, len(pairs)):
        # for j in [ 0,  7,  9, 11, 14, 15, 17, 19, 25, 27, 30, 34, 36, 37, 40, 42, 43,45, 51, 55, 62, 65, 66, 69, 71, 72, 73, 77, 80, 82, 83, 94, 99]:
        # pdb.set_trace()
        # for j in [80]:
        for j in range(len(data)):
            batch_data = data[j:j+1]
            count=pairs[j][0].split('.')[0]
            ctx = self.ctx[ : min(len(batch_data), len(self.ctx))]
            nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split = False) for i in range(len(batch_data[0]))]
            # nd_data = [gluon.utils.split_and_load([record for record in batch_data], ctx, even_split = False)]
            # nd_data = [gluon.utils.split_and_load(record, ctx, even_split = False) for record in batch_data]
            for img1, img2, lmk1, lmk2 in zip(*nd_data):
                img1, img2 = img1 / 255.0, img2 / 255.0
                img1, img2, rgb_mean = self.centralize(img1, img2)
                pred, _,_,_ = self.network(img1, img2)
                flow = self.upsampler(pred[-1])
                warp = self.reconstruction(img2, flow)
                dist_loss_mean12, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,\
                    lmk_dist_large_mean4,lmk_dist_large_mean5,warped_lmk1to2, _,lmk_dist_ori,lmk_dist_valid = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1, lmk2, [flow])
                if j==80:
                    dist_loss_mean12_large, _,_,_,_,_,_,_,_,_, _,_,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk1[:,50:52,:], lmk2[:,50:52,:], [flow])
                if 1:#lmk_dist_large_mean5>200:
                    raw_loss = self.raw_loss_op(img1, warp)
                    valid_kps_num=lmk_dist_valid.shape[0]
                    plt.figure()
                    plt.imshow(warp[0,0,:,:].asnumpy())
                    plt.scatter(lmk1[0,:,1].asnumpy(),lmk1[0,:,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean12_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean12.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    # plt.savefig(savepath+checkpoint+'pre_'+str(count)+'_2warp1.jpg')
                    plt.savefig(savepath+'pre_'+str(count)+'_2warp1.jpg',dpi=600)
                    plt.close()
                    plt.figure()
                    plt.imshow(img2[0,0,:,:].asnumpy())
                    lmk_mask = (1 - nd.prod(lmk2[:,:,0:2] <= 0, axis=-1)) > 0.5
                    idx=nd.topk(lmk_mask, axis=1, k=int(lmk_mask.sum().asnumpy()[0]), ret_typ='indices').squeeze()
                    plt.scatter(warped_lmk1to2[0,idx,1].asnumpy(),warped_lmk1to2[0,idx,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean12_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean12.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.savefig(savepath+'pre_'+str(count)+'_warp_lmk1to2.jpg',dpi=600)
                    plt.close()
                    plt.figure()
                    plt.imshow(img1[0,0,:,:].asnumpy())
                    plt.scatter(lmk1[0,:,1].asnumpy(),lmk1[0,:,0].asnumpy(), color='red', marker='o',s=0.2)
                    plt.savefig(savepath+'pre_'+str(count)+'_img1.jpg',dpi=600)
                    plt.close()
                    
                    
                    # colorimg1=np.expand_dims(np.transpose(cv2.resize(skimage.io.imread(colorpath+str(count)+'.jpg'),(512,512)),(2,0,1)),0)
                    # colorimg2=np.expand_dims(np.transpose(cv2.resize(skimage.io.imread(colorpath+str(count).split('_')[0]+'_HE_val.jpg'),(512,512)),(2,0,1)),0)
                    colorimg1=np.expand_dims(np.transpose(skimage.io.imread(colorpath+str(count)+'.jpg'),(2,0,1)),0)
                    colorimg2=np.expand_dims(np.transpose(skimage.io.imread(colorpath+str(count).split('_')[0]+'_HE_val.jpg'),(2,0,1)),0)
                    colorimg1=nd.array(colorimg1,ctx=img1.context)
                    colorimg2=nd.array(colorimg2,ctx=img2.context)
                    # pdb.set_trace()
                    warp2to1_color = self.reconstruction(colorimg2, 8*nd.transpose(nd.array(np.expand_dims(cv2.resize(nd.transpose(flow,(2,3,0,1)).asnumpy().squeeze(),(4096,4096)),2),ctx=img1.context),(2,3,0,1)))
                    skimage.io.imsave(savepath+str(count)+'_2warp1.jpg', np.transpose(np.uint8(warp2to1_color[0,:,:,:].asnumpy()),(1,2,0)))
                    pred, _,_,_ = self.network(img2, img1)
                    flow = self.upsampler(pred[-1])
                    warp2 = self.reconstruction(img1, flow)
                    dist_loss_mean21, dist_loss_median,dist_loss_75th,dist_loss_90th,lmk_dist_large_mean1,lmk_dist_large_mean2,lmk_dist_large_mean3,lmk_dist_large_mean4,lmk_dist_large_mean5\
                                                        ,warped_lmk2to1, _,lmk_dist_ori,lmk_dist_valid= self.landmark_dist_for_4_indicators_original_ACROBAT(lmk2, lmk1, [flow])
                    if j==80:
                        dist_loss_mean21_large, _,_,_,_,_,_,_,_,_, _,_,_ = self.landmark_dist_for_4_indicators_original_ACROBAT(lmk2[:,50:52,:], lmk1[:,50:52,:], [flow])
                    raw_loss = self.raw_loss_op(img2, warp2)
                    plt.figure()
                    plt.imshow(warp2[0,0,:,:].asnumpy())
                    plt.scatter(warped_lmk1to2[0,idx,1].asnumpy(),warped_lmk1to2[0,idx,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean12_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean12.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.savefig(savepath+'_pre_'+str(count)+'_1warp2_with_warp_lmk1to2.jpg',dpi=600)
                    plt.close()
                    plt.figure()
                    plt.imshow(warp2[0,0,:,:].asnumpy())
                    plt.scatter(lmk2[0,:,1].asnumpy(),lmk2[0,:,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean21_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean21.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.savefig(savepath+'_pre_'+str(count)+'_1warp2.jpg',dpi=600)
                    plt.close()
                    plt.figure()
                    plt.imshow(img1[0,0,:,:].asnumpy())
                    plt.scatter(warped_lmk2to1[0,idx,1].asnumpy(),warped_lmk2to1[0,idx,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean21_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean21.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.savefig(savepath+'pre_'+str(count)+'_warp_lmk2to1.jpg',dpi=600)
                    plt.close()
                    # # if count==121 and dist_loss_mean21.asnumpy().mean()<0.004 and raw_loss.asnumpy().mean()>0.10 and raw_loss.asnumpy().mean()<0.11:
                        # # print('this model is good:'+checkpoint+'.params')
                    plt.figure()
                    plt.imshow(img2[0,0,:,:].asnumpy())
                    plt.scatter(lmk2[0,:,1].asnumpy(),lmk2[0,:,0].asnumpy(), color='red', marker='o', s=0.2)
                    plt.savefig(savepath+'pre_'+str(count)+'_img2.jpg',dpi=600)
                    plt.close()
                    plt.figure()
                    plt.imshow(warp[0,0,:,:].asnumpy())
                    plt.scatter(warped_lmk2to1[0,idx,1].asnumpy(),warped_lmk2to1[0,idx,0].asnumpy(), color='red', marker='o', s=0.2)
                    # plt.title(str(dist_loss_mean21_large.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.title(str(dist_loss_mean21.asnumpy().mean())+'_'+str(raw_loss.asnumpy().mean()))
                    plt.savefig(savepath+'_pre_'+str(count)+'_2warp1_with_warp_lmk2to1.jpg',dpi=600)
                    plt.close()
                    warp1to2_color = self.reconstruction(colorimg1, 8*nd.transpose(nd.array(np.expand_dims(cv2.resize(nd.transpose(flow,(2,3,0,1)).asnumpy().squeeze(),(4096,4096)),2),ctx=img1.context),(2,3,0,1)))
                    skimage.io.imsave(savepath+str(count)+'_1warp2.jpg', np.transpose(np.uint8(warp1to2_color[0,:,:,:].asnumpy()),(1,2,0)))
                    scio.savemat(savepath+str(count)+'.mat',{'lmk1':lmk1[0,:,:].asnumpy(),'lmk2':lmk2[0,:,:].asnumpy(),'warped_lmk1to2':warped_lmk1to2[0,:,:].asnumpy(),'warped_lmk2to1':warped_lmk2to1[0,:,:].asnumpy(),'dist12':dist_loss_mean12.asnumpy().mean(),'dist21':dist_loss_mean21.asnumpy().mean()})
        return 0
    
    
    def validate(self,dist_weight,img1s,img2s,lmk1s,lmk2s,count):
        raws = []
        dist_mean = []
        # nd_data = [gluon.utils.split_and_load([record[i] for record in batch_data], ctx, even_split=False) for i in range(len(batch_data[0]))]
        
        img1s, img2s, lmk1s,lmk2s = map(lambda x: gluon.utils.split_and_load(x, self.ctx),(img1s, img2s,lmk1s,lmk2s))
        for img1, img2, lmk1, lmk2 in zip(img1s, img2s, lmk1s, lmk2s):#zip(*nd_data):
            img1, img2 = img1 / 255.0, img2 / 255.0
            img1, img2, rgb_mean = self.centralize(img1, img2)
            pred, _,_,_ = self.network(img1, img2)
            shape = img1.shape
            flow = self.upsampler(pred[-1])
            if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            warp = self.reconstruction(img2, flow)
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean())
            # pdb.set_trace()
            dist_loss_mean, warped_lmk, lmk2new = self.landmark_dist(lmk1, lmk2, flows)
            dist_mean.append(dist_loss_mean*dist_weight)
            
            # ##########################################plot the warped images with the warped lmk
            # pred, _,_,_ = self.network(img2, img1)
            # flow = self.upsampler(pred[-1])
            # if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                # flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
            # warp = self.reconstruction(img1, flow)
            # im1=self.appendimages(warp[0, 0, :, :].squeeze().asnumpy(),img2[0, 0, :, :].squeeze().asnumpy())
            # plt.figure()
            # plt.imshow(im1)
            # for i in range (6):
                # plt.plot([warped_lmk[0,i,1].asnumpy()[0],lmk2[0,i,1].asnumpy()[0]+512],[warped_lmk[0,i,0].asnumpy()[0],lmk2[0,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
            # plt.title(str((dist_loss_mean*dist_weight).asnumpy()[0]))
            # plt.savefig('/data/wxy/association/Association/images/validate_analyse_7caJan21_5760/'+str(count)+'_'+str((dist_loss_mean*dist_weight).asnumpy()[0])+'.jpg', dpi=600)
            # plt.close()###############validate_analyse_91eAug30    7caJan21      aa8Aug29
            # # if count==78:
                # # pdb.set_trace()
            
        del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, flow,warp,flows
        rawmean = []
        for raw in raws:
            raw = raw.asnumpy()
            rawmean.append(raw)
        distmean = []
        for distm in dist_mean:
            distm = distm.asnumpy()
            distmean.append(distm)
        return distmean#np.median(results_median)

    

    def predict(self, img1, img2, batch_size, resize = None):
        ''' predict the whole dataset
        '''
        size = len(img1)
        bs = batch_size
        for j in range(0, size, bs):
            batch_img1 = img1[j: j + bs]
            batch_img2 = img2[j: j + bs]

            batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
            batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))

            batch_flow = []
            batch_occ_mask = []
            batch_warped = []

            ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
            nd_img1, nd_img2 = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), (batch_img1, batch_img2))
            for img1s, img2s in zip(nd_img1, nd_img2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                flow, occ_mask, warped, _ = self.do_batch(img1s, img2s, resize = resize)
                batch_flow.append(flow)
                batch_occ_mask.append(occ_mask)
                batch_warped.append(warped)
            flow = np.concatenate([x.asnumpy() for x in batch_flow])
            occ_mask = np.concatenate([x.asnumpy() for x in batch_occ_mask])
            warped = np.concatenate([x.asnumpy() for x in batch_warped])
            
            flow = np.transpose(flow, (0, 2, 3, 1))
            flow = np.flip(flow, axis = -1)
            occ_mask = np.transpose(occ_mask, (0, 2, 3, 1))
            warped = np.transpose(warped, (0, 2, 3, 1))
            for k in range(len(flow)):
                yield flow[k], occ_mask[k], warped[k]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def generate_dense(self, img1, img2,name_nums):
        losses = []
        reg_losses = []
        raw_losses = []
        dist_losses = []
        dist_losses2 = []
        dist_losses3 = []
        dist_losses4 = []
        batch_size = img1.shape[0]
        # pdb.set_trace()
        img1, img2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s in zip(img1, img2):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                img1s_2, img2s_2, _ = self.centralize(img1s, img2s)
                shape = img1s_2.shape
                # savepath='/data/wxy/association/Maskflownet_association/dense/a0cAug30_3356/'
                savepath='/data/wxy/association/Maskflownet_association/dense/a0cAug30_3356_5roi_16rotate/'
                desc1s=np.zeros([1,196,shape[2],shape[3]])
                desc2s=np.zeros([1,196,shape[2],shape[3]])
                for i in range(shape[2]):
                    for j in range (shape[3]):
                        kp1_x1,kp1_y1=j,i
                        patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                        patch_img2s=nd.concat(nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                                nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                        if patch_img1s.sum()==0 or patch_img2s.sum()==0:
                            continue
                        else:
                            # pdb.set_trace()
                            patch_img1s_2, patch_img2s_2, _ = self.centralize(patch_img1s, patch_img2s)
                            _, c1s, c2s,_= self.network(patch_img1s_2, patch_img2s_2)#5,196,1,1
                            _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s_2,45), mx.image.imrotate(patch_img2s_2,45))#5,196,1,1
                            _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s_2,90), mx.image.imrotate(patch_img2s_2,90))#5,196,1,1
                            _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s_2,135), mx.image.imrotate(patch_img2s_2,135))#5,196,1,1
                            _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s_2,180), mx.image.imrotate(patch_img2s_2,180))#5,196,1,1
                            _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s_2,225), mx.image.imrotate(patch_img2s_2,225))#5,196,1,1
                            _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s_2,270), mx.image.imrotate(patch_img2s_2,270))#5,196,1,1
                            _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s_2,315), mx.image.imrotate(patch_img2s_2,315))#5,196,1,1
                            _,c1s_9, c2s_9,_= self.network(mx.image.imrotate(patch_img1s_2,22.5), mx.image.imrotate(patch_img2s_2,22.5))#5,196,1,1
                            _,c1s_10, c2s_10,_= self.network(mx.image.imrotate(patch_img1s_2,67.5), mx.image.imrotate(patch_img2s_2,67.5))#5,196,1,1
                            _,c1s_11, c2s_11,_= self.network(mx.image.imrotate(patch_img1s_2,112.5), mx.image.imrotate(patch_img2s_2,112.5))#5,196,1,1
                            _,c1s_12, c2s_12,_= self.network(mx.image.imrotate(patch_img1s_2,157.5), mx.image.imrotate(patch_img2s_2,157.5))#5,196,1,1
                            _,c1s_13, c2s_13,_= self.network(mx.image.imrotate(patch_img1s_2,202.5), mx.image.imrotate(patch_img2s_2,202.5))#5,196,1,1
                            _,c1s_14, c2s_14,_= self.network(mx.image.imrotate(patch_img1s_2,247.5), mx.image.imrotate(patch_img2s_2,247.5))#5,196,1,1
                            _,c1s_15, c2s_15,_=self.network(mx.image.imrotate(patch_img1s_2,292.5), mx.image.imrotate(patch_img2s_2,292.5))#5,196,1,1
                            _,c1s_16, c2s_16,_=self.network(mx.image.imrotate(patch_img1s_2,337.5), mx.image.imrotate(patch_img2s_2,337.5))#5,196,1,1
                            desc1s[:,:,i,j]=nd.concat(c1s.mean(0),c1s_2.mean(0),c1s_3.mean(0),c1s_4.mean(0),c1s_5.mean(0),c1s_6.mean(0),c1s_7.mean(0),c1s_8.mean(0),c1s_9.mean(0),c1s_10.mean(0),c1s_11.mean(0),c1s_12.mean(0),c1s_13.mean(0),c1s_14.mean(0),c1s_15.mean(0),c1s_16.mean(0),dim=1).mean(1).squeeze().asnumpy()
                            desc2s[:,:,i,j]=nd.concat(c2s.mean(0),c2s_2.mean(0),c2s_3.mean(0),c2s_4.mean(0),c2s_5.mean(0),c2s_6.mean(0),c2s_7.mean(0),c2s_8.mean(0),c2s_9.mean(0),c2s_10.mean(0),c2s_11.mean(0),c2s_12.mean(0),c2s_13.mean(0),c2s_14.mean(0),c2s_15.mean(0),c2s_16.mean(0),dim=1).mean(1).squeeze().asnumpy()
                        # print('{},{}'.format(kp1_x1,kp1_y1))
                scio.savemat(savepath+name_nums+'_1.mat',{'desc1s':desc1s})
                scio.savemat(savepath+name_nums+'_2.mat',{'desc2s':desc2s})
        return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def generate_dense(self, img1, img2,name_nums):
        # losses = []
        # reg_losses = []
        # raw_losses = []
        # dist_losses = []
        # dist_losses2 = []
        # dist_losses3 = []
        # dist_losses4 = []
        # batch_size = img1.shape[0]
        # img1, img2 = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2))
        # hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        # with autograd.record():
            # for img1s, img2s in zip(img1, img2):
                # img1s, img2s = img1s / 255.0, img2s / 255.0
                # #img1s, img2s = aug(img1s, img2s) # no only geo_aug, but also padding the image size to (64*n1)*(64*n2), gl:check and visualized the img1s and img2s
                # # img1s, img2s = color_aug(img1s, img2s) # gl, check and visualized whether this is necessary or should be deleted
                # img1s_2, img2s_2, _ = self.centralize(img1s, img2s)
                # shape = img1s_2.shape
                # savepath='/data/wxy/association/Maskflownet_association/dense/a0cAug30_3356/'
                
                # desc1s=np.zeros([1,196,shape[2],shape[3]])
                # desc2s=np.zeros([1,196,shape[2],shape[3]])
                # for i in range(shape[2]):
                    # for j in range (shape[3]):
                        # kp1_x1,kp1_y1=j,i
                        # # pdb.set_trace()
                        # patch_img1s=nd.contrib.BilinearResize2D(img1s[0:1,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64)
                        # patch_img2s=nd.contrib.BilinearResize2D(img2s[0:1,:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64)
                        # if patch_img1s.sum()==0 or patch_img2s.sum()==0:
                            # continue
                        # else:
                            # patch_img1s_2, patch_img2s_2, _ = self.centralize(patch_img1s, patch_img2s)
                            # _, c1s, c2s,_= self.network(patch_img1s_2, patch_img2s_2)#5,196,1,1
                            # # pdb.set_trace()
                            # c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                            # desc1s[:,:,i,j]=c1s
                            # desc2s[:,:,i,j]=c2s
                # # pdb.set_trace()
                # scio.savemat(savepath+name_nums+'_1.mat',{'desc1s':desc1s})
                # scio.savemat(savepath+name_nums+'_2.mat',{'desc2s':desc2s})
        # return 0
    
    
    
    
    
    def train_batch_dense(self, img1, img2, dense1s, dense2s):
        losses = []
        reg_losses = []
        raw_losses = []
        batch_size = img1.shape[0]
        img1, img2, dense1s, dense2s = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, dense1s, dense2s))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        with autograd.record():
            for img1s, img2s, dense1, dense2 in zip(img1, img2, dense1s, dense2s):
                img1s, img2s = img1s / 255.0, img2s / 255.0
                img1s, img2s, _ = self.centralize(img1s, img2s)
                pred, _, _ ,_= self.network(img1s, img2s) # this warpeds is not mean the warped image
                shape = img1s.shape
                flow = self.upsampler(pred[-1])
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                # pdb.set_trace()
                warp = self.reconstruction(dense2, flow)
                flows = []
                flows.append(flow)
                # raw loss calculation
                raw_loss = self.raw_loss_op(dense1, warp)
                # raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow)
                self.raw_weight=1
                self.reg_weight = 1#/20 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        , "reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses]))}
