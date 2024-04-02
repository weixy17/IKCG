import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
import pdb
from .MaskFlownet import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth
import copy
import skimage.io
import os
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import cv2
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
    def train_batch(self, dist_weight, img1, img2, lmk1s, lmk2s,orb1s,orb2s,steps):
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
                
                
                ######################800_whole_image
                # orb_warp=nd.zeros((shape[0],800,2), ctx=flow.context)
                orb_maskflownet=nd.zeros((shape[0],800,2), ctx=flow.context)
                orb1=nd.zeros((shape[0],800,2), ctx=flow.context)
                for k in range (shape[0]):
                    kp2, _ = orb.detectAndCompute((img2s[k,0,:,:]*255).asnumpy().astype('uint8'), None)
                    kp1, _ = orb.detectAndCompute((img1s[k,0,:,:]*255).asnumpy().astype('uint8'), None)
                    filtered_coords2s=self.addsiftkp(kp2)
                    filtered_coords1s=self.addsiftkp(kp1)
                    for m in range (orb1.shape[1]-len(filtered_coords2s)):
                        filtered_coords2s.extend([[0,0]])
                    for m in range (orb1.shape[1]-len(filtered_coords1s)):
                        filtered_coords1s.extend([[0,0]])
                
                    plt.figure()
                    plt.imshow(img2s[k,0,:,:].asnumpy())
                    for i in range (len(filtered_coords2s)):
                        plt.scatter(filtered_coords2s[i][1],filtered_coords2s[i][0],s=0.05)
                    plt.savefig('/data/wxy/association/Maskflownet_association/images/img2s_key_points/'+str(steps)+'_'+str(k)+'_warp.jpg',dpi=600)
                    plt.close()
                    orb_maskflownet[k,:,:]=nd.array(filtered_coords2s,ctx=flow.context)
                    orb1[k,:,:]=nd.array(filtered_coords1s,ctx=flow.context)
                
                # ###########################16s8
                # orb_warp=nd.zeros((shape[0],orb1.shape[1],2), ctx=flow.context)
                # for k in range (shape[0]):
                    # filtered_coords2s=[]
                    # for i in range(int(np.floor((shape[2]-search_range)/step_range))):
                        # for j in range(int(np.floor((shape[2]-search_range)/step_range))):
                            # select_roi=(warp[k,0,step_range*i:(step_range*i+search_range),step_range*j:(step_range*j+search_range)]*255).asnumpy().astype('uint8')
                            # if np.abs((select_roi-select_roi.mean())).sum()>0:
                                # kp2, _ = orb_2.detectAndCompute(select_roi, None)
                                # if kp2!=():
                                    # filtered_coords2= self.addsiftkp2(kp2,i,j)
                                    # filtered_coords2s.extend(filtered_coords2)
                    # for m in range (orb1.shape[1]-len(filtered_coords2s)):
                        # filtered_coords2s.extend([[0,0]])
                    # orb_warp[k,:,:]=nd.array(filtered_coords2s,ctx=flow.context)
                    # plt.figure()
                    # plt.imshow(warp[k,0,:,:].asnumpy())
                    # for i in range (orb1.shape[1]):
                        # plt.scatter(orb_warp[k,i,1].asnumpy(),orb_warp[k,i,0].asnumpy(),s=0.05)
                    # plt.savefig('/data/wxy/association/Maskflownet_association/images/warp_key_points/'+str(steps)+'_'+str(k)+'_warp.jpg',dpi=600)

                desc1s=np.zeros([5,1568,shape[0],orb1.shape[1]])
                desc2s=np.zeros([5,1568,shape[0],orb1.shape[1]])
                for k in range (shape[0]):
                    for i in range(orb1.shape[1]):
                        kp1_x1,kp1_y1=orb1[k,i,1].asnumpy(),orb1[k,i,0].asnumpy()
                        patch_img1s=nd.concat(nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img1s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)

                        kp1_x1,kp1_y1=orb_maskflownet[k,i,1].asnumpy(),orb_maskflownet[k,i,0].asnumpy()
                        patch_img2s=nd.concat(nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange2)),0):min(int((kp1_y1+arange2)),shape[2]),max(int((kp1_x1-arange2)),0):min(int((kp1_x1+arange2)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange3)),0):min(int((kp1_y1+arange3)),shape[2]),max(int((kp1_x1-arange3)),0):min(int((kp1_x1+arange3)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange4)),0):min(int((kp1_y1+arange4)),shape[2]),max(int((kp1_x1-arange4)),0):min(int((kp1_x1+arange4)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange5)),0):min(int((kp1_y1+arange5)),shape[2]),max(int((kp1_x1-arange5)),0):min(int((kp1_x1+arange5)),shape[2])],height=64, width=64),
                                            nd.contrib.BilinearResize2D(img2s[k:(k+1),:,max(int((kp1_y1-arange6)),0):min(int((kp1_y1+arange6)),shape[2]),max(int((kp1_x1-arange6)),0):min(int((kp1_x1+arange6)),shape[2])],height=64, width=64),dim=0)
                        
                        patch_img1s, patch_img2s, _ = self.centralize(patch_img1s, patch_img2s)
                        _, c1s, c2s,_= self.network(patch_img1s, patch_img2s)#5,196,1,1
                        _,c1s_2, c2s_2,_= self.network(mx.image.imrotate(patch_img1s,45), mx.image.imrotate(patch_img2s,45))#5,196,1,1
                        _,c1s_3, c2s_3,_= self.network(mx.image.imrotate(patch_img1s,90), mx.image.imrotate(patch_img2s,90))#5,196,1,1
                        _,c1s_4, c2s_4,_= self.network(mx.image.imrotate(patch_img1s,135), mx.image.imrotate(patch_img2s,135))#5,196,1,1
                        _,c1s_5, c2s_5,_= self.network(mx.image.imrotate(patch_img1s,180), mx.image.imrotate(patch_img2s,180))#5,196,1,1
                        _,c1s_6, c2s_6,_= self.network(mx.image.imrotate(patch_img1s,225), mx.image.imrotate(patch_img2s,225))#5,196,1,1
                        _,c1s_7, c2s_7,_= self.network(mx.image.imrotate(patch_img1s,270), mx.image.imrotate(patch_img2s,270))#5,196,1,1
                        _,c1s_8, c2s_8,_=self.network(mx.image.imrotate(patch_img1s,315), mx.image.imrotate(patch_img2s,315))#5,196,1,1
                        c1s, c2s=c1s.squeeze().asnumpy(),c2s.squeeze().asnumpy()
                        c1s_2, c2s_2=c1s_2.squeeze().asnumpy(),c2s_2.squeeze().asnumpy()
                        c1s_3, c2s_3=c1s_3.squeeze().asnumpy(),c2s_3.squeeze().asnumpy()
                        c1s_4, c2s_4=c1s_4.squeeze().asnumpy(),c2s_4.squeeze().asnumpy()
                        c1s_5, c2s_5=c1s_5.squeeze().asnumpy(),c2s_5.squeeze().asnumpy()
                        c1s_6, c2s_6=c1s_6.squeeze().asnumpy(),c2s_6.squeeze().asnumpy()
                        c1s_7, c2s_7=c1s_7.squeeze().asnumpy(),c2s_7.squeeze().asnumpy()
                        c1s_8, c2s_8=c1s_8.squeeze().asnumpy(),c2s_8.squeeze().asnumpy()
                        # pdb.set_trace()
                        desc1s[:,:,k,i]=np.concatenate((c1s,c1s_2,c1s_3,c1s_4,c1s_5,c1s_6,c1s_7,c1s_8),1)
                        desc2s[:,:,k,i]=np.concatenate((c2s,c2s_2,c2s_3,c2s_4,c2s_5,c2s_6,c2s_7,c2s_8),1)#(5,196*8,1,1)
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
                
                
                
                
                
                ################################
                desc1s=nd.array(desc1s,ctx=flow.context)#(5,196*8,N,K)
                desc2s=nd.array(desc2s,ctx=flow.context)#(5,196*8,N,K)
                normalized_desc1s = nd.transpose(desc1s/nd.norm(desc1s,ord=2,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                normalized_desc2s = nd.transpose(desc2s/nd.norm(desc2s,ord=2,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                sim_mats = nd.batch_dot(normalized_desc1s, normalized_desc2s)#(5,N,K,K)
                sim_mat_12=nd.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K,K)
                sim_mat_21=nd.swapaxes(sim_mat_12,1,2)#(N,K,K)
                # pdb.set_trace()
                ####orb1(N,K,2)    orb_warp(N,K,2)    orb_maskflownet(N,K,2)
                dis=nd.abs(nd.sum(orb1*orb1,axis=2,keepdims=True)+nd.swapaxes(nd.sum(orb_maskflownet*orb_maskflownet,axis=2,keepdims=True),1,2)-2*nd.batch_dot(orb1,nd.swapaxes(orb_maskflownet,1,2)))#N,K,K
                mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                # print(mask_zone.sum().asnumpy()[0])
                mid_indices, mask12 = self.associate(sim_mat_12*mask_zone,orb_maskflownet)#(N,K,1)
                max_indices,mask21 = self.associate(sim_mat_21*(mask_zone.transpose((0,2,1))),orb1)#(N,K,1)
                
                
                
                # ################################
                # # pdb.set_trace()
                # normalized_desc1s = np.transpose(desc1s/np.linalg.norm(desc1s,axis=1,keepdims=True),(0,2,3,1))#(5,N,K,196*8)
                # normalized_desc2s = np.transpose(desc2s/np.linalg.norm(desc2s,axis=1,keepdims=True),(0,2,1,3))#(5,N,196*8,K)
                # sim_mats = np.matmul(normalized_desc1s, normalized_desc2s)#(5,N,K,K)
                
                # sim_mat_12=np.squeeze(0.2*sim_mats[0:1,:,:,:]+0.2*sim_mats[1:2,:,:,:]+0.2*sim_mats[2:3,:,:,:]+0.2*sim_mats[3:4,:,:,:]+0.2*sim_mats[4:5,:,:,:],axis=0)#(N,K,K)
                # sim_mat_21=sim_mat_12.transpose((0,2,1))#(N,K,K)
                # ####orb1(N,K,2)    orb_warp(N,K,2)    
                # dis=np.absolute(np.sum(orb1.asnumpy()*orb1.asnumpy(),axis=2,keepdims=True)+np.transpose(np.sum(orb_warp.asnumpy()*orb_warp.asnumpy(),axis=2,keepdims=True),(0,2,1))-2*np.matmul(orb1.asnumpy(),np.transpose(orb_warp.asnumpy(),(0,2,1))))#N,K,K
                # mask_zone=dis<(0.028**2)*(shape[2]**2)*2#0.015 0.04#(N,K,K)
                # print(mask_zone.sum())
                # mid_indices, mask12 = self.associate(nd.array(sim_mat_12*mask_zone,ctx=flow.context),orb_warp)#(N,K,1)
                # max_indices, mask21 = self.associate(nd.array(sim_mat_21*(mask_zone.transpose((0,2,1))),ctx=flow.context),orb1)#(N,K,1)
                
                
                # pdb.set_trace()
                # print(mask12.sum().asnumpy())
                # print(mask21.sum().asnumpy())
                # pdb.set_trace()
                indices = nd.diag(nd.gather_nd(nd.swapaxes((max_indices+1)*(mask21*2-1)-1,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1)).squeeze(2)##N,K
                
                indices_2 = indices*mask12.squeeze(2)#N,K
                
                mask=nd.broadcast_equal(indices,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=flow.context),0))##N,K
                mask2=nd.broadcast_equal(indices_2,nd.expand_dims(nd.array(np.arange(orb1.shape[1]),ctx=flow.context),0))##N,K
                mask=mask*mask2==1##N,K
                mid_orb_warp=nd.diag(nd.gather_nd(nd.swapaxes(orb_maskflownet,0,1),nd.transpose(mid_indices,axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(N,K,2)
                coor1=nd.stop_gradient(orb1*mask.expand_dims(2))
                coor2=nd.stop_gradient(mid_orb_warp*mask.expand_dims(2))
                # for k in range (shape[0]):
                    # im1=self.appendimages(img1s[k, 0, :, :].asnumpy(),warp[k, 0, :, :].asnumpy())
                    # plt.figure()
                    # plt.imshow(im1)
                    # for i in range (orb1.shape[1]):
                        # plt.plot([coor1[k,i,1].asnumpy()[0],coor2[k,i,1].asnumpy()[0]+shape[2]],[coor1[k,i,0].asnumpy()[0],coor2[k,i,0].asnumpy()[0]], '#FF0033',linewidth=0.5)
                    # plt.savefig('/data/wxy/association/Maskflownet_association/images/warp_key_points/'+str(steps)+'_'+str(k)+'_pairs.jpg', dpi=600)
                    # plt.close()
                '''
                # ##########把mask12==1的索引找出来，乱序排列为index
                # index=nd.topk(mask12*1.0, axis=1, k=int(mask12.sum().asnumpy()), ret_typ='indices')#N,orb_nums,1
                # mid_indices_valid=nd.pick(mid_indices,index.squeeze(2),1)#(N,orb_nums)
                # ##########把mask21==1的索引找出来，乱序排列为index2
                # index2=nd.topk(mask21*1.0, axis=1, k=int(mask21.sum().asnumpy()), ret_typ='indices')#N,orb_nums,1
                # max_indices_valid=nd.pick(max_indices,1,index2.squeeze(2),1)#(N,orb_nums)
                # ################因为有0的存在，所以会存在误判
                # indices = nd.pick((max_indices+1)*(mask21*2-1)-1, mid_indices_valid, 1)#*mask21   #N,orb_nums
                # indices2 = nd.pick((mid_indices+1)*(mask12*2-1)-1, max_indices_valid, 1)#*mask21   #N,orb_nums
                # mask=(index.squeeze(2)==indices)#N,orb_nums
                # mask2=(index2.squeeze(2)==indices2)#N,orb_nums
                
                # index=nd.topk(mask*1.0, axis=1, k=int(mask12.sum().asnumpy()), ret_typ='indices')#N,orb_nums,1
                # mid_indices_valid=nd.pick(mid_indices,index.squeeze(2),1)#(N,orb_nums)
                
                
                # index_valid=torch.masked_select(index,mask)#[K]
                # index_valid2=torch.masked_select(index2,mask2)#[K]
                # print(mask.sum().detach().cpu().numpy())
                # print(mask2.sum().detach().cpu().numpy())
                # if (mask.sum().detach().cpu().numpy()!=mask2.sum().detach().cpu().numpy()):
                    # pdb.set_trace()'''
                
                
                flows = []
                flows.append(flow)
                dist_loss, _, _ = self.landmark_dist(lmk1, lmk2, flows)
                dist_loss2, _, _ = self.landmark_dist(coor1, coor2, flows)
                # dist_loss2 = nd.sum(nd.sqrt(nd.sum(nd.square(coor1 - coor2), axis=-1) + 1e-5), axis=-1)/ (shape[2]*1.414) /(nd.sum(mask,axis=1)+1e-5)*(np.sum(mask, axis=1)!=0)
                # raw loss calculation
                # raw_loss = self.raw_loss_op(sift1s, warp)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                self.raw_weight=1
                self.reg_weight = 1/20 #0.2
                dist_weight = 20#200#0#50 # 10#1 #50 #100 #200
                dist_weight2=5
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight + dist_loss*dist_weight +dist_loss2*dist_weight2 # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                #loss = dist_loss * dist_weight  # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
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
    
    
    
    def associate(self, sim_mat,fkp):
        #############sim_mat:(N,K,K)  fkp(N,K,2)    
        # pdb.set_trace()
        indice = nd.topk(sim_mat, axis=2, k=2, ret_typ='indices')#(N,K,2)
        fkp_ref=nd.diag(nd.gather_nd(nd.swapaxes(fkp,0,1),nd.transpose(nd.slice_axis(indice,axis=2,begin=0,end=1),axes=(2,0,1))),axis1=0,axis2=2).transpose((2,0,1))#(K,2,N)#(N,K,N,2)
        d_temp=nd.abs(nd.sum(fkp_ref*fkp_ref,axis=2,keepdims=True)+nd.sum(fkp*fkp,axis=2,keepdims=True).transpose((0,2,1))-2*nd.batch_dot(fkp_ref,fkp.transpose((0,2,1))))#N,K,K
        mask_nms1=d_temp>=(0.004**2)*(512**2)*2#0.005
        mask_nms2=d_temp==0
        mask_nms=((mask_nms1+mask_nms2)>=1)
        sim = nd.topk(sim_mat*mask_nms,axis=2, k=2, ret_typ='value')
        mask1=nd.broadcast_lesser(nd.slice_axis(sim,axis=2,begin=1,end=2),(nd.slice_axis(sim,axis=2,begin=0,end=1)*0.98))
        mask2=nd.slice_axis(sim,axis=2,begin=0,end=1)>0.85#.85#.9
        mask=mask1*mask2==1
        return indice[:,:,0:1],mask#(N,K,1)
    
    
    
    
    
    def landmark_dist(self, lmk1, lmk2, flows):
        if np.shape(lmk2)[0] > 0:
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)#[512,512]
            lmk_mask = (1 - nd.prod(lmk1 == 0, axis=-1))*(1 - nd.prod(lmk2 == 0, axis=-1)) > 0.5
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
