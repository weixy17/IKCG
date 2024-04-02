import mxnet as mx
import numpy as np
from numpy import *
from mxnet import nd, gluon, autograd
import pdb
from .MaskFlownet import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth
import copy
import os
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import cv2
import time
from .association import AssociationLoss
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import scipy.io as scio
import sys 
sys.path.append("..") 
import logger
import time

def build_network(name):
    return eval(name)

def get_coords(img):
    shape = img.shape
    range_x = nd.arange(shape[2], ctx = img.context).reshape(shape = (1, 1, -1, 1)).tile(reps = (shape[0], 1, 1, shape[3]))
    range_y = nd.arange(shape[3], ctx = img.context).reshape(shape = (1, 1, 1, -1)).tile(reps = (shape[0], 1, shape[2], 1))
    return nd.concat(range_x, range_y, dim = 1)


class PipelineFlownet:
    _lr = None

    def __init__(self, ctx, config):
        self.ctx = ctx
        self.network = build_network(getattr(config.network, 'class').get('MaskFlownet'))(config=config)
        self.network.hybridize()
        self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4,'wd':config.optimizer.wd.value})
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

        self.multiscale_weights = config.network.mw.get([.005, .01, .02, .08, .32])
        if len(self.multiscale_weights) != 5:
            self.multiscale_weights = [.005, .01, .02, .08, .32]
        self.multiscale_association_weights=[.005, .01, .02, .08, .32,.565]
        self.multiscale_epe = MultiscaleEpe(
            scales = self.strides, weights = self.multiscale_weights, match = 'upsampling',
            eps = 1e-8, q = config.optimizer.q.get(None))
        self.multiscale_epe.hybridize()

        self.reconstruction = Reconstruction2DSmooth(3)
        self.reconstruction.hybridize()
        
        self.max_displacement=5
        self.stride1=1
        self.association = AssociationLoss(metric='cos', spagg=True, spagg_alpha=0.5, asso_topk=1,
            print_info=False,max_displacement=self.max_displacement,stride1=self.stride1)
        self.association.hybridize()
        self.association_kl = AssociationLoss('kl')
        self.association_kl.hybridize()

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
        # rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
        # return img1 - rgb_mean, img2 - rgb_mean, rgb_mean
        rgb_min = nd.concat(img1, img2, dim = 2).min(axis = (2, 3)).reshape((-2, 1, 1))
        rgb_max = nd.concat(img1, img2, dim = 2).max(axis = (2, 3)).reshape((-2, 1, 1))
        rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
        # pdb.set_trace()
        return (img1 - rgb_min)/(rgb_max-rgb_min)+1, (img2 - rgb_min)/(rgb_max-rgb_min)+1, rgb_min



    def train_batch_association(self, img1, img2,kp1,kp2,dist_weight,step):
        losses = []
        reg_losses = []
        raw_losses = []
        dist1_losses = []
        dist2_losses = []
        dist3_losses = []
        img_association_losses= []
        feature_association_losses= []
        img_dist1_losses= []
        img_dist2_losses= []
        feature_dist1_losses= []
        feature_dist2_losses= []
        association_losses=[]
        batch_size = img1.shape[0]
        #threshold=[0.8]
        img1, img2,kp1,kp2= map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,kp1,kp2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        count=0
        with autograd.record():
            for img1s, img2s,kp1,kp2 in zip(img1, img2,kp1,kp2):
                
                
                img1s, img2s = img1s/ 255.0+0.5, img2s/ 255.0+0.5
                
                # img1s, img2s, _ = self.centralize(img1s0, img2s0)
                shape = img1s.shape
                pred, c1s,c2s,_ = self.network(img1s, img2s) # this warpeds is not mean the warped image
                
                flow = self.upsampler(pred[-1])
                
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                # dist_loss, warped_lmk, _ = self.landmark_dist(lmk1, lmk2, [flow])
                


                
                association_losses,dist1s=0,0
                coor1s,coor2s,masks=[],[],[]
                num=0
                for i in range (6):
                    f1=c1s[i]##N,C,H,W
                    f2=c2s[i]
                    fkp1=nd.swapaxes(nd.expand_dims((kp1/(2**(i+1))),3),1,2)#(N, 2, 626, 1)
                    fkp2=nd.swapaxes(nd.expand_dims((kp2/(2**(i+1))),3),1,2)#(N, 2, 626, 1)
                    desc1=nd.BilinearSampler(f1, fkp1)#(N, C, 626, 1)
                    desc2=nd.BilinearSampler(f2, fkp2)#(N, C, 626, 1)
                    
                    association_loss,dist1,coor1,coor2,mask=self.Association(desc1, desc2,fkp1,fkp2)
                    coor1s.append(coor1)
                    coor2s.append(coor2)
                    masks.append(mask)
                    num=num+mask.sum().asnumpy()
                    association_losses,dist1s=association_losses+association_loss*self.multiscale_association_weights[5-i],dist1s+dist1*self.multiscale_association_weights[5-i]
                if num>0:
                    img1ss=nd.squeeze(img1s[0,0,:,:]).asnumpy()
                    img2ss=nd.squeeze(img2s[0,0,:,:]).asnumpy()
                    fig=plt.figure()
                    ax1 = fig.add_subplot(2, 3, 1)
                    ax2 = fig.add_subplot(2, 3, 2)
                    ax3 = fig.add_subplot(2, 3, 3)
                    ax4 = fig.add_subplot(2, 3, 4)
                    ax5 = fig.add_subplot(2, 3, 5)
                    ax6 = fig.add_subplot(2, 3, 6)
                    ax=[ax1,ax2,ax3,ax4,ax5,ax6]
                    im1=self.appendimages(img1ss,img2ss)
                    for i in range (6):
                        ax[i].imshow(im1)
                        num_slice=masks[i].sum().asnumpy()
                        ax[i].set_title('i-j*,num='+str(num_slice))
                        if num_slice>0:
                            index=nd.topk(mask,axis=1,k=min(int(num_slice),800))#N,800,1
                            coor1,coor2=coor1*(2**(i+1)),coor2*(2**(i+1))
                            for i1 in range(min(int(num_slice),800)):
                                ax[i].plot([coor1[0,index[i1],1],coor2[0,index[i1],1]],[coor1[0,index[i1],0],coor2[0,index[i1],0]], self.randomcolor(),linewidth=0.1)
                    if not os.path.exists('/data3/wxy/Association/images/images_orb/'):
                        os.mkdir('/data3/wxy/Association/images/images_orb/')
                    plt.savefig('/data3/wxy/Association/images/images_orb/'+str(step)+'_'+str(count)+'.jpg', dpi=600)
                    plt.close(fig)



                association_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight+association_weight*(dist1s)+association_losses#*1000#+(feature_dist1)*0.1+(feature_association_loss)#
                # loss=nd.abs(loss-0.18)+0.18
                #loss = dist_loss * dist_weight  # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                img_association_losses.append(association_losses)
                # feature_association_losses.append(feature_association_loss)
                img_dist1_losses.append(dist1s)
                # feature_dist1_losses.append(feature_dist1)
                del img1s, img2s#,feature_dist1,feature_association_loss#img_dist1,img_association_loss,
                count=count+1
        del img1, img2
        for loss in losses:
            loss.backward()
        self.trainer.step(batch_size)
        return {"loss": np.mean(np.concatenate([loss.asnumpy() for loss in losses])), "raw loss": np.mean(np.concatenate([loss.asnumpy() for loss in raw_losses]))
        ,"reg loss": np.mean(np.concatenate([loss.asnumpy() for loss in reg_losses])), "img dist1 loss": np.mean(np.concatenate([loss.asnumpy() for loss in img_dist1_losses]))
        ,"img association loss": np.mean(np.concatenate([loss.asnumpy() for loss in img_association_losses]))}



    def corr(self,x1,x2,N,H,W,kernel_size = 15, D = 10,stride1=2,stride2=2):
        H_new=int(np.ceil((H-2*(D+(kernel_size-1)/2))/stride1))
        W_new=int(np.ceil((W-2*(D+(kernel_size-1)/2))/stride1))
        radius=int(D/stride2)
        width=2*radius+1
        top_channel=width*width
        for t in range (H_new):
            for s in range(W_new):
                h0=int((D+t*stride1+(kernel_size-1)/2))
                w0=int(D+(kernel_size-1)/2+s*stride1)
                # img1=nd.slice(x1,begin=(None,None,int(h0-(kernel_size-1)/2),int(w0-(kernel_size-1)/2)),end=(None,None,int(h0+1+(kernel_size-1)/2),int(w0+1+(kernel_size-1)/2)))
                # ncc=nd.expand_dims(nd.expand_dims(nd.reshape(nd.sum(img1*img1,axis=[1,2,3]),(-1,1)),2),2)#sim_mat[:,1,1,1]
                count=-1
                for p in range(-radius,radius+1):
                    for q in range(-radius,radius+1):
                        count=count+1
                        img2=nd.slice(x2,begin=(None,None,int(h0-(kernel_size-1)/2)+p*stride2,int(w0-(kernel_size-1)/2)+q*stride2),end=(None,None,int(h0+1+(kernel_size-1)/2)+p*stride2,int(w0+1+(kernel_size-1)/2)+q*stride2))
                        ncc2=nd.expand_dims(nd.expand_dims(nd.reshape(nd.sum(img2*img2,axis=[1,2,3]),(-1,1)),2),2)#sim_mat[:,1,1,1]
                        if count==0:
                            sim_mat_2_count=ncc2
                        else:
                            sim_mat_2_count=nd.concat(sim_mat_2_count,ncc2,dim=1)
                if s==0:
                    # sim_mat_s=ncc
                    sim_mat_s2=sim_mat_2_count
                else:
                    # sim_mat_s=nd.concat(sim_mat_s,ncc,dim=3)
                    sim_mat_s2=nd.concat(sim_mat_s2,sim_mat_2_count,dim=3)
            if t==0:
                # sim_mat=sim_mat_s
                sim_mat2=sim_mat_s2
            else:
                # sim_mat=nd.concat(sim_mat,sim_mat_s,dim=2)
                sim_mat2=nd.concat(sim_mat2,sim_mat_s2,dim=2)
        return sim_mat2#sim_mat


    def compute_sim_mat(self,x, ref,N,H,W,D,kernel_size,stride1,stride2):#x:img1,ref:img2，D是在img2中搜索的范围，stride1=stride2,分别代表img1和img2中选取点的间隔
        H_new=int(np.ceil((H-2*(D+(kernel_size-1)/2))/stride1))#从x中选取的关键点行数
        W_new=int(np.ceil((W-2*(D+(kernel_size-1)/2))/stride1))#从x中选取的关键点列数
        radius=int(D/stride2)#在img2中间隔搜索的点的半径
        width=2*radius+1#在img2中间隔搜索的点的宽度
        sim_mat=nd.Correlation(x,ref, pad_size = 0, kernel_size = kernel_size, max_displacement = D,stride1=stride1,stride2=stride2, is_multiply = 1)#互相关
        
        #############归一化
        sim_mat_self1=nd.slice(nd.Correlation(x,x, pad_size = 0, kernel_size = kernel_size, max_displacement = D,stride1=stride1,stride2=stride2, is_multiply = 1)
                                            ,begin=(None,int((width*width-1)/2),None,None),end=(None,int((width*width-1)/2)+1,None,None))#img1的自相关
        ##########################img2的自相关
        H_new2=int(np.ceil((H-2*((kernel_size-1)/2))/stride1))
        W_new2=int(np.ceil((W-2*((kernel_size-1)/2))/stride1))
        sim_mat_self2=nd.Correlation(ref,ref, pad_size = 0, kernel_size = kernel_size, max_displacement = 0,stride1=stride1,stride2=stride2, is_multiply = 1)#N,1,H_new2,W_new2
        # pdb.set_trace()
        for t in range (H_new):
            for s in range(W_new):
                h0=int(D/stride1)+t
                w0=int(D/stride1)+s
                sim_mat_2_count=nd.swapaxes(nd.expand_dims(nd.reshape(nd.slice(sim_mat_self2,begin=(None,None,int(h0-radius),int(w0-radius))
                                           ,end=(None,None,int(h0+radius+1),int(w0+radius+1))),(0,0,-1)),3),1,2)#sim_mat[:,K,1,1]
                if s==0:
                    sim_mat_s2=sim_mat_2_count
                else:
                    sim_mat_s2=nd.concat(sim_mat_s2,sim_mat_2_count,dim=3)
            if t==0:
                sim_mat2=sim_mat_s2
            else:
                sim_mat2=nd.concat(sim_mat2,sim_mat_s2,dim=2)
        sim_mat_self2=sim_mat2
        ############################
        sim_mat=nd.broadcast_div(sim_mat,(sim_mat_self1**0.5))/(sim_mat_self2**0.5)###########相似性
        ######################################
        
        return sim_mat.reshape(0,0,int(H_new*W_new)) #N,K,H,W
        

    def compute_corr(self,x, ref):
        #(N, C, 800, 1)
        out=nd.batch_dot(nd.swapaxes(nd.reshape(x,(0,0,0)),1,2),nd.reshape(ref,(0,0,0)))
        xx=nd.norm(x,2,1)
        ref_ref=nd.norm(ref,2,1)
        normalized_out=out/(xx*ref_ref)
        # pdb.set_trace()
        return normalized_out #N,800,800
    def build_correlation(self,x1, x2):
        #x1,x2:N,C,800,1
        sim_mat_12 = self.compute_corr(x1, x2)#N,800,800
        sim_mat_21 = sim_mat_12.transpose((0,2,1))#N,800,800
        print(sim_mat_12.max())
        judge0=nd.ones_like(sim_mat_12)*1#0.8#
        mask=nd.broadcast_greater(sim_mat_12,judge0)
        # sim_mat_12 = self.scoring(sim_mat_12)
        # sim_mat_21 = self.scoring(sim_mat_21)
        return sim_mat_12, sim_mat_21,mask

    def associate(self,sim_mat, topk=2):
        #sim_mat:N,800,800
        indices = nd.topk(sim_mat, axis=2, k=topk, ret_typ='indices')#N,800,2
        sim = nd.topk(sim_mat, axis=2, k=topk, ret_typ='value')#N,800,2
        ratio=nd.slice_axis(sim,axis=2,begin=0,end=1)/nd.slice_axis(sim,axis=2,begin=1,end=2)
        print(ratio.max())
        pdb.set_trace()
        judge=nd.ones_like(ratio)*1.05#1.002#
        mask=nd.broadcast_greater(ratio,judge)
        return nd.slice_axis(indices,axis=2,begin=0,end=1), nd.slice_axis(sim,axis=2,begin=0,end=1),mask##N,800,1
    
    
    def scoring(self,x):
        eps = 1e-10
        x=nd.broadcast_div(nd.broadcast_sub(x,nd.min(x)),(nd.broadcast_sub(nd.max(x),nd.min(x))+ eps))####normalize to 0-1; instead of mean 0
        score = nd.softmax(x, axis=2)
        
        ##N,800,800
        # eps = 1e-10
        # mean = nd.mean(x, 2, keepdims=True)
        # x_zeros_mean=x-mean
        # std = nd.norm(x_zeros_mean, 2, 2)
        # x = x_zeros_mean / (std+eps)
        # score = nd.softmax(x, axis=2)
        return score

    
    def cycle_associate_loss(self,mid_indices,fkp1,fkp2,reassociated_sim,associated_sim,N,mask):
        # pdb.set_trace()
        sim=associated_sim * reassociated_sim#D(i,j*)*D(j*,i*),i*,j*
        sim=-nd.log(sim)#N,1,H*W
        # association loss
        association_loss = nd.squeeze(nd.mean(sim*mask,axis=(1,2)))
        
        # L2 norm loss
        #fkp1:(N, 2, 800, 1)
        # pdb.set_trace()
        masked_mid_indices=mid_indices*mask##N,800,1
        coor1=nd.swapaxes(nd.squeeze(nd.broadcast_mul(fkp1, mask),3),1,2)#i-i*|j* #N,800,2
        coor2=self.gather2(fkp2, masked_mid_indices,N)#i-i*|j* #N,800,2
        dist1=nd.squeeze(nd.sum(nd.norm((coor1-coor2),axis=2),1))
        return association_loss,dist1,coor1,coor2
    def gather2(self,data, index,N): #data:(N, 2, 800, 1)   ######j-i* d ######i-j* matrix index
        index=nd.transpose(index,(2,0,1))#1,N,800
        data=nd.reshape(nd.transpose(data,(2,0,1,3)),(0,0,0))#800,N,2
        out=nd.gather_nd(data,index)
        for i in range(N):
            out_slice=nd.expand_dims(nd.reshape(nd.slice(out,begin=(i,None,i,None),end=(i+1,None,i+1,None)),(-3,-3)),0)
            if i==0:
                output=out_slice
            else:
                output=nd.concat(output,out_slice,dim=0)
        # output=nd.expand_dims(output,axis=1)
        return output
    
    def gather(self,data, index,N): #N,800,1   ######j-i* d ######i-j* matrix index
        index=nd.transpose(index,(2,0,1))#1,N,800
        data=nd.reshape(nd.transpose(data,(1,0,2)),(0,0))#800,N
        out=nd.gather_nd(data,index)
        # pdb.set_trace()
        for i in range(N):
            out_slice=nd.reshape(nd.slice(out,begin=(i,None,i),end=(i+1,None,i+1)),(1,-1))
            if i==0:
                output=out_slice
            else:
                output=nd.concat(output,out_slice,dim=0)
        output=nd.expand_dims(output,axis=2)
        return output
    
    def Association(self,x1, x2,fkp1,fkp2):
        N=x1.shape[0]
        eps = 1e-10
        normalized_x1=nd.broadcast_div(nd.broadcast_sub(x1,nd.min(x1)),(nd.broadcast_sub(nd.max(x1),nd.min(x1))+ eps))####normalize to 0-1; instead of mean 0
        normalized_x2=nd.broadcast_div(nd.broadcast_sub(x2,nd.min(x2)),(nd.broadcast_sub(nd.max(x2),nd.min(x2))+ eps))####normalize to 0-1; instead of mean 0
        
        sim_mat_12,sim_mat_21,mask_first = self.build_correlation(normalized_x1, normalized_x2)#obtain the w    (N,K,H*W)
        # scio.savemat('/home/wxy/Association/images/test/sim_mats.mat',{'sim_mat_12':sim_mat_12.asnumpy(),'sim_mat_21':sim_mat_21.asnumpy()})
        mid_indices, associated_sim,mask_new0 = self.associate(sim_mat_12)######i-j*  #N,800,1
        max_indices,max_sim,mask_new1 = self.associate(sim_mat_21)#N,1,H*W   ######j-i*   #N,800,1
        mask_new=nd.stop_gradient(nd.broadcast_logical_and(mask_new0,mask_new1))
        # scio.savemat('/home/wxy/Association/images/test/indices.mat',{'mid_indices':mid_indices.asnumpy(),'max_indices':max_indices.asnumpy()})
        # print(mask_new0.sum().asnumpy())
        # print(mask_new1.sum().asnumpy())
        # print(mask_new.sum().asnumpy())
        
        grid=nd.linspace(0,799,800,ctx=self.ctx[0])
        grid=nd.repeat(nd.expand_dims(nd.expand_dims(grid,1),0),repeats=N,axis=0)#N,800,1
        # pdb.set_trace()
        mask1=(grid!=max_indices)
        mask2=(grid!=mid_indices)
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask1,mask2))
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask,mask_new))
        
        indices = self.gather(max_indices, mid_indices,N)#i-i*|j*
        # scio.savemat('/home/wxy/Association/images/test/re_index.mat',{'indices':indices.asnumpy()})
        reassociated_sim = self.gather(max_sim, mid_indices,N)#i-i*|j*
        
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask,indices==grid))
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask,mask_first))
        # print(mask.sum().asnumpy())
        # print('*******')
        association_loss,dist1,coor1,coor2=self.cycle_associate_loss(mid_indices,fkp1,fkp2,reassociated_sim,associated_sim,N,mask)
        return association_loss,dist1,coor1,coor2,mask




    def landmark_dist(self, lmk1, lmk2, flows):
        if np.shape(lmk2)[0] > 0:
            flow_len = len(flows)
            shape = nd.array(flows[0].shape[2: 4], ctx=flows[0].context)
            # old lmk_mask is when lmk1 and lmk2 are all not 0, 不知为何，会出现非零的补充。所以改为和200项最后一项相同的就不要
            lmk_mask = (1 - nd.prod(lmk1 == lmk1[0][199][0] * lmk1[0][199][1], axis=-1)) * (
                        1 - nd.prod(lmk2 == lmk2[0][199][0] * lmk2[0][199][1], axis=-1)) > 0.5
            for flow in flows:
                batch_lmk = lmk1 / (nd.reshape(shape, (1, 1, 2)) - 1) * 2 - 1
                batch_lmk = batch_lmk.transpose((0, 2, 1)).expand_dims(axis=3)
                warped_lmk = lmk1 + nd.BilinearSampler(flow, batch_lmk.flip(axis=1)).squeeze(axis=-1).transpose(
                    (0, 2, 1))
                lmk1 = warped_lmk
            lmk_dist = nd.mean(nd.sqrt(nd.sum(nd.square(warped_lmk - lmk2), axis=-1) * lmk_mask + 1e-5), axis=-1)
            lmk_dist = lmk_dist/(np.sum(lmk_mask, axis=1)+1e-5)*200 # 消除当kp数目为0的时候的影响
            lmk_dist = lmk_dist*(np.sum(lmk_mask, axis=1)!=0) # 消除当kp数目为0的时候的影响
            return lmk_dist / (shape[0]*1.414), warped_lmk, lmk2
        else:
            return 0, [], []

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


            # association_loss,dist1,dist2,dist3,mask,associated_sim,max_sim,matrix_index,matrix_index2,matrix_index3,grid=self.association(nd.expand_dims(img1[:,0,:,:],1), nd.expand_dims(img2[:,0,:,:],1))
            # mask=mask.reshape((1,1,512,512))
            # matrix_index=matrix_index.reshape((1,1,512,512))
            # matrix_index2=matrix_index2.reshape((1,1,512,512))
            # matrix_index3=matrix_index3.reshape((1,1,512,512))
            # grid=grid.reshape((1,1,512,512))
            # xs_ori=grid%shape[3]
            # ys_ori=nd.floor(grid/shape[3])
            # xs=matrix_index%shape[3]
            # ys=nd.floor(matrix_index/shape[3])
            # xs2=matrix_index2%shape[3]
            # ys2=nd.floor(matrix_index2/shape[3])
            # xs3=matrix_index3%shape[3]
            # ys3=nd.floor(matrix_index3/shape[3])
            # xs=xs*mask
            # ys=ys*mask
            # xs2=xs2*mask
            # ys2=ys2*mask
            # xs3=xs3*mask
            # ys3=ys3*mask
            # xs_ori=nd.squeeze(xs_ori[0,0,:,:]).asnumpy()
            # ys_ori=nd.squeeze(ys_ori[0,0,:,:]).asnumpy()
            # xs=nd.squeeze(xs[0,0,:,:]).asnumpy()
            # ys=nd.squeeze(ys[0,0,:,:]).asnumpy()###i-j*
            # xs2=nd.squeeze(xs2[0,0,:,:]).asnumpy()
            # ys2=nd.squeeze(ys2[0,0,:,:]).asnumpy()####i-i*
            # xs3=nd.squeeze(xs3[0,0,:,:]).asnumpy()
            # ys3=nd.squeeze(ys3[0,0,:,:]).asnumpy()###j*-i*
            
            
            # mask=nd.squeeze(mask[0,0,:,:]).asnumpy()
            # mask=np.where(mask==1)
            # list1=list(map(list,zip(*mask)))
            # if len(list1)>0:
                # # print(list1)
                # # pdb.set_trace()
                # img1ss=nd.squeeze(img1[0,0,:,:]).asnumpy()
                # warpss=nd.squeeze(warp[0,0,:,:]).asnumpy()
                # fig=plt.figure()
                # ax1 = fig.add_subplot(2, 2, 1)
                # ax2 = fig.add_subplot(2, 2, 2)
                # ax3 = fig.add_subplot(2, 2, 3)
                # im1=self.appendimages(img1ss,warpss)
                # ax1.imshow(im1)
                # ax1.set_title('i-j*')
                # array1=np.array(list1)
                # for i1 in range(len(list1)):
                    # ax1.plot([xs_ori[array1[i1,0],array1[i1, 1]],xs[array1[i1,0],array1[i1, 1]]+shape[2]],[ys_ori[array1[i1,0],array1[i1, 1]],ys[array1[i1,0],array1[i1, 1]]], self.randomcolor(),linewidth=0.1)
                # ax2.imshow(im1)
                # ax2.set_title('i-i*')
                # array1=np.array(list1)
                # for i1 in range(len(list1)):
                    # ax2.plot([xs_ori[array1[i1,0],array1[i1, 1]],xs2[array1[i1,0],array1[i1, 1]]+shape[2]],[ys_ori[array1[i1,0],array1[i1, 1]],ys2[array1[i1,0],array1[i1, 1]]], self.randomcolor(),linewidth=0.1)
                # ax3.imshow(im1)
                # ax3.set_title('j*-i*, num_points='+str(len(list1)))
                # array1=np.array(list1)
                # for i1 in range(len(list1)):
                    # ax3.plot([xs_ori[array1[i1,0],array1[i1, 1]],xs3[array1[i1,0],array1[i1, 1]]+shape[2]],[ys_ori[array1[i1,0],array1[i1, 1]],ys3[array1[i1,0],array1[i1, 1]]], self.randomcolor(),linewidth=0.1)
                
                # # plt.show()
                # # pdb.set_trace()
                # plt.savefig('/home/gelin/s4_deformable_reg_siftflow_as_loss_1024/image_0.2_21_img1_img2/'+str(count)+'.jpg', dpi=600)
                # plt.close(fig)
            
            
            flows = []
            flows.append(flow)
            raw = self.raw_loss_op(img1, warp)
            raws.append(raw.mean())
            dist_loss_mean, warped_lmk, lmk2new = self.landmark_dist(lmk1, lmk2, flows)
            #print('dist_loss_mean={}'.format(dist_loss_mean))
            dist_mean.append(dist_loss_mean*dist_weight)
        #del img1, img2, lmk1,rgb_mean, lmk2,img1s, img2s, lmk1s, lmk2s,pred, occ_masks, flow, warpeds,warp,flows
        rawmean = []
        for raw in raws:
            raw = raw.asnumpy()
            rawmean.append(raw)
        distmean = []
        for distm in dist_mean:
            distm = distm.asnumpy()
            distmean.append(distm)
        return np.median(distmean), np.mean(distmean), np.median(distmean), np.mean(distmean)#,img1.asnumpy(), warp.asnumpy(),flow.asnumpy()#np.median(results_median)



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

        return concatenate((im1, im2), axis=1)
    def randomcolor(self):
        colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0, 14)]
        return "#" + color