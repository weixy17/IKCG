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



    def train_batch_association(self, img1, img2,lmk1,lmk2,dist_weight,step):
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
        img1, img2,lmk1,lmk2= map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2,lmk1,lmk2))
        hsh = "".join(random.sample(string.ascii_letters + string.digits, 10))
        count=0
        with autograd.record():
            for img1s, img2s,lmk1,lmk2 in zip(img1, img2,lmk1,lmk2):
                
                
                img1s, img2s = img1s/ 255.0+0.5, img2s/ 255.0+0.5
                
                # img1s, img2s, _ = self.centralize(img1s0, img2s0)
                shape = img1s.shape
                pred, c1s,warps,_ = self.network(img1s, img2s) # this warpeds is not mean the warped image
                
                flow = self.upsampler(pred[-1])
                
                if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
                    flow = nd.contrib.BilinearResize2D(flow, height=shape[2], width=shape[3]) * nd.array([shape[d] / flow.shape[d] for d in (2, 3)], ctx=flow.context).reshape((1, 2, 1, 1))
                warp = self.reconstruction(img2s, flow)
                raw_loss = self.raw_loss_op(img1s, warp)
                reg_loss = self.regularization_op(flow) + self.boundary_loss_op(flow) * self.boundary_weight # gl, in this program, although cascaded, len(flow)=1
                dist_loss, warped_lmk, _ = self.landmark_dist(lmk1, lmk2, [flow])
                
                
                
                
                
                delete=150
                delete2=-150
                input1=nd.expand_dims(img1s[:,0,delete:delete2,delete:delete2],1)
                input2=nd.expand_dims(warp[:,0,delete:delete2,delete:delete2],1)
                N,C,H,W=input1.shape[0],input1.shape[1],input1.shape[2],input1.shape[3]
                # print(H)
                D=min(5,int(H/2-1))
                stride1=1
                stride2=1
                radius=int(D/stride2)
                width=int(radius*2+1)
                kernel_size=11
                img_association_loss,img_dist1,mid_indices,associated_sim,max_indices,mask,sim_mat_12,sim_mat_21=self.Association(input1, input2,input1.shape,
                            D=D,stride1=stride1,stride2=stride2,kernel_size=kernel_size,metric='img')
                # img_association_loss,dist1,dist2,dist3=self.association(input1, input2)
                # img_dist1=dist1+dist2+dist3
                
                ##############可视化
                H_new=int(np.ceil((H-2*(D+(kernel_size-1)/2))/stride1))
                W_new=int(np.ceil((W-2*(D+(kernel_size-1)/2))/stride1))
                for t in range (H_new):
                    grid_slice=nd.linspace(int((D+t*stride1+(kernel_size-1)/2)*W+D+(kernel_size-1)/2),int((D+t*stride1+(kernel_size-1)/2)*W+D+(kernel_size-1)/2+(int(W_new)-1)*stride1),int(W_new),ctx=self.ctx[0])
                    if t==0:
                        grid=grid_slice
                    else:
                        grid=nd.concat(grid,grid_slice,dim=0)
                grid=nd.repeat(nd.expand_dims(nd.expand_dims(grid,0),0),repeats=N,axis=0)#N,1,H*W
                xs=((mid_indices)%width-radius)*stride2
                ys=(nd.floor((mid_indices)/width)-radius)*stride2
                dis_index=ys*W+xs
                matrix_index=(dis_index+grid).clip(0, int(H*W)-1)#N,1,H*W ######i-j*
                matrix_index=matrix_index*mask
                
                grid2=nd.linspace(0,int(H_new*W_new)-1,int(H_new*W_new),ctx=self.ctx[0])
                grid2=nd.repeat(nd.expand_dims(nd.expand_dims(grid2,0),0),repeats=N,axis=0)#N,1,H*W
                xs=((mid_indices)%width-radius)
                ys=(nd.floor((mid_indices)/width)-radius)
                dis_index=ys*W_new+xs
                matrix_index2=(dis_index+grid2).clip(0, int(H_new*W_new)-1)#N,1,H*W ######i-j*
                matrix_index2=matrix_index2*mask
                num=int(mask.sum().asnumpy())
                if num>0 and step<3000:
                    index=nd.topk(mask,axis=2,k=min(int(num),200))#N,1,num
                    xs_ori=nd.take((grid)%(W),index,axis=2)
                    ys_ori=nd.take(nd.floor((grid)/(W)),index,axis=2)
                    xs=nd.take((matrix_index)%(W),index,axis=2)#N,1,N,1,num
                    ys=nd.take(nd.floor(matrix_index/(W)),index,axis=2)#N,1,N,1,num
                    s2_pre=nd.take((matrix_index2),index,axis=2)#N,1,N,1,num
                    for p in range (N):
                        s2_pre_slice=nd.expand_dims(nd.reshape(nd.squeeze(nd.slice(s2_pre,begin=(p,None,p,None,None),end=(p+1,None,p+1,None,None))),(1,-1)),0)
                        if p==0:
                            s2_post=s2_pre_slice
                        else:
                            s2_post=nd.concat(s2_post,s2_pre_slice,dim=0)
                    s2=nd.take(max_indices,s2_post,axis=2)
                    xs2=(s2%width-radius)*stride2+xs
                    ys2=(nd.floor(s2/width)-radius)*stride2+ys
                    for s in range(N):
                        ori_lmk_slice=nd.expand_dims(nd.concat(nd.reshape(nd.squeeze(nd.slice(xs_ori,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),nd.reshape(nd.slice(ys_ori,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                        s_lmk_slice=nd.expand_dims(nd.concat(nd.reshape(nd.squeeze(nd.slice(xs,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),nd.reshape(nd.slice(ys,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                        s2_lmk_slice=nd.expand_dims(nd.concat(nd.reshape(nd.squeeze(nd.slice(xs2,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),nd.reshape(nd.slice(ys2,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                        if s==0:
                            ori_lmk=ori_lmk_slice
                            s_lmk=s_lmk_slice
                            s2_lmk=s2_lmk_slice
                        else:
                            ori_lmk=nd.concat(ori_lmk,ori_lmk_slice,dim=0)
                            s_lmk=nd.concat(s_lmk,s_lmk_slice,dim=0)
                            s2_lmk=nd.concat(s2_lmk,s2_lmk_slice,dim=0)
                    ori_lmk=ori_lmk+delete
                    s_lmk=s_lmk+delete
                    s2_lmk=s2_lmk+delete
                    img1ss=nd.squeeze(img1s[0,0,:,:]).asnumpy()
                    img2ss=nd.squeeze(warp[0,0,:,:]).asnumpy()
                    plt.figure()
                    im1=self.appendimages(img1ss,img2ss)
                    plt.imshow(im1)
                    plt.title('i-j*,num='+str(num))
                    for i1 in range(min(int(num),200)):
                        plt.plot([(ori_lmk.asnumpy())[0,i1,0],(s_lmk.asnumpy())[0,i1,0]+shape[2]],[(ori_lmk.asnumpy())[0,i1,1],(s_lmk.asnumpy())[0,i1,1]], '#FF0000',linewidth=0.1)
                    for i1 in range(200):
                        plt.plot([(lmk1.asnumpy())[0,i1,1],(warped_lmk.asnumpy())[0,i1,1]+shape[2]],[(lmk1.asnumpy())[0,i1,0],(warped_lmk.asnumpy())[0,i1,0]], '#FFFF00',linewidth=0.1)
                    if not os.path.exists('/data3/wxy/Association/images/images_5_11_with_labels/'):
                        os.mkdir('/data3/wxy/Association/images/images_5_11_with_labels/')
                    plt.savefig('/data3/wxy/Association/images/images_5_11_with_labels/'+str(step)+'_'+str(count)+'.jpg', dpi=600)
                    plt.close()
                    
                    # scio.savemat('/data3/wxy/Association/images/image_5_11/'+str(step)+'_'+str(count)+'.mat',{'mid_indices':mid_indices.asnumpy(),'associated_sim':associated_sim.asnumpy()
                    # ,'sim_mat_12':sim_mat_12.asnumpy(),'sim_mat_21':sim_mat_21.asnumpy(),'mask':mask.asnumpy(),'img1':img1ss,'img2':img2ss})
                    print(ori_lmk)
                    print(s_lmk)
                    
                

                
                # feature_association_loss,feature_dist1=0,0
                # stride1=1
                # stride2=1
                # # radius=int(D/stride2)
                # # width=int(radius*2+1)
                
                # for i in range (5):
                    # f1=c1s[i]##N,C,H,W
                    # warp_f2=warps[i]
                    # shape_new=f1.shape
                    # D=min(5,int(shape_new[3]/4-1))
                    # kernel_size=min(5,int(shape_new[3]/4-1))
                    # f1=nd.broadcast_div(nd.broadcast_minus(f1,nd.min(f1,axis=1,keepdims=1)),nd.broadcast_minus(nd.max(f1,axis=1,keepdims=1),nd.min(f1,axis=1,keepdims=1)))
                    # warp_f2=nd.broadcast_div(nd.broadcast_minus(warp_f2,nd.min(warp_f2,axis=1,keepdims=1)),nd.broadcast_minus(nd.max(warp_f2,axis=1,keepdims=1),nd.min(warp_f2,axis=1,keepdims=1)))
                    # feat_association_loss,feat_dist1,mid_indices,associated_sim,max_indices,mask,sim_mat_12,sim_mat_21=self.Association(f1, warp_f2,shape_new,
                            # D=D,stride1=stride1,stride2=stride2,kernel_size=kernel_size,metric='feature')
                    # feature_association_loss=feature_association_loss+feat_association_loss*self.multiscale_weights[i]
                    # feature_dist1=feature_dist1+feat_dist1*self.multiscale_weights[i]
                
                association_weight=1
                self.reg_weight = 1 #0.2
                loss = raw_loss * self.raw_weight + reg_loss * self.reg_weight+association_weight*(img_dist1)+img_association_loss#*1000#+(feature_dist1)*0.1+(feature_association_loss)#
                # loss=nd.abs(loss-0.18)+0.18
                #loss = dist_loss * dist_weight  # gl, in this program no regulation in yaml, so self.reg_weight=0. add after if necessary
                losses.append(loss)
                reg_losses.append(reg_loss)
                raw_losses.append(raw_loss)
                img_association_losses.append(img_association_loss)
                # feature_association_losses.append(feature_association_loss)
                img_dist1_losses.append(img_dist1)
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
        


    def build_correlation(self,x1, x2,N,H,W,D,kernel_size,stride1,stride2, metric='feature'):
        #x1,x2:N,C,H,W
        if metric == 'feature':
            sim_mat_12 = self.compute_sim_mat(x1, x2,N,H,W,D,kernel_size,stride1,stride2)
            # pdb.set_trace()
            sim_mat_21 = self.compute_sim_mat(x2, x1,N,H,W,D,kernel_size,stride1,stride2)
        elif metric == 'img':
            sim_mat_12 = self.compute_sim_mat(x1, x2,N, H,W,D,kernel_size,stride1,stride2)
            # pdb.set_trace()
            sim_mat_21 = self.compute_sim_mat(x2, x1,N, H,W,D,kernel_size,stride1,stride2)
            # pdb.set_trace()
        else:
            raise NotImplementedError
        judge0=nd.ones_like(sim_mat_12)*0.95#0.8#
        mask0=nd.broadcast_greater(sim_mat_12,judge0)
        mask1=nd.broadcast_greater(sim_mat_21,judge0)
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask0,mask1))
        # print(mask.sum().asnumpy())
        sim_mat_12 = self.scoring(sim_mat_12*mask)
        sim_mat_21 = self.scoring(sim_mat_21*mask)
        sim_mat_12 = self.scoring(sim_mat_12)
        sim_mat_21 = self.scoring(sim_mat_21)
        return sim_mat_12, sim_mat_21

    def associate(self,sim_mat, topk=2):
        #sim_mat:N,K,H*W
        indices = nd.topk(sim_mat, axis=1, k=topk, ret_typ='indices')#N,2,h*w
        sim = nd.topk(sim_mat, axis=1, k=topk, ret_typ='value')#N,2,h*w
        ratio=nd.slice_axis(sim,axis=1,begin=0,end=1)/nd.slice_axis(sim,axis=1,begin=1,end=2)
        # print(ratio.max())
        judge=nd.ones_like(ratio)*1.05#1.002#
        mask=nd.broadcast_greater(ratio,judge)
        return nd.slice_axis(indices,axis=1,begin=0,end=1), nd.slice_axis(sim,axis=1,begin=0,end=1),mask
    
    
    def scoring(self,x, dim=1):
        eps = 1e-10
        x=nd.broadcast_div(nd.broadcast_sub(x,nd.min(x)),(nd.broadcast_sub(nd.max(x),nd.min(x))+ eps))####normalize to 0-1; instead of mean 0
        score = nd.softmax(x, axis=dim)
        return score

    
    def cycle_associate_loss(self,mid_indices, reassociated_sim,associated_sim,N,C,H,W,H_new,W_new,width,mask):
        # pdb.set_trace()
        sim=associated_sim * reassociated_sim#D(i,j*)*D(j*,i*),i*,j*
        sim=-nd.log(sim)#N,1,H*W
        # association loss
        association_loss = nd.squeeze(nd.mean(sim*mask,axis=(1,2)))
        # L2 norm loss
        difference1=(mid_indices-(width*width-1)/2)*mask###i-j*    #N,1,H*W
        print(mask.sum().asnumpy())
        # print(difference1.sum().asnumpy())
        dist1=nd.squeeze(nd.norm(difference1,axis=2))/H/1.414
        # print(dist1.asnumpy())
        return association_loss,dist1
    def gather(self,data, index,N): #N,1,H*W   ######j-i* d ######i-j* matrix index
        index=nd.transpose(index,(1,0,2))#1,N,H*W
        data=nd.reshape(nd.transpose(data,(2,1,0)),(0,-1))#H*W,N
        out=nd.gather_nd(data,index)
        for i in range(N):
            out_slice=nd.reshape(nd.slice(out,begin=(i,None,i),end=(i+1,None,i+1)),(1,-1))
            if i==0:
                output=out_slice
            else:
                output=nd.concat(output,out_slice,dim=0)
        output=nd.expand_dims(output,axis=1)
        return output
    
    def Association(self,x1, x2,shape,D,stride1,stride2,kernel_size,metric):
        N,C,H,W=shape[0],shape[1],shape[2],shape[3]
        radius=int(D/stride2)
        width=int(radius*2+1)
        H_new=int(np.ceil((H-2*(D+(kernel_size-1)/2))/stride1))
        W_new=int(np.ceil((W-2*(D+(kernel_size-1)/2))/stride1))
        sim_mat_12,sim_mat_21 = self.build_correlation(x1, x2,N,H,W,D,kernel_size,stride1,stride2, metric)#obtain the w    (N,K,H*W)
        # scio.savemat('/home/wxy/Association/images/test/sim_mats.mat',{'sim_mat_12':sim_mat_12.asnumpy(),'sim_mat_21':sim_mat_21.asnumpy()})
        mid_indices, associated_sim,mask_new0 = self.associate(sim_mat_12)######i-j*
        max_indices,max_sim,mask_new1 = self.associate(sim_mat_21)#N,1,H*W   ######j-i*
        # scio.savemat('/home/wxy/Association/images/test/indices.mat',{'mid_indices':mid_indices.asnumpy(),'max_indices':max_indices.asnumpy()})
        
        mask_new=nd.stop_gradient(nd.broadcast_logical_and(mask_new0,mask_new1))
        # print(mask_new0.sum().asnumpy())
        # print(mask_new1.sum().asnumpy())
        # print(mask_new.sum().asnumpy())
        grid=nd.linspace(0,int(H_new*W_new)-1,int(H_new*W_new),ctx=self.ctx[0])
        grid=nd.repeat(nd.expand_dims(nd.expand_dims(grid,0),0),repeats=N,axis=0)#N,1,H*W
        xs=(mid_indices%width-radius)#*stride2
        ys=(nd.floor(mid_indices/width)-radius)#*stride2
        dis_index=ys*W_new+xs
        matrix_index=(dis_index+grid).clip(0, int(H_new*W_new)-1)#N,1,H*W ######i-j*
        
        mask0=nd.stop_gradient(nd.broadcast_logical_and((dis_index+grid)>=0,(dis_index+grid)<=int(H_new*W_new)-1))
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask0,mask_new))
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask,matrix_index!=grid))
        # print(mask.sum().asnumpy())
        indices = self.gather(max_indices, matrix_index,N)#i-i*|j*
        # scio.savemat('/home/wxy/Association/images/test/re_index.mat',{'indices':indices.asnumpy()})
        reassociated_sim = self.gather(max_sim, matrix_index,N)#i-i*|j*
        xs=(indices%width-radius)#*stride2
        ys=(nd.floor(indices/width)-radius)#*stride2
        dis_index=ys*W_new+xs
        matrix_index2=(dis_index+matrix_index).clip(0, int(H_new*W_new)-1)#N,1,H*W ######i-i*
        # scio.savemat('/home/wxy/Association/images/test/matrix_index.mat',{'matrix_index':matrix_index.asnumpy(),'matrix_index2':matrix_index2.asnumpy()})
        mask=nd.stop_gradient(nd.broadcast_logical_and(mask,matrix_index2==grid))
        # print(mask.sum().asnumpy())
        # print('*******')

        association_loss,dist1=self.cycle_associate_loss(mid_indices,reassociated_sim,associated_sim,N,C,H,W,H_new,W_new,width,mask)
        return association_loss,dist1,mid_indices,associated_sim,max_indices,mask,sim_mat_12,sim_mat_21




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