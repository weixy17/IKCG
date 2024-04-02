from mxnet.gluon import nn
from mxnet import nd
import math
import pdb


class AssociationLoss(nn.HybridBlock):
    def __init__(self, metric='cos', spagg=False, spagg_alpha=0.5, asso_topk=1,
            print_info=False,max_displacement=3,stride1=1, **kwargs):
        super().__init__(**kwargs)
        #super(AssociationLoss, self).__init__()
        self.metric = metric
        self.spagg = spagg
        self.spagg_alpha = spagg_alpha
        self.asso_topk = asso_topk
        self.print_info = print_info
        self.max_displacement=max_displacement
        self.stride1=stride1
        self.stride2=1
        self.threshold=0.8
    
    
    def corr(self, F,x, ref, pad_size = 7, kernel_size =21, max_displacement = 7, stride1 = 1, stride2 = 1, is_multiply = 1 ):
        return F.Correlation( x, ref, pad_size = int(self.max_displacement+(kernel_size-1)/2), kernel_size = kernel_size, max_displacement = self.max_displacement, stride1 = self.stride1, stride2 = self.stride2, is_multiply = is_multiply)
    
    ############compute the similarity
    def compute_sim_mat(self,F, x, ref,N,H,W):
        x=F.reshape(x,(0,0,int(H*W))) #N,C,H*W
        ref=F.reshape(ref,(0,0,int(H*W))) #N,C,H*W
        for s in range (N):
            for t in range (int(H*W)):
                F.slice(x1,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))
        x1[j, :] = (desc2[j, :] - min(desc2[j, :])) / (max(desc2[j, :]) - min(desc2[j, :]) + 1.0E-08)
        dotprod2m = np.dot(desc1[i, :], desc2[j, :].T) / ((np.dot(desc1[i, :], desc1[i, :].T)) ** 0.5 + 1.0E-08) / ((np.dot(desc2[j, :], desc2[j, :].T)) ** 0.5 + 1.0E-08)
        
        #return corr
        out=F.reshape(corr,(0,0,int(H*W/self.stride1/self.stride1)))
        return out

    def compute_sim_mat_kl(self,F, x1, x2,N,H,W):
        eps = 1e-10
        log_x1 = F.log(x1+eps)
        log_x2 = F.log(x2+eps)
        neg_ent = F.reshape(F.sum(x1 * log_x1, axis=1),(0, -1, 1))
        new_x1=F.transpose(F.reshape(x1,(0, 0, H*W)),(1, 2))
        new_x2=F.reshape(log_x2,(0, 0, H*W))
        cross_ent = -1.0 * F.batch_dot(new_x1,new_x2)
        kl = neg_ent + cross_ent
        return -1.0 * kl

    def build_correlation(self,F, x1, x2,N,C,H,W, metric='cos'):
        if metric == 'cos':
            sim_mat_12 = self.compute_sim_mat(F,x1, x2,N,H,W)
            sim_mat_21 = self.compute_sim_mat(F,x2, x1,N,H,W)

        elif metric == 'kl':
            sim_mat_12 = self.compute_sim_mat_kl(F,x1, x2,N, H,W)
            sim_mat_21 = self.compute_sim_mat_kl(F,x2, x1,N, H,W)

        else:
            raise NotImplementedError
        #return sim_mat_12, sim_mat_21
        sim_mat_12 = self.scoring(F,sim_mat_12)
        sim_mat_21 = self.scoring(F,sim_mat_21)
        return sim_mat_12, sim_mat_21


    def associate_mask(self, F,sim_mat, topk=1):
        indices = F.topk(sim_mat, axis=1, k=topk, ret_typ='indices')
        return indices
    def associate_mask_new(self, F,sim_mat, topk=2):
        dist1=1
        count=0
        judge2=1
        indices = F.topk(sim_mat, axis=1, k=1, ret_typ='indices')
        indices_ori=F.ones_like(indices)
        sim = F.topk(sim_mat, axis=1, k=topk, ret_typ='value')
        difference=F.abs(F.slice_axis(sim,axis=1,begin=0,end=1)-F.slice_axis(sim,axis=1,begin=1,end=2))
        threshold=self.threshold
        while (judge2 and count<=3):
            judge=F.max(difference)-(F.max(difference)-F.min(difference))*threshold
            mask=F.broadcast_greater(difference,judge)
            
            
            diff=indices/self.max_displacement/2/(self.max_displacement+1)-indices_ori
            difference_masked=diff*F.stop_gradient(mask)
            dist1=F.sum(F.norm(difference_masked,axis=2))
            judge2=dist1>0
            count=count+1
            threshold=threshold/2
        return threshold
    def associate(self, F,sim_mat, threshold,topk=2):
        indices = F.topk(sim_mat, axis=1, k=topk, ret_typ='indices')
        sim = F.topk(sim_mat, axis=1, k=topk, ret_typ='value')
        difference=F.abs(F.slice_axis(sim,axis=1,begin=0,end=1)-F.slice_axis(sim,axis=1,begin=1,end=2))
        judge=F.max(difference)-(F.max(difference)-F.min(difference))*threshold
        mask=F.broadcast_greater(difference,judge)
        sim=F.topk(sim_mat, axis=1, k=1, ret_typ='value')
        indices = F.topk(sim_mat, axis=1, k=1, ret_typ='indices')
        return indices, sim,mask

    def associate_gt(self,F, gt, indices):
        gt = F.broadcast_axis(F.reshape(gt,(0, -1, 1)),axis=2, size=1)
        end_gt = gt
################################################
        associated_gt = self.gather(end_gt, 1, indices)
        gt = (gt == associated_gt).type(F.cuda.FloatTensor)
        return gt.reshape(0, H, W, K)
    
    
    def scoring(self,F, x, dim=1):
        #0, L1, L2 = F.shape_array(x)
        eps = 1e-10
        # mean = F.mean(x, axis=dim, keepdims=True)
        # std = F.expand_dims(F.norm(F.broadcast_sub(x,mean),ord=2,axis=dim),1)/math.sqrt((2*self.max_displacement+1)*(2*self.max_displacement+1)-1)
        # std=F.mean(std)
        # x = F.broadcast_div(F.broadcast_sub(x,mean) , (std+eps))
        x=F.broadcast_div(F.broadcast_sub(x,F.min(x)),(F.broadcast_sub(F.max(x),F.min(x))+ eps))####normalize to 0-1; instead of mean 0
        
        score = F.softmax(x, axis=dim)
        return score

    def spatial_agg(self,F, x,C,H,W, mask=None, metric='cos'):
        x_clone=F.ones_like(x)
        x_clone=x_clone*x
        if metric == 'cos':
            sim_mat = self.compute_sim_mat(F,x, x_clone, H,W)
        elif metric == 'kl':
            sim_mat = self.compute_sim_mat_kl(F,x, x_clone, H,W)
        else:
             raise NotImplementedError

        if metric == 'cos':
            sim_mat = self.scoring(F,sim_mat)
        else:
            sim_mat = F.softmax(sim_mat, axis=1)#the w
        ones=F.ones_like(x)/C
        for i in range (int(C)):
            x_slice=F.slice_axis(x,axis=1,begin=i,end=i+1)
            x_slices=F.broadcast_axis(x_slice,axis=1,size=int(C))
            if i==0:
                self_corr=self.corr(F,x_slices, ones, pad_size = int(self.max_displacement), kernel_size = 1, max_displacement = self.max_displacement, stride1 = self.stride1, stride2 =self.stride2, is_multiply = 1 )
                self_corr=F.reshape(self_corr,(0,0,H*W))
                out = F.reshape(F.expand_dims(F.sum(self_corr*sim_mat,axis=1, keepdims=True),axis=2),(0,0,H,W))
            else:
                self_corr_pre=self.corr(F,x_slices, ones, pad_size = int(self.max_displacement), kernel_size = 1, max_displacement = self.max_displacement, stride1 = self.stride1, stride2 = self.stride2, is_multiply = 1 )
                self_corr=F.reshape(self_corr,(0,0,H*W))
                #out = F.stack(out,F.reshape(F.expand_dims(F.sum(self_corr*sim_mat,axis=1, keepdims=True),axis=2),(0,0,H,W)),axis=1)
                out=F.concat(out,F.reshape(F.expand_dims(F.sum(self_corr*sim_mat,axis=1, keepdims=True),axis=2),(0,0,H,W)),dim=1)
        return out

    
    
    def cycle_associate_loss(self,F,indices,reassociated_sim,mid_indices,max_indices, associated_sim,N,C,H,W,masks):
        sim=associated_sim * reassociated_sim#D(i,j*)*D(j*,i*),i*,j*
        sim=-F.log(sim)#-sim#
        sim = F.reshape(F.expand_dims(F.transpose(sim,(0,2,1)),axis=3),(0, int(H/self.stride1),int(W/self.stride1), -1))
        
        # mean = F.mean(sim)
        # std = F.norm(F.broadcast_sub(sim,mean))/math.sqrt((2*self.max_displacement+1)*(2*self.max_displacement+1)-1)
        # sim = F.broadcast_div(F.broadcast_sub(sim,mean),std+1e-10)#N,H,W,1
        sim_mask=F.stop_gradient(F.reshape(masks,(0,int(H/self.stride1),int(W/self.stride1),1)))
        sim_masked=sim*sim_mask
        # association loss
        association_loss = F.squeeze(F.mean(sim_masked,axis=(1,2)))
        
        # L2 norm loss
        indices_ori=F.ones_like(indices)#N,1,H*W
        difference=indices/self.max_displacement/2/(self.max_displacement+1)-indices_ori ###i-i*
        difference_masked=difference*F.stop_gradient(masks)

        difference2=mid_indices/self.max_displacement/2/(self.max_displacement+1)-indices_ori###i-j*
        difference_masked2=difference2*F.stop_gradient(masks)
        
        difference3=max_indices/self.max_displacement/2/(self.max_displacement+1)-indices_ori###j*-i*
        difference_masked3=difference3*F.stop_gradient(masks)
        dist1=F.squeeze(F.norm(difference_masked,axis=2))
        dist2=F.squeeze(F.norm(difference_masked2,axis=2))
        dist3=F.squeeze(F.norm(difference_masked3,axis=2))
        return association_loss,dist1,dist2,dist3
    def gather(self,F,data, index,N): #N,1,H*W/stride2/stride2   ######j-i* d          ######i-j* matrix index
        index=F.transpose(index,(1,0,2))#1,N,H*W/stride2/stride2
        data=F.reshape(F.transpose(data,(2,1,0)),(0,-1))#H*W/stride2/stride2,N
        out=F.gather_nd(data,index)
        for i in range(N):
            out_slice=F.reshape(F.slice(out,begin=(i,None,i),end=(i+1,None,i+1)),(1,-1))
            if i==0:
                output=out_slice
            else:
                output=F.concat(output,out_slice,dim=0)
        output=F.expand_dims(output,axis=1)
        return output
    
    def hybrid_forward(self,F, x1, x2,ctx):
        N,C,H,W=2,196,8,8

        threshold=0.2
        sim_mat_12, sim_mat_21 = self.build_correlation(F,x1, x2,N,C,H,W, metric=self.metric)#obtain the w    (N,K,H*W)
        mid_indices, associated_sim,mask_new0 = self.associate(F,sim_mat_12,threshold)######i-j*
        max_indices,max_sim,mask_new1 = self.associate(F,sim_mat_21,threshold)#N,1,H*W/stride2/stride2   ######j-i*
        mask_new=F.stop_gradient(F.broadcast_logical_and(mask_new0,mask_new1))
        
        grid=F.linspace(0,int(H*W/self.stride1/self.stride1)-1,int(H*W/self.stride1/self.stride1))
        grid=F.repeat(F.expand_dims(F.expand_dims(grid,0),0),repeats=N,axis=0)
        xs=mid_indices%(self.max_displacement/self.stride2*2+1)-self.max_displacement/self.stride2
        ys=F.floor(mid_indices/(self.max_displacement/self.stride2*2+1))-self.max_displacement/self.stride2
        dis_index=ys*W+xs
        matrix_index=(dis_index+grid).clip(0, int(H*W/self.stride1/self.stride1)-1)#N,1,H*W/stride2/stride2 ######i-j*
        mask00=matrix_index>=0
        mask01=matrix_index<=int(H*W/self.stride1/self.stride1)-1
        mask0=F.stop_gradient(F.broadcast_logical_and(mask00,mask01))
        mask=F.stop_gradient(F.broadcast_logical_and(mask0,mask_new))
        
        
        
        indices = self.gather(F,max_indices, matrix_index,N)
        reassociated_sim = self.gather(F,max_sim, matrix_index,N)
        
        xs=indices%(self.max_displacement/self.stride2*2+1)-self.max_displacement/self.stride2
        ys=F.floor(indices/(self.max_displacement/self.stride2*2+1))-self.max_displacement/self.stride2
        dis_index=ys*W+xs
        matrix_index2=(dis_index+grid).clip(0, int(H*W/self.stride1/self.stride1)-1)######i-i*
        
        
        xs=max_indices%(self.max_displacement/self.stride2*2+1)-self.max_displacement/self.stride2
        ys=F.floor(max_indices/(self.max_displacement/self.stride2*2+1))-self.max_displacement/self.stride2
        dis_index=ys*W+xs
        matrix_index3=(dis_index+grid).clip(0, int(H*W/self.stride1/self.stride1)-1)######j-i*
        
        # association_loss,dist1,dist2,dist3=self.cycle_associate_loss(F,indices,reassociated_sim,mid_indices,max_indices, associated_sim,N,C,H,W,mask)
        
        
        
        
        mask_new=(matrix_index2==grid)
        mask=F.broadcast_logical_and(mask,mask_new)
        if mask.sum()>0:
            index=F.topk(mask,axis=2,k=200)
            xs_ori=F.take((grid*mask)%W,index,axis=2)
            ys_ori=F.take(F.floor((grid*mask)/W),index,axis=2)
            xs=F.take((matrix_index*mask)%W,index,axis=2)
            ys=F.take(F.floor((matrix_index*mask)/W),index,axis=2)
            xs3=F.take((matrix_index3*mask)%W,index,axis=2)
            ys3=F.take(F.floor((matrix_index3*mask)/W),index,axis=2)
            for s in range(W):
                ori_lmk_slice=F.expand_dims(F.concat(F.reshape(F.squeeze(F.slice(xs_ori,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),F.reshape(F.slice(ys_ori,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                s_lmk_slice=F.expand_dims(F.concat(F.reshape(F.squeeze(F.slice(xs,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),F.reshape(F.slice(ys,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                s3_lmk_slice=F.expand_dims(F.concat(F.reshape(F.squeeze(F.slice(xs3,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None))),(-1,1)),F.reshape(F.slice(ys3,begin=(s,None,s,None,None),end=(s+1,None,s+1,None,None)).squeeze(),(-1,1)),dim=1),0)
                if s==0:
                    ori_lmk=ori_lmk_slice
                    s_lmk=s_lmk_slice
                    s3_lmk=s3_lmk_slice
                else:
                    ori_lmk=F.concat(ori_lmk,ori_lmk_slice,dim=0)
                    s_lmk=F.concat(s_lmk,s_lmk_slice,dim=0)
                    s3_lmk=F.concat(s3_lmk,s3_lmk_slice,dim=0)
        else:
            ori_lmk=F.zeros((N,200, 2), ctx[0])
            s_lmk= F.zeros((N,200, 2), ctx[0])
            s3_lmk= F.zeros((N,200, 2), ctx[0])
        
        return ori_lmk,s_lmk,s3_lmk
        # return association_loss,dist1,dist2,dist3,mask,associated_sim,max_sim,matrix_index,matrix_index2,matrix_index3,grid
        # return mask,matrix_index,matrix_index2,matrix_index3,grid


# if __name__=='__main__':

    # loss=AssociationLoss()
    # loss.hybridize()
    # #x=F.random.randn(2,3,512,512)
    # while True:
        # x=nd.ones((1,1,20,20))
        # #x[0,0,4,4]=3
        # # y=nd.random.randn(1,3,10,10)
        # y=nd.random.randn(1,1,20,20)
        # # print(x)
        # # print(y)
        # #mid_indices, associated_sim,max_indices,max_sim,indices,reassociated_sim,association_loss,dist1,dist2=loss(x,y)#, associated_sim,max_indices,max_sim
        # #association_loss,dist1,dist2=loss(x,y)#, associated_sim,max_indices,max_sim
        # #print(association_loss,dist1,dist2)
        # judge_zeros,association_loss,dist1,dist2,dist3=loss(x,y)
        # print(judge_zeros,association_loss,dist1,dist2,dist3)
        # loss_all=association_loss+(dist1+dist2+dist3)/1000
        # loss_all.backward()
    
    