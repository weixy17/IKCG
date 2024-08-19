import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.distributed as dist
import pdb

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, ds_weights=None):
        super(SegCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ds_weights = ds_weights

    def forward(self, preds, target):
        loss = 0
        if not isinstance(preds, list):
            loss = self.criterion(pred, target)
        else:
            count = 0
            for pred in preds:
                if self.ds_weights is None or len(self.ds_weights) == 0:
                    cur_weight = 1.0
                elif count > len(self.ds_weights) - 1:
                    cur_weight = self.ds_weights[-1]
                else:
                    cur_weight = self.ds_weights[count]

                loss += cur_weight * self.criterion(pred, target)
                count += 1

        return loss

class AssociationLoss(nn.Module):
    def __init__(self, metric='cos',img_size=512,
            print_info=False):
        super(AssociationLoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        self.metric = metric
        self.img_size = img_size
        self.print_info = print_info

    def compute_sim_mat(self, x, ref):#x1(5, C, 1, 800)
        assert(len(x.size()) == 4), x.size()
        N, C, H, W = x.size()
        _, _, Hr, Wr = ref.size()
        assert(x.shape[:2] == ref.shape[:2]), ref.size()

        normalized_x = F.normalize(x.view(-1, C, H*W).transpose(1, 2), dim=2)
        normalized_ref = F.normalize(ref.view(-1, C, Hr*Wr), dim=1)
        sim_mat = torch.matmul(normalized_x, normalized_ref)
        return torch.mean(sim_mat,0,keepdim=True),normalized_x, normalized_ref,sim_mat

    def compute_sim_mat_kl(self, x1, x2):
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        eps = 1e-10
        log_x1 = torch.log(x1+eps)
        log_x2 = torch.log(x2+eps)
        neg_ent = torch.sum(x1 * log_x1, dim=1).view(N, -1, 1)
        cross_ent = -1.0 * torch.matmul(x1.view(N, -1, H*W).transpose(1, 2), log_x2.view(N, -1, H2*W2))
        kl = neg_ent + cross_ent
        return -1.0 * kl

    def build_correlation(self, x1, x2, metric='cos'):
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        if metric == 'cos':
            sim_mat_12,normalized_x, normalized_ref,sim_mat = self.compute_sim_mat(x1, x2)
            sim_mat_21 = sim_mat_12.transpose(1, 2)

        elif metric == 'kl':
            sim_mat_12 = self.compute_sim_mat_kl(x1, x2)
            sim_mat_21 = self.compute_sim_mat_kl(x2, x1)

        else:
            raise NotImplementedError

        return sim_mat_12, sim_mat_21,normalized_x, normalized_ref,sim_mat

    def associate(self, sim_mat,fkp, topk=1):
        #############fkp :(N,1,61,2)
        indices = torch.topk(sim_mat, dim=2, k=2).indices.detach()
        fkp_ref=torch.index_select(fkp,2,indices[:,:,0].squeeze())
        d_temp=(torch.sum(fkp_ref*fkp_ref,3,True)+torch.sum(fkp*fkp,3,True).transpose(2,3)-2*torch.matmul(fkp_ref,fkp.transpose(3,2))).abs()#N,1,800,800
        mask_nms1=d_temp>=(0.004**2)*(self.img_size**2)*2#0.005
        mask_nms2=d_temp==0
        mask_nms=((mask_nms1+mask_nms2)>=1)
        sim = torch.topk(sim_mat*(mask_nms.squeeze(1)), dim=2, k=2).values
        mask1=(sim[:,:,1].unsqueeze(2)<sim[:,:,0].unsqueeze(2)*0.995)########0.99
        mask2=sim[:,:,0].unsqueeze(2)>0.75#0.8#.85#.9
        mask=mask1*mask2==1
        return indices[:,:,0].unsqueeze(2), sim[:,:,0].unsqueeze(2),mask

    def associate_gt(self, gt, indices):
        N, H, W = gt.size()
        K = indices.size(2)
        gt = gt.view(N, -1, 1).expand(N, -1, K)
        end_gt = gt

        associated_gt = torch.gather(end_gt, 1, indices)
        gt = (gt == associated_gt).type(torch.cuda.FloatTensor).detach()
        return gt.view(N, H, W, K)
       
    def cycle_associate(self, sim_mat_12, sim_mat_21,fkp1,fkp2):#fkp(N, 1,800, 2) [-1,1]
        N, Lh, Lw = sim_mat_12.size()#N,200,200
        d=(torch.sum(fkp1*fkp1,3,True)+torch.sum(fkp2*fkp2,3,True).transpose(2,3)-2*torch.matmul(fkp1,fkp2.transpose(3,2))).abs()#N,1,800,800
        mask_zone=d<(0.028**2)*(self.img_size**2)*2
        print(mask_zone.sum().detach().cpu().numpy())
        mid_indices, associated_sim,mask12 = self.associate(sim_mat_12*mask_zone.squeeze(1),fkp2)#N,800,1
        max_indices,max_sim,mask21 = self.associate(sim_mat_21*mask_zone.transpose(3,2).squeeze(1),fkp1)#N,800,1

        print(mask12.sum().detach().cpu().numpy())
        print(mask21.sum().detach().cpu().numpy())
        index=torch.topk(mask12*1.0,k=int(mask12.sum().detach().cpu().numpy()),dim=1).indices#N,orb_nums,1
        mid_indices_valid=torch.index_select(mid_indices,1,index.squeeze())
        index2=torch.topk(mask21*1.0,k=int(mask21.sum().detach().cpu().numpy()),dim=1).indices#N,orb_nums,1
        max_indices_valid=torch.index_select(max_indices,1,index2.squeeze())
        indices = torch.gather((max_indices+1)*(mask21*2-1)-1, 1, mid_indices_valid)#*mask21   #N,orb_nums,1
        indices2 = torch.gather((mid_indices+1)*(mask12*2-1)-1, 1, max_indices_valid)#*mask21   #N,orb_nums,1
        
        reassociated_sim = torch.gather(max_sim, 1, mid_indices)
        mask=(index==indices)
        mask2=(index2==indices2)
        index_valid=torch.masked_select(index,mask)#[K]
        index_valid2=torch.masked_select(index2,mask2)#[K]
        print(mask.sum().detach().cpu().numpy())
        print(mask2.sum().detach().cpu().numpy())
        if (mask.sum().detach().cpu().numpy()!=mask2.sum().detach().cpu().numpy()):
            pdb.set_trace()
        return associated_sim * reassociated_sim, indices, max_indices,mid_indices,index_valid,index_valid2,mask_zone,d,mask12,associated_sim,max_sim

    def scoring(self, x, dim=2):
        N, L1, L2 = x.size()
        eps = 1e-10
        mean = torch.mean(x, dim=dim, keepdim=True).detach()
        std = torch.std(x, dim=dim, keepdim=True).detach()
        x = (x-mean) / (std+eps)
        score = F.softmax(x, dim=dim)
        return score

    def spatial_agg(self, x, mask=None, metric='cos'):
        assert(len(x.size()) == 4), x.size()
        N, _, H, W = x.size()
        if metric == 'cos':
            sim_mat = self.compute_sim_mat(x, x.clone())
        elif metric == 'kl':
            sim_mat = self.compute_sim_mat_kl(x, x.clone())
        else:
             raise NotImplementedError

        if metric == 'cos':
            sim_mat = self.scoring(sim_mat)
        else:
            sim_mat = F.softmax(sim_mat, dim=2)

        x = torch.matmul(x.view(N, -1, H*W), sim_mat.transpose(1, 2)).view(N, -1, H, W)
        return sim_mat, x

    def forward(self, x1, x2,fkp1,fkp2): ##x1#(5,4096*8,1,K) #fkp(N, 1,800, 2)
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        sim_mat_12, sim_mat_21, normalized_x, normalized_ref,sim_mat = self.build_correlation(x1, x2, metric=self.metric)#N,200,200
        sim, indices, max_indices,mid_indices,index_valid,index_valid2,mask_zone,d,mask_sim_mat,associated_sim,max_sim = self.cycle_associate(sim_mat_12, sim_mat_21, fkp1,fkp2)#N,200,1
        return indices, max_indices,mid_indices,index_valid,index_valid2,sim_mat_12,sim_mat_21,mask_zone, normalized_x, normalized_ref,d,mask_sim_mat,associated_sim,max_sim,sim_mat

