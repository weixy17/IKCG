# from __future__ import print_function, division
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.bernoulli import Bernoulli
import random
import torch.distributed as dist


class AssociationLoss(nn.Module):
    def __init__(self, metric='cos', spagg=True, spagg_alpha=0.5, asso_topk=1,
            print_info=False):
        super(AssociationLoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        self.metric = metric
        self.spagg = spagg
        self.spagg_alpha = spagg_alpha
        self.asso_topk = asso_topk
        self.print_info = print_info

    def compute_sim_mat(self, x, ref):
        assert(len(x.size()) == 4), x.size()
        N, C, H, W = x.size()
        _, _, Hr, Wr = ref.size()
        assert(x.shape[:2] == ref.shape[:2]), ref.size()
        
        normalized_x = x.reshape(-1, C, H*W).transpose(1, 2)#F.normalize(x.reshape(-1, C, H*W).transpose(1, 2), dim=2)
        reshaped_ref = ref.reshape(-1, C, Hr*Wr)
        normalized_ref = reshaped_ref#F.normalize(reshaped_ref, dim=1)
        sim_mat = torch.matmul(normalized_x, normalized_ref)
        return sim_mat

    def compute_sim_mat_kl(self, x1, x2):
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        eps = 1e-10
        log_x1 = torch.log(x1+eps)
        log_x2 = torch.log(x2+eps)
        neg_ent = torch.sum(x1 * log_x1, dim=1).reshape(N, -1, 1)
        cross_ent = -1.0 * torch.matmul(x1.reshape(N, -1, H*W).transpose(1, 2), log_x2.reshape(N, -1, H2*W2))
        kl = neg_ent + cross_ent
        return -1.0 * kl

    def build_correlation(self, x1, x2, metric='cos'):
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        if metric == 'cos':
            sim_mat_12 = self.compute_sim_mat(x1, x2)
            sim_mat_21 = sim_mat_12.transpose(1, 2)

        elif metric == 'kl':
            sim_mat_12 = self.compute_sim_mat_kl(x1, x2)
            sim_mat_21 = self.compute_sim_mat_kl(x2, x1)

        else:
            raise NotImplementedError

        # sim_mat_12 = self.scoring(sim_mat_12)
        # sim_mat_21 = self.scoring(sim_mat_21)
        return sim_mat_12, sim_mat_21

    def associate(self, sim_mat, topk=1):
        indices = torch.topk(sim_mat, dim=2, k=topk).indices.detach()
        sim = torch.topk(sim_mat, dim=2, k=topk).values
        return indices, sim

    def associate_gt(self, gt, indices):
        N, H, W = gt.size()
        K = indices.size(2)
        gt = gt.reshape(N, -1, 1).expand(N, -1, K)
        end_gt = gt

        associated_gt = torch.gather(end_gt, 1, indices)
        gt = (gt == associated_gt).type(torch.cuda.FloatTensor).detach()
        return gt.reshape(N, H, W, K)
       
    def cycle_associate(self, sim_mat_12, sim_mat_21, topk=1):
        N, Lh, Lw = sim_mat_12.size()
        mid_indices, associated_sim = self.associate(sim_mat_12)
        pdb.set_trace()
        N, Lh, K = mid_indices.size()
        reassociate = torch.max(sim_mat_21, dim=2)
        max_indices = reassociate.indices.unsqueeze(-1).expand(N, -1, K)
        max_sim = reassociate.values.unsqueeze(-1).expand(N, -1, K)
        indices = torch.gather(max_indices, 1, mid_indices)
        reassociated_sim = torch.gather(max_sim, 1, mid_indices)
        return associated_sim * reassociated_sim, indices, mid_indices,max_indices

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

        x = torch.matmul(x.reshape(N, -1, H*W), sim_mat.transpose(1, 2)).reshape(N, -1, H, W)
        return sim_mat, x

    def eval_correct_ratio(self, select_mask, covered_indices, gt_T, gt):
        N, H, W, K = select_mask.size()
        select_gt_T = torch.gather(gt_T[0].reshape(-1, 1).expand(-1, K), 0, covered_indices[0]).reshape(H, W, K)
        select_mask = select_mask[0] * (select_gt_T != 255)
        select_gt_T = torch.masked_select(select_gt_T, select_mask)
        select_gt = torch.masked_select(gt[0].unsqueeze(-1).expand(H, W, K), select_mask)
        if select_gt.numel() == 0:
            return -1.0
        else:
            return  1.0 * torch.sum(select_gt_T == select_gt).item() / (select_gt.numel())
 
    def forward(self, x1, x2): 
        N, _, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        assert(x1.shape[:2] == x2.shape[:2]), x2.size()
        if self.spagg:
            alpha = self.spagg_alpha
            assert(alpha < 1.0 and alpha > 0.0)
            agg_x2 = self.spatial_agg(x2, metric=self.metric)[-1]
            x2 = (1.0 - alpha) * x2 + alpha * agg_x2
                
        sim_mat_12, sim_mat_21 = self.build_correlation(x1, x2, metric=self.metric)
        pdb.set_trace()
        sim, indices, mid_indices,max_indices = self.cycle_associate(sim_mat_12, sim_mat_21, topk=self.asso_topk)
        return indices, mid_indices,max_indices
import scipy.io as scio
import pdb
if __name__=='__main__':

    loss=AssociationLoss(metric='cos', 
                spagg=False, spagg_alpha=0, 
                asso_topk=1, print_info=False)
    data=scio.loadmat("/data3/wxy/Association/images/image_4_2/1_0.mat")
    input1=torch.from_numpy(data['input1'])
    input2=torch.from_numpy(data['input2'])
    # pdb.set_trace()
    indices, mid_indices,max_indices=loss(input1,input2)
    scio.savemat('/data3/wxy/Association/images/image_4_2/test.mat',{'indices':indices.numpy(),'mid_indices':mid_indices.numpy(),'max_indices':max_indices.numpy()})
    pdb.set_trace()