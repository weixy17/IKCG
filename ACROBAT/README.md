# Unsupervised Histological Image Registration Network Guided by Keypoint Correspondences Based on Learnable Deep Features With Iterative Training

By Xingyue Wei, Lin Ge, Lijie Huang, Jianwen Luo, and Yan Xu.

[[arXiv]](https://arxiv.org/pdf/2003.10955.pdf) [[ResearchGate]](https://www.researchgate.net/publication/340115724)

```
@inproceedings{Wei2024IPCG,
  author = Wei, Xingyue and Ge, Lin and Huang, Lijie and Luo, Jianwen and Xu, Yan},
  title = {Unsupervised Histological Image Registration Network Guided by Keypoint Correspondences Based on Learnable Deep Features With Iterative Training},
  booktitle = {IEEE Transactions on Medical Imaging},
  year = {2024}
}
```

## Introduction


Histological image registration is a fundamental task in histological image analysis. It can be considered as multimodal registration because of huge appearance differences due to multi-staining. Unsupervised deep learning-based registration methods are attractive because they avoid laborious annotations. Keypoint correspondences, i.e., matched keypoint pairs, have recently been introduced to guide the unsupervised methods to handle such a multimodal registration task. Keypoint correspondences rely on accurate feature descriptors. Imprecise descriptors such as handcrafted feature descriptors may generate numerous outliers, thus hampering unsupervised methods. Fortunately, deep features extracted from deep learning (DL) networks are more discriminative than handcrafted features, benefiting from the deep and hierarchical nature of DL networks. With networks trained on specific datasets, learnable deep features revealing unique information for specific datasets can be extracted. This paper proposes an iterative keypoint correspondence-guided (IPCG) unsupervised DL network for histological image registration. Deep features and learnable deep features are introduced as keypoint descriptors to automatically establish accurate keypoint correspondences, the distances between which are used as loss functions to train the registration network. An iterative training strategy is adopted to jointly train the registration network and optimize learnable deep features. Learnable deep features optimized with iterative training are accurate enough to establish correct correspondences with large displacement, thus solving the large displacement problem. For more details, please refer to our [paper](https://arxiv.org/pdf/2003.10955.pdf).

This repository includes:

- Codes for IPCG using Python and MXNet; and
- Pretrained models.

Code has been tested with Python 3.6 and MXNet 1.9.0.

## Datasets

We use ANHIR and ACROBAT datasets:

- [ANHIR](https://anhir.grand-challenge.org/)
- [ACROBAT](https://acrobat.grand-challenge.org/)


## Training

The following script is for training:

<center>

| # | Methods         | Training Dataset         | Validation Dataset    | Command Line |
|---|---|---|---|---|
| 1 | D-PCG | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_DFS_SFG_multiscale512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 4` |
| 2 | LD-PCG | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_LFS_SFG_multiscale512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1 --gkps` |
| 3 | PCG | ACROBAT training dataset | ACROBAT validation dataset  | `python TMI_rebuttle_main_PCG_multiscale_512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c e5fFeb21-1113:3641 --clear_steps --weight 1000 --batch 1 --gkps` |
| 4 | IPCG   | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_PCG_multiscale_512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 0 -c 0c5Feb25-1003:7 --clear_steps --weight 1000 --batch 1` |

</center>

## Pretrained Models

Pretrained models for IPCG on ACROBAT dataset are given (see `./weights/`).

## Inferring

