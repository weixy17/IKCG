# Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training

By Xingyue Wei, Lin Ge, Lijie Huang, Jianwen Luo, and Yan Xu.

[[arXiv]](https://arxiv.org/pdf/2003.10955.pdf) [[ResearchGate]](https://www.researchgate.net/publication/340115724)

```
@inproceedings{Wei2024IKCG,
  author = Wei, Xingyue and Ge, Lin and Huang, Lijie and Luo, Jianwen and Xu, Yan},
  title = {Unsupervised Non-rigid Histological Image Registration Guided by Keypoint Correspondences Based on Learnable Deep Features with Iterative Training},
  booktitle = {IEEE Transactions on Medical Imaging},
  year = {2024}
}
```

## Introduction


Histological image registration is a fundamental task in histological image analysis. It can be considered as multimodal registration because of huge appearance differences due to multi-staining. Unsupervised deep learning-based registration methods are attractive because they avoid laborious annotations. Keypoint correspondences, i.e., matched keypoint pairs, have recently been introduced to guide the unsupervised methods to handle such a multimodal registration task. Keypoint correspondences rely on accurate feature descriptors. Imprecise descriptors such as handcrafted feature descriptors may generate numerous outliers, thus hampering unsupervised methods. Fortunately, deep features extracted from deep learning (DL) networks are more discriminative than handcrafted features, benefiting from the deep and hierarchical nature of DL networks. With networks trained on specific datasets, learnable deep features revealing unique information for specific datasets can be extracted. This paper proposes an iterative keypoint correspondence-guided (IPCG) unsupervised DL network for histological image registration. Deep features and learnable deep features are introduced as keypoint descriptors to automatically establish accurate keypoint correspondences, the distances between which are used as loss functions to train the registration network. An iterative training strategy is adopted to jointly train the registration network and optimize learnable deep features. Learnable deep features optimized with iterative training are accurate enough to establish correct correspondences with large displacement, thus solving the large displacement problem. For more details, please refer to our paper.

This repository includes:

- Codes for IKCG using Python and MXNet; and
- Pretrained models.

Code has been tested with Python 3.6 and MXNet 1.9.0.

## Datasets

We use ANHIR and ACROBAT datasets:

- [ANHIR](https://anhir.grand-challenge.org/)
- [ACROBAT](https://acrobat.grand-challenge.org/)

## Pipeline
- 1. To generate keypoints based on ORB and match keypoint pairs based on fixed deep features, please go to IKCG\ACROBAT\Generate_kps_pairs_based_on_fixed_deep_features and run step1_ORB_generate_keypoints.py and step2_match_keypoint_pairs_multiscale.py
- 2. With keypoint pairs matched based on fixed deep feature, you can train F-KCG
- 3. With converged F-KCG, you can match new keypoint pairs based on learnable deep features. The command line is:
   `python TMI_rebuttle_main_L-KCG_multiscale512-gkps.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1 --gkps` 
- 4. Then you can train L-KCG and LF-KCG
- 5. You can go to IKCG\ACROBAT\DenseSIFTmap and run multiscale_256_512_1024siftflowmap_to_1024.m to generate dense SIFT feature maps, then you can train KCG.
- 6. With converged KCG, you can match new keypoint pairs based on learnable deep features. The command line is:
  `python TMI_rebuttle_main_KCG_multiscale_512-gkps.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1 --gkps` 
- 7. Then you can train IKCG for the first iteration.
- 8. By iteratively perform 6 and 7, you can get IKCG at different iterations.

## Training

The following script is for training:

<center>

| # | Methods         | Training Dataset         | Validation Dataset    | Command Line |
|---|---|---|---|---|
| 1 | Baseline | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_baseline512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 4` |
| 2 | F-KCG | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_F-KCG_multiscale512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 4` |
| 3 | L-KCG | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_L-KCG_multiscale512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1` |
| 4 | LF-KCG | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_LF-KCG_multiscale512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1` |
| 5 | KCG | ACROBAT training dataset | ACROBAT validation dataset  | `python TMI_rebuttle_main_KCG_multiscale_512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1`|
| 6 | IKCG   | ACROBAT training dataset  | ACROBAT validation dataset  | `python TMI_rebuttle_main_IKCG_multiscale_512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 0 -c 2afApr28-1544 --clear_steps --weight 1000 --batch 1` |

</center>

## Pretrained Models

Pretrained models for IKCG on ACROBAT dataset are given (see `./weights/`).

## Inferring

