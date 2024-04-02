import os
import pdb
import sys 
sys.path.append('..') 
import numpy as np
import csv
import pandas as pd
from numpy import *
# orb_kp_path="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.5_15_0_20_0.25_25_0_30_0.25_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
# orb_kp_path="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_multiscale/"
# orb_kp_path="/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_rotate16_0.98_0.95_corrected/"
# orb_kp_path="/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale/"
orb_kp_path="/data/wxy/association/Maskflownet_association_1024/kps/LFS_SFG_multiscale_kps_1024/"
def LoadANHIR(data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"):##"/data3/gl/wxy_data/"
    count=-1
    with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if reader.line_num == 1:
                continue
            num = int(row[0])
            if row[5] == 'training':
                count=count+1
                if num>=93 and num<=129:
                
                    print('num={}'.format(num))
                    fimg = str(num)+'_1.jpg'
                    flmk = str(num)+'_1.csv'
                    fkp=str(num)+'_1'
                    lmk1_orb = pd.read_csv(orb_kp_path+flmk)
                    
                    lmk1_orb = np.array(lmk1_orb)
                    lmk1_orb = lmk1_orb[:, [2, 1]].tolist()
                    

                    fimg = str(num) + '_2.jpg'
                    flmk = str(num) + '_2.csv'
                    fkp=str(num)+'_2'
                    lmk2_orb = pd.read_csv(orb_kp_path+flmk)
                    lmk2_orb = np.array(lmk2_orb)
                    lmk2_orb = lmk2_orb[:, [2, 1]].tolist()
                    
                    name = ['X', 'Y']
                    lmk1=np.asarray(lmk1_orb)
                    lmk1=lmk1[:,[1,0]]
                    lmk2=np.asarray(lmk2_orb)
                    lmk2=lmk2[:,[1,0]]
                    outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                    outlmk1.to_csv(orb_kp_path+str(num)+'_2.csv')
                    outlmk2 = pd.DataFrame(columns=name, data=lmk2)
                    outlmk2.to_csv(orb_kp_path+str(num)+'_1.csv')
    return 0
if __name__ == '__main__':
    _=LoadANHIR()