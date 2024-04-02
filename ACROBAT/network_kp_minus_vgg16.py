import os
import pdb
import sys 
import numpy as np
import sys
import csv
from skimage import io
import pandas as pd
from numpy import *
import skimage
import scipy.io as scio
import matplotlib.pyplot as plt
count=0
data_path = r"/data/wxy/Pixel-Level-Cycle-Association-main/data/"
network_kp_path='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98/'
csvfilepath='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'
if not os.path.exists(csvfilepath):
    os.mkdir(csvfilepath)
name_as_num_path='/data/wxy/association/Maskflownet_association/kps/a0cAug30_3356_img2s_key_points_0.95_0.98_name_as_num/'
vgg16_kp_path1='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.028_0.004_0.8_01norm/'
vgg16_kp_path2='/data/wxy/Pixel-Level-Cycle-Association-main/output/kps_resnet/vgg16_features_ORB16s8_fc6_10_0.2_15_0.2_20_0.2_25_0.2_30_0.2_rotate8_0.99_0.06_0.08_0.01_0.75_01norm_for_large_displacement/'
prep_path1='/data/wxy/Pixel-Level-Cycle-Association-main/data/512after_affine/'

def appendimages(im1, im2):
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
######################################remove the same kps based on the name_as_num
with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'training':
            flmk = str(num)+'_1.csv'
            try:
                lmk = pd.read_csv(os.path.join(vgg16_kp_path2, flmk))
                lmk = np.array(lmk)
                lmk = lmk[:, [2, 1]]
            except:
                try:
                    lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    lmk2 = np.array(lmk2)
                    lmk2 = lmk2[:, [2, 1]]
                    lmk = lmk2
                except:
                    lmk =np.array([])
            else:
                try:
                    lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    lmk2 = np.array(lmk2)
                    lmk2 = lmk2[:, [2, 1]]
                    lmk = np.pad(lmk, lmk2, "constant")
                except:
                    pass
                else:
                    pass

            try:
                lmk_net = pd.read_csv(os.path.join(name_as_num_path, flmk))
                lmk_net = np.array(lmk_net)
                lmk_net = lmk_net[:, [2, 1]]
            except:
                lmk_net =np.array([])
            else:
                pass
            
            delete1=[]
            delete2=[]
            list_lmk=lmk.tolist()
            list_lmk_net=lmk_net.tolist()
            
            
            
            
            
            flmk = str(num)+'_2.csv'
            try:
                lmk = pd.read_csv(os.path.join(vgg16_kp_path2, flmk))
                lmk = np.array(lmk)
                lmk = lmk[:, [2, 1]]
            except:
                try:
                    lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    lmk2 = np.array(lmk2)
                    lmk2 = lmk2[:, [2, 1]]
                    lmk = lmk2
                except:
                    lmk =np.array([])
            else:
                try:
                    lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    lmk2 = np.array(lmk2)
                    lmk2 = lmk2[:, [2, 1]]
                    lmk = np.pad(lmk, lmk2, "constant")
                except:
                    pass
                else:
                    pass

            try:
                lmk_net = pd.read_csv(os.path.join(name_as_num_path, flmk))
                lmk_net = np.array(lmk_net)
                lmk_net = lmk_net[:, [2, 1]]
            except:
                lmk_net =np.array([])
            else:
                pass
            
            list_lmk2=lmk.tolist()
            list_lmk_net2=lmk_net.tolist()
            
            if len(list_lmk_net)!=0 and len(list_lmk)!=0:
                for i in range (len(list_lmk_net)):
                    if list_lmk_net[i] in list_lmk:
                        delete1.append(list_lmk_net[i])
                        delete2.append(list_lmk_net2[i])
            for i in range (len(delete1)):
                list_lmk_net.remove(delete1[i])
                list_lmk_net2.remove(delete2[i])
            if len(list_lmk_net)>0:
                name = ['X', 'Y']
                lmk1=np.asarray(list_lmk_net)
                lmk1=lmk1[:,[1,0]]
                outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                outlmk1.to_csv(csvfilepath+str(num)+'_1.csv')
                lmk1=np.asarray(list_lmk_net2)
                lmk1=lmk1[:,[1,0]]#.transpose(0,1)
                outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                outlmk1.to_csv(csvfilepath+str(num)+'_2.csv')

# ######################################update the csv name
# with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'training':
            # count=count+1
            # flmk = str(num)+'_1.csv'
            # fkp=str(count)+'_1.csv'

            # try:
                # lmk_net = pd.read_csv(os.path.join(network_kp_path, fkp))
                # lmk_net = np.array(lmk_net)
                # lmk_net = lmk_net[:, [2, 1]]
                # list_lmk_net=lmk_net.tolist()
                # name = ['X', 'Y']
                # lmk1=np.asarray(list_lmk_net)
                # lmk1=lmk1[:,[1,0]]
                # outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                # outlmk1.to_csv(name_as_num_path+str(num)+'_1.csv')
            # except:
                # pass
            # else:
                # pass
            
            # flmk = str(num)+'_2.csv'
            # fkp=str(count)+'_2.csv'
            
            # try:
                # lmk_net = pd.read_csv(os.path.join(network_kp_path, fkp))
                # lmk_net = np.array(lmk_net)
                # lmk_net = lmk_net[:, [2, 1]]
                # list_lmk_net=lmk_net.tolist()
                # name = ['X', 'Y']
                # lmk1=np.asarray(list_lmk_net)
                # lmk1=lmk1[:,[1,0]]
                # outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                # outlmk1.to_csv(name_as_num_path+str(num)+'_2.csv')
            # except:
                # pass
            # else:
                # pass

# ######################################remove the same kps
# with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'training':
            # count=count+1
            # flmk = str(num)+'_1.csv'
            # fkp=str(count)+'_1.csv'
            # try:
                # lmk = pd.read_csv(os.path.join(vgg16_kp_path2, flmk))
                # lmk = np.array(lmk)
                # lmk = lmk[:, [2, 1]]
            # except:
                # try:
                    # lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    # lmk2 = np.array(lmk2)
                    # lmk2 = lmk2[:, [2, 1]]
                    # lmk = lmk2
                # except:
                    # lmk =np.array([])
            # else:
                # try:
                    # lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    # lmk2 = np.array(lmk2)
                    # lmk2 = lmk2[:, [2, 1]]
                    # lmk = np.pad(lmk, lmk2, "constant")
                # except:
                    # pass
                # else:
                    # pass

            # try:
                # lmk_net = pd.read_csv(os.path.join(network_kp_path, fkp))
                # lmk_net = np.array(lmk_net)
                # lmk_net = lmk_net[:, [2, 1]]
            # except:
                # lmk_net =np.array([])
            # else:
                # pass
            
            # delete1=[]
            # delete2=[]
            # list_lmk=lmk.tolist()
            # list_lmk_net=lmk_net.tolist()
            
            
            
            
            
            # flmk = str(num)+'_2.csv'
            # fkp=str(count)+'_2.csv'
            # try:
                # lmk = pd.read_csv(os.path.join(vgg16_kp_path2, flmk))
                # lmk = np.array(lmk)
                # lmk = lmk[:, [2, 1]]
            # except:
                # try:
                    # lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    # lmk2 = np.array(lmk2)
                    # lmk2 = lmk2[:, [2, 1]]
                    # lmk = lmk2
                # except:
                    # lmk =np.array([])
            # else:
                # try:
                    # lmk2 = pd.read_csv(os.path.join(vgg16_kp_path1, flmk))
                    # lmk2 = np.array(lmk2)
                    # lmk2 = lmk2[:, [2, 1]]
                    # lmk = np.pad(lmk, lmk2, "constant")
                # except:
                    # pass
                # else:
                    # pass

            # try:
                # lmk_net = pd.read_csv(os.path.join(network_kp_path, fkp))
                # lmk_net = np.array(lmk_net)
                # lmk_net = lmk_net[:, [2, 1]]
            # except:
                # lmk_net =np.array([])
            # else:
                # pass
            
            # list_lmk2=lmk.tolist()
            # list_lmk_net2=lmk_net.tolist()
            
            # if len(list_lmk_net)!=0 and len(list_lmk)!=0:
                # for i in range (len(list_lmk_net)):
                    # if list_lmk_net[i] in list_lmk:
                        # delete1.append(list_lmk_net[i])
                        # delete2.append(list_lmk_net2[i])
            # for i in range (len(delete1)):
                # list_lmk_net.remove(delete1[i])
                # list_lmk_net2.remove(delete2[i])
            # if len(list_lmk_net)>0:
                # name = ['X', 'Y']
                # lmk1=np.asarray(list_lmk_net)
                # lmk1=lmk1[:,[1,0]]
                # outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                # outlmk1.to_csv(csvfilepath+str(num)+'_1.csv')
                # lmk1=np.asarray(list_lmk_net2)
                # lmk1=lmk1[:,[1,0]]#.transpose(0,1)
                # outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                # outlmk1.to_csv(csvfilepath+str(num)+'_2.csv')
# ######################################show the images
# with open(os.path.join(data_path, "matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'training':
            # count=count+1
            # flmk = str(num)+'_1.csv'
            # fimg = str(num)+'_1.jpg'
            # im_temp1 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
            # try:
                # lmk1 = pd.read_csv(os.path.join(csvfilepath, flmk))
                # lmk1 = np.array(lmk1)
                # lmk1 = lmk1[:, [2, 1]]
            # except:
                # lmk1 =np.array([])
            # else:
                # pass
            
            # flmk = str(num)+'_2.csv'
            # fimg = str(num)+'_2.jpg'
            # im_temp2 = io.imread(os.path.join(prep_path1, fimg), as_gray=True)
            # try:
                # lmk2 = pd.read_csv(os.path.join(csvfilepath, flmk))
                # lmk2 = np.array(lmk2)
                # lmk2 = lmk2[:, [2, 1]]
            # except:
                # lmk2 =np.array([])
            # else:
                # pass
            # lmk1=lmk1.tolist()
            # lmk2=lmk2.tolist()
            
            # im1=appendimages(im_temp1,im_temp2)
            # if len(lmk1)>0:
                # plt.figure()
                # plt.imshow(im1)
                # for i in range (len(lmk1)):
                    # plt.plot([lmk1[i][1],lmk2[i][1]+512],[lmk1[i][0],lmk2[i][0]], '#FF0033',linewidth=0.5)
                # plt.savefig('/data/wxy/association/Maskflownet_association/images/a0cAug30_3356_img2s_key_points_0.95_0.98_more_than_vgg16/'+str(num)+'.jpg',dpi=600)
                # plt.close()
