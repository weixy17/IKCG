import os
import pandas as pd
import numpy as np
from affine_ransac import Ransac
import csv
import pdb
from skimage import io
import matplotlib.pyplot as plt
import shutil
data_path="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/train_after_affine_4096_to1024/"
kpspath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9/"
savepath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_images/baseline_ite2_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
if not os.path.exists(savepath):
    os.mkdir(savepath)
csvfilepath="/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/rebuttle_kps/baseline_ite2_multiscale_kps1024_0.97_0.9_delete_near_ransac3/"
if not os.path.exists(csvfilepath):
    os.mkdir(csvfilepath)
def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)


files=os.listdir(data_path)
for file in files:
    if file.split('_')[1]=='HE':
        continue
    num = file.split('.')[0]
    fimg = file
    flmk = str(num)+'_1.csv'
    im_temp1 = io.imread(os.path.join(data_path, fimg), as_gray=True)
    img1=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
    try:
        lmk1 = pd.read_csv(kpspath+flmk)
    except:
        continue
    lmk1_ori = np.array(lmk1)
    lmk1 = lmk1_ori[:, [2, 1]].astype('int').reshape(-1,2)
    
    fimg = file.split('_')[0]+'_HE_train.jpg'
    flmk = str(num) + '_2.csv'
    im_temp1 = io.imread(os.path.join(data_path, fimg), as_gray=True)
    img2=np.concatenate((np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0),np.expand_dims(im_temp1,0)),0)
    lmk2 = pd.read_csv(kpspath+flmk)
    lmk2_ori = np.array(lmk2)
    lmk2 = lmk2_ori[:, [2, 1]].astype('int').reshape(-1,2)
    
    delete_index=[]
    try:
        index=np.argsort(lmk1_ori[:,3])
    except:
        index=np.array(range(0,lmk1_ori.shape[0]))
    for temp in range(0,lmk1_ori.shape[0]):
        lmk1_temp=lmk1[index[temp],:]
        lmk1_comparison=lmk1[index[temp+1:],:]
        dis1=((lmk1_temp-lmk1_comparison)**2).sum(1)**0.5
        if np.shape(np.where(dis1<5)[0])[0]>0:
            delete_index.extend([index[temp]])
        else:
            lmk2_temp=lmk2[index[temp],:]
            lmk2_comparison=lmk2[index[temp+1:],:]
            dis2=((lmk2_temp-lmk2_comparison)**2).sum(1)**0.5
            if np.shape(np.where(dis2<5)[0])[0]>0:
                delete_index.extend([index[temp]])
                
    inliers=[[i for i in range(0,lmk1.shape[0]) if i not in delete_index]]
    lmk1=lmk1[inliers[0],:]
    lmk2=lmk2[inliers[0],:]
    lmk1_ori=lmk1_ori[inliers[0],:]
    lmk2_ori=lmk2_ori[inliers[0],:]
    
    
    
    
    if lmk1.shape[0]<=5:
        inliers=None#[]#[[i for i in range(0,lmk1.shape[0])]]
    else:
        if 1:#num not in [0,1,2]:
            thresholds=0.5##0.5##
            rs = Ransac(K=int(lmk1.shape[0]), threshold=thresholds)
            A_rsc, t_rsc, inliers,residual = rs.ransac_fit(np.transpose(lmk1), np.transpose(lmk2))
        else:
            inliers=[[i for i in range(0,lmk1.shape[0])]]
            
    if not(inliers is None):
        lmk1_new=lmk1[inliers[0],:]
        lmk2_new=lmk2[inliers[0],:]
        H,W=img1.shape[1],img1.shape[2]
        im1=appendimages(img1[0,:,:],img2[0,:,:])
        plt.figure()
        plt.imshow(im1)
        plt.title(str(num)+'key_points='+str(len(inliers[0])))
        plt.plot([lmk1_new[:,1],lmk2_new[:,1]+H],[lmk1_new[:,0],lmk2_new[:,0]], '#FF0033',linewidth=0.5)
        plt.savefig(savepath+str(num)+'key_points='+str(len(inliers[0]))+'.jpg', dpi=600)
        plt.close()
        lmk1=lmk1_ori[inliers[0],1:]
        lmk2=lmk2_ori[inliers[0],1:]
        
        name = ['X', 'Y','W']
        # name = ['X', 'Y']
        outlmk1 = pd.DataFrame(columns=name, data=lmk1)
        outlmk1.to_csv(csvfilepath+str(num)+'_1.csv')
        outlmk2 = pd.DataFrame(columns=name, data=lmk2)
        outlmk2.to_csv(csvfilepath+str(num)+'_2.csv')
    else:
        #pass
        H,W=img1.shape[1],img1.shape[2]
        im1=appendimages(img1[0,:,:],img2[0,:,:])
        plt.figure()
        plt.imshow(im1)
        plt.title(str(num)+'key_points=0')
        plt.savefig(savepath+str(num)+'key_points=0.jpg', dpi=600)
        plt.close()



















# import os
# import pdb
# import sys 
# sys.path.append('..') 
# import numpy as np
# import csv
# import pandas as pd
# from numpy import *
# from skimage import io
# import matplotlib.pyplot as plt
# imgpath = "/ssd1/wxy/Pixel-Level-Cycle-Association-main/data/1024after_affine"
# # orb_kp_path="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC/"
# # errors={'28':[1]}
# # savepath="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/images/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024_delete_near_RANSAC/"
# orb_kp_path="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/kps_vgg16/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"
# errors={'28':[1]}
# savepath="/ssd1/wxy/Pixel-Level-Cycle-Association-main/rebuttle_output/images/vgg16_features_ORB16s8_fc6_20_0.2_30_0.2_40_0.2_50_0.2_60_0.2_rotate8_0.99_0.028_0.004_0.8_01norm_same_pixels_multiscale_1024/"


# def appendimages(im1, im2):
    # """ Return a new image that appends the two images side-by-side. """
    # # select the image with the fewest rows and fill in enough empty rows
    # rows1 = im1.shape[0]
    # rows2 = im2.shape[0]
    # if rows1 < rows2:
        # im1 = np.concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    # elif rows1 > rows2:
        # im2 = np.concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # # if none of these cases they are equal, no filling needed.
    # return np.concatenate((im1, im2), axis=1)


# def LoadANHIR():
    # count=-1
    # with open("/ssd1/wxy/association/For_submit_wxy/matrix_sequence_manual_validation_with_preprocessing_paras.csv", newline="") as f:
        # reader = csv.reader(f)
        # for row in reader:
            # if reader.line_num == 1:
                # continue
            # num = int(row[0])
            # if row[5] == 'evaluation':
                # count=count+1
                # if str(num) not in errors:
                    # continue
                # delete_index=errors[str(num)]
                # delete_index.sort(key=lambda x:int(x))
                # delete1=[]
                # delete2=[]
                
                # print('num={}'.format(num))
                # fimg = str(num)+'_1.jpg'
                # flmk = str(num)+'_1.csv'
                # fkp=str(num)+'_1'
                # lmk1_orb = pd.read_csv(orb_kp_path+flmk)
                # lmk1_orb = np.array(lmk1_orb)
                # lmk1_orb = lmk1_orb[:, [2, 1]].tolist()
                # img1 = io.imread(os.path.join(imgpath, fimg), as_gray=True)
                
                # fimg = str(num) + '_2.jpg'
                # flmk = str(num) + '_2.csv'
                # fkp=str(num)+'_2'
                # lmk2_orb = pd.read_csv(orb_kp_path+flmk)
                # lmk2_orb = np.array(lmk2_orb)
                # lmk2_orb = lmk2_orb[:, [2, 1]].tolist()
                # img2 = io.imread(os.path.join(imgpath, fimg), as_gray=True)
                # pdb.set_trace()
                # for i in range(len(delete_index)-1,-1,-1):
                    # lmk1_orb.pop(delete_index[i])
                    # lmk2_orb.pop(delete_index[i])
                    # print('remove:{}'.format(delete_index[i]))
                # if len(lmk1_orb)==0 or len(lmk2_orb)==0:
                    # os.remove(orb_kp_path+str(num)+'_1.csv')
                    # os.remove(orb_kp_path+str(num)+'_2.csv')
                # else:
                    # name = ['X', 'Y']
                    # lmk1=np.asarray(lmk1_orb)
                    # lmk1=lmk1[:,[1,0]]
                    # lmk2=np.asarray(lmk2_orb)
                    # lmk2=lmk2[:,[1,0]]
                    # outlmk1 = pd.DataFrame(columns=name, data=lmk1)
                    # outlmk1.to_csv(orb_kp_path+str(num)+'_1.csv')
                    # outlmk2 = pd.DataFrame(columns=name, data=lmk2)
                    # outlmk2.to_csv(orb_kp_path+str(num)+'_2.csv')
                # im1=appendimages(img1,img2)
                # plt.figure()
                # plt.imshow(im1)
                # plt.title(str(num)+'key_points='+str(len(lmk1_orb)))
                # plt.plot([lmk1_orb[:,1],lmk2_orb[:,1]+im1.shape[1]],[lmk1_orb[:,0],lmk2_orb[:,0]], '#FF0033',linewidth=0.5)
                # plt.savefig(savepath+str(num)+'key_points='+str(len(lmk1_orb))+'.jpg', dpi=600)
                # plt.close()
                
    # return 0
# if __name__ == '__main__':
    # _=LoadANHIR()