
# ################################judge
# import os
# import pdb
# import numpy as np
# import pandas as pd
# path="/ssd1/wxy/association/Maskflownet_association_1024/dataset/1024after_affine_with_original_size/"
# path2="/ssd1/wxy/association/Maskflownet_association_1024/dataset/1024after_affine/"
# files=os.listdir(path)
# for filename in files:
    # if filename[-4:]=='.csv':
        # data = np.loadtxt(os.path.join(path,filename), delimiter=',', dtype=str, encoding='utf-8')
        # data2 = np.loadtxt(os.path.join(path2,filename), delimiter=',', dtype=str, encoding='utf-8')
        # judge=(data[:,0:3]!=data2)
        # if judge.sum()>0:
            # print(filename)
# ########################




import cv2
import numpy as np
import pdb
import skimage
from skimage import io
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import copy
import shutil
#import openpyxl
import xlrd
from PIL import Image
import pdb






basepath='/ssd1/wxy/association/For_submit_wxy/'


path=basepath+'original_size_and_change_size_record.csv'
data = np.loadtxt(path, delimiter=',', dtype=str, encoding='utf-8')
imgname=['_'.join(i.split('/')) for i in data[:,1].tolist()]
imgname=[i.split('.')[0]+'.jpg' for i in imgname]



workbook = xlrd.open_workbook(basepath+'manually_set.xlsx')
sheet1 = workbook.sheet_by_name('Sheet1')
nrows = sheet1.nrows
ncols = sheet1.ncols
changed_num=[]
matrix_seqs = []
for i in range(nrows):
    matrix_seq = []
    for j in range(ncols):
        matrix_seq.append(sheet1.cell(i, j).value)
    matrix_seqs.append(matrix_seq)
imgnames=['_'.join(i.split('/')) for i in np.array(matrix_seqs)[:,1].tolist()]
imgnames=[i.split('.')[0]+'.jpg' for i in imgnames]
sample_seq = pd.read_csv(basepath+'matrix_sequence_manual_validation.csv')
data_output = np.loadtxt(basepath+'matrix_sequence_manual_validation.csv', delimiter=',', dtype=str, encoding='utf-8')
data_output=np.concatenate((data_output,np.zeros([data_output.shape[0],20])),1)
sample_seq = np.array(sample_seq)[:, 1:]
name = ['X', 'Y']
for j in range(1, 482):
    if 1:
        img1name = sample_seq[j-1][0]  # source image
        img2name = sample_seq[j-1][2]  # target image
        index=imgnames.index(img1name)
        if index==295:
            pdb.set_trace()
        data_output[j,6]=np.array(matrix_seqs)[index,2]####updown
        data_output[j,7]=np.array(matrix_seqs)[index,3]######leftright###updown==1 + leftright==1   =====rotation180
        data_output[j,8]=np.array(matrix_seqs)[index,4]####enlarge
        data_output[j,9]=np.array(matrix_seqs)[index,5]####anti_clockwise90
        index=imgname.index(img1name)
        data_output[j,10]=data[imgname.index(img1name.split('_scale')[0]+'.jpg'),2]####maxh
        data_output[j,11]=data[imgname.index(img1name.split('_scale')[0]+'.jpg'),3]####maxw
        data_output[j,12]=data[index,2]####h
        data_output[j,13]=data[index,3]####w
        data_output[j,14]=data[index,4]######h_begin
        data_output[j,15]=data[index,5]#########w_begin
        
        index2=imgnames.index(img2name)
        data_output[j,16]=np.array(matrix_seqs)[index2,2]
        data_output[j,17]=np.array(matrix_seqs)[index2,3]
        data_output[j,18]=np.array(matrix_seqs)[index2,4]
        data_output[j,19]=np.array(matrix_seqs)[index2,5]
        index2=imgname.index(img2name)
        data_output[j,20]=data[imgname.index(img2name.split('_scale')[0]+'.jpg'),2]
        data_output[j,21]=data[imgname.index(img2name.split('_scale')[0]+'.jpg'),3]
        data_output[j,22]=data[index2,2]
        data_output[j,23]=data[index2,3]
        data_output[j,24]=data[index2,4]
        data_output[j,25]=data[index2,5]
df = pd.DataFrame(data_output)
df.to_csv(basepath+"matrix_sequence_manual_validation_with_preprocessing_paras.csv", index=False, header=False)
        
                           
                           
                            # if not matrix_seqs[i][4] == 0:
                                # cha = np.floor((int(imgsize[0]*matrix_seqs[i][4]) - imgsize[0])/2)
                                # temp=np.zeros((int(imgsize[0]+2*cha),int(imgsize[1]+2*cha)))
                                # temp2=np.zeros((int(imgsize[0]+2*cha),int(imgsize[1]+2*cha)))
                                # temp[int(cha):int(cha + imgsize[0]), int(cha):int(cha + imgsize[1])]=np.asarray(imgresult)
                                # temp2[int(cha):int(cha + imgsize[0]), int(cha):int(cha + imgsize[1])]=np.asarray(imgresult1_warpped)
                                # temp=Image.fromarray(temp)
                                # imgresult=temp.resize((int(imgsize[0]/matrix_seqs[i][4]), int(imgsize[1]/matrix_seqs[i][4])), Image.ANTIALIAS)
                                # imgresult1_warpped=temp2.resize((int(imgsize[0]/matrix_seqs[i][4]), int(imgsize[1]/matrix_seqs[i][4])), Image.ANTIALIAS)
                                # lmkresult=(lmk- imgsize[1]/2)/matrix_seqs[i][4]+imgsize[1]/2
                                # warpped_lmk1result=(warpped_lmk1- imgsize[1]/2)/matrix_seqs[i][4]+imgsize[1]/2

                            # imgresult = np.asarray(imgresult)
                            # imgresult1_warpped = np.asarray(imgresult1_warpped)
                            # if matrix_seqs[i][5] == 1:
                                # imgresult = np.rot90(np.rot90(np.rot90(imgresult)))
                                # imgresult1_warpped = np.rot90(np.rot90(np.rot90(imgresult1_warpped)))
                                # lmktemp2 = copy.deepcopy(lmkresult)
                                # lmktemp2=lmktemp2-imgsize[0]/2
                                # lmkresult[:,1]=lmktemp2[:,0]
                                # lmkresult[:,0]=-lmktemp2[:,1]
                                # lmkresult=lmkresult+imgsize[0]/2
                                
                                # warpped_lmk1resulttemp2 = copy.deepcopy(warpped_lmk1result)
                                # warpped_lmk1resulttemp2=warpped_lmk1resulttemp2-imgsize[0]/2
                                # warpped_lmk1result[:,1]=warpped_lmk1resulttemp2[:,0]
                                # warpped_lmk1result[:,0]=-warpped_lmk1resulttemp2[:,1]
                                # warpped_lmk1result=warpped_lmk1result+imgsize[0]/2
                            # skimage.io.imsave(os.path.join(outputfile, afterimg2name), imgresult.astype(np.uint8))
                            # skimage.io.imsave(os.path.join(outputfile, warppedimg1name), imgresult1_warpped.astype(np.uint8))
                            # name = ['X', 'Y']
                            # outlmk = pd.DataFrame(columns=name, data=lmkresult)
                            # output_prefix = os.path.join(outputfile, afterlmk2name)
                            # outlmk.to_csv(output_prefix)
                            # outlmk = pd.DataFrame(columns=name, data=warpped_lmk1result)
                            # output_prefix = os.path.join(outputfile, warppedlmk1name)
                            # outlmk.to_csv(output_prefix)
                            # print('{}evaluation save!'.format(afterimg2name))
    # else:
        # print('group:', j)
        # img1name = sample_seq[j][0]  # source image
        # img2name = sample_seq[j][2]  # target image
        # lmk1name = sample_seq[j][1]  # source image
        # lmk2name = sample_seq[j][3]  # target image
        # afterimg1name = str(j) + '_1.jpg'
        # afterimg2name = str(j) + '_2.jpg'
        # warppedimg1name = str(j) + '_1_warpped.jpg'
        # afterlmk1name = str(j) + '_1.csv'
        # warppedlmk1name = str(j) + '_1_warpped.csv'
        # for i in range(nrows):
            # matrix_seq_name = matrix_seqs[i][1][:-4]
            # matrix_seq_suffix = matrix_seqs[i][1][-3:]
            # if matrix_seq_suffix == 'jpg' or matrix_seq_suffix == 'png':
                # imgname = '_'
                # imgname = imgname.join(matrix_seq_name.split('/')) + '.jpg'
                # if imgname == img1name or imgname == img2name:
                    # if imgname == img1name:
                        # shutil.copy(os.path.join(imgfile, afterimg1name),os.path.join(outputfile, afterimg1name))
                        # shutil.copy(os.path.join(imgfile, afterlmk1name),os.path.join(outputfile, afterlmk1name))
                    # else:
                        
                        # warpped_lmk1 = pd.read_csv(os.path.join(imgfile, warppedlmk1name))
                        # warpped_lmk1 = np.array(warpped_lmk1)
                        # warpped_lmk1 = warpped_lmk1[:, [1, 2]]
                        # if matrix_seqs[i][2] == 0 and matrix_seqs[i][3] == 0 and matrix_seqs[i][4] == 0 and matrix_seqs[i][5] == 0:
                            # shutil.copy(os.path.join(imgfile, afterimg2name),os.path.join(outputfile, afterimg2name))
                            # shutil.copy(os.path.join(imgfile, warppedimg1name),os.path.join(outputfile, warppedimg1name))
                            # shutil.copy(os.path.join(imgfile, warppedlmk1name),os.path.join(outputfile, warppedlmk1name))
                            # continue
                        # else:
                            # img0 = Image.open(os.path.join(imgfile, afterimg2name))
                            # img1_warpped = Image.open(os.path.join(imgfile, warppedimg1name))
                            # imgsize = np.shape(np.asarray(img0))
                            # imgresult = copy.deepcopy(img0)
                            # imgresult1_warpped = copy.deepcopy(img1_warpped)
                            # warpped_lmk1result = copy.deepcopy(warpped_lmk1)

                            # if matrix_seqs[i][2] == 1 and matrix_seqs[i][3] == 0:
                                # print('Unnormalize situation, updown but not rightleft')
                            # if matrix_seqs[i][2] == 0 and matrix_seqs[i][3] == 1:
                                # print('Unnormalize situation, rightleft but not updown')
                            # if matrix_seqs[i][2] == 1 and matrix_seqs[i][3] == 1:
                                # imgresult = imgresult.transpose(Image.ROTATE_180)
                                # imgresult1_warpped = imgresult1_warpped.transpose(Image.ROTATE_180)
                                # warpped_lmk1result=imgsize[0]-warpped_lmk1result

                            # if not matrix_seqs[i][4] == 0:
                                # cha = np.floor((int(imgsize[0]*matrix_seqs[i][4]) - imgsize[0])/2)
                                # temp=np.zeros((int(imgsize[0]+2*cha),int(imgsize[1]+2*cha)))
                                # temp2=np.zeros((int(imgsize[0]+2*cha),int(imgsize[1]+2*cha)))
                                # temp[int(cha):int(cha + imgsize[0]), int(cha):int(cha + imgsize[1])]=np.asarray(imgresult)
                                # temp2[int(cha):int(cha + imgsize[0]), int(cha):int(cha + imgsize[1])]=np.asarray(imgresult1_warpped)
                                # temp=Image.fromarray(temp)
                                # temp2=Image.fromarray(temp2)
                                # imgresult=temp.resize((int(imgsize[0]/matrix_seqs[i][4]), int(imgsize[1]/matrix_seqs[i][4])), Image.ANTIALIAS)
                                # # pdb.set_trace()
                                # imgresult1_warpped=temp2.resize((int(imgsize[0]/matrix_seqs[i][4]), int(imgsize[1]/matrix_seqs[i][4])), Image.ANTIALIAS)
                                # warpped_lmk1result=(warpped_lmk1- imgsize[1]/2)/matrix_seqs[i][4]+imgsize[1]/2

                            # imgresult = np.asarray(imgresult)
                            # imgresult1_warpped = np.asarray(imgresult1_warpped)
                            # if matrix_seqs[i][5] == 1:
                                # imgresult = np.rot90(np.rot90(np.rot90(imgresult)))
                                # imgresult1_warpped = np.rot90(np.rot90(np.rot90(imgresult1_warpped)))
                                
                                # warpped_lmk1resulttemp2 = copy.deepcopy(warpped_lmk1result)
                                # warpped_lmk1resulttemp2=warpped_lmk1resulttemp2-imgsize[0]/2
                                # warpped_lmk1result[:,1]=warpped_lmk1resulttemp2[:,0]
                                # warpped_lmk1result[:,0]=-warpped_lmk1resulttemp2[:,1]
                                # warpped_lmk1result=warpped_lmk1result+imgsize[0]/2
                            # skimage.io.imsave(os.path.join(outputfile, afterimg2name), imgresult.astype(np.uint8))
                            # skimage.io.imsave(os.path.join(outputfile, warppedimg1name), imgresult1_warpped.astype(np.uint8))
                            # name = ['X', 'Y']
                            # outlmk = pd.DataFrame(columns=name, data=warpped_lmk1result)
                            # output_prefix = os.path.join(outputfile, warppedlmk1name)
                            # outlmk.to_csv(output_prefix)
                            # print('{}evaluation save!'.format(afterimg2name))

