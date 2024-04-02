import scipy.io as scio
import os
import numpy as np
import pdb
import shutil
import csv
outputpath='/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'
path='/data/wxy/association/Maskflownet_association_1024/kps_for_submit/evaluation_mean_rTRE/'
folderpath='/data/wxy/association/Maskflownet_association_1024/kps_for_submit/'
all_data=np.zeros([len(os.listdir(path)),251])
filenames=os.listdir(path)
for count,filename in enumerate(filenames):
    # print(filename)
    data=scio.loadmat(path+filename)
    data=data['rtre']
    all_data[count,:]=data
pdb.set_trace()
index=np.argmin(all_data,axis=0)
foldernames=[temp.split('.m')[0]+'_without_dist' for temp in filenames]
nums=[]
with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'evaluation':
            nums.append(num)
print(len(nums))
for count,i in enumerate(index.tolist()):
    shutil.copy(folderpath+foldernames[i]+'/'+str(nums[count])+'_1.jpg',outputpath)
    shutil.copy(folderpath+foldernames[i]+'/'+str(nums[count])+'_2.jpg',outputpath)
    shutil.copy(folderpath+foldernames[i]+'/'+str(nums[count])+'_1_warpped.jpg',outputpath)
    shutil.copy(folderpath+foldernames[i]+'/'+str(nums[count])+'_1_warpped.csv',outputpath)
    shutil.copy(folderpath+foldernames[i]+'/'+str(nums[count])+'_1.csv',outputpath)