import csv
import os
import numpy as np
import pandas as pd
import pdb
# path='/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/'
path='/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                fileList.append(os.path.join(root, fileName))
    return fileList
nums=[]
with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'evaluation':
            nums.append(num)
# print(nums)
artres=[]
mrtres=[]
maxrtres=[]
for validate_num in range(251):
    csvfilespath=test_findfile(path,'.csv',str(nums[validate_num])+'_')
    # print(csvfilespath)
    # pdb.set_trace()
    if len(csvfilespath)!=2:
        print(validate_num)
    else:
        lmk = pd.read_csv(csvfilespath[0])
        lmk = np.array(lmk)
        lmk = lmk[:, [2, 1]]
        lmk2 = pd.read_csv(csvfilespath[1])
        lmk2 = np.array(lmk2)
        lmk2 = lmk2[:, [2, 1]]
        artre=np.mean(np.sqrt(np.sum(np.square(lmk-lmk2),axis=1)))/1024/np.sqrt(2)
        mrtre=np.median(np.sqrt(np.sum(np.square(lmk-lmk2),axis=1)))/1024/np.sqrt(2)
        maxrtre=np.amax(np.sqrt(np.sum(np.square(lmk-lmk2),axis=1)))/1024/np.sqrt(2)
        artres.append(artre)
        mrtres.append(mrtre)
        maxrtres.append(maxrtre)
print('aa={},aa_std={},am={},am_std={},ma={},mm={},amax={},amax_std={},mmax={}'.format(np.mean(artres),np.std(artres),np.mean(mrtres),np.std(mrtres),np.median(artres),np.median(mrtres),np.mean(maxrtres),np.std(maxrtres),np.median(maxrtres)))