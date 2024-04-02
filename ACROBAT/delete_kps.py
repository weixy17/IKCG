import scipy.io as scio
import numpy as np
import pandas as pd
import os
import shutil
def test_findfile(directory, fileType, file_prefix):
    fileList = []
    for root, subDirs, files in os.walk(directory):
        for fileName in files:
            if fileName.endswith(fileType) and fileName.startswith(file_prefix):
                # fileList.append(os.path.join(root, fileName))
                fileList.append(fileName)
    return fileList
validimgpath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57/"#######third submission
validfilenames=test_findfile(validimgpath,'.jpg','')
savekpspath="/ssd2/wxy/IPCG_Acrobat/data/data_for_deformable_network/valid_after_affine_4096_to512_all_kps_add1_0127-python_add_large57_delete_kps/"
if not os.path.exists(savekpspath):
    os.mkdir(savekpspath)
csvpath="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/Predicted_mask_for_Under_segmentation_val_pixel_10X_jpg_new.xlsx"
csvpath2="/ssd2/wxy/IPCG_Acrobat/data/annotated_files/annotated_kps_validation.csv"
csvdata =np.array(pd.read_excel(csvpath,header=None,index_col=None))
csvdata2 =np.array(pd.read_excel(csvpath2,header=None,index_col=None))
idxs=scio.loadmat("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/idx.mat")['data']
valid_count=-1
for validfilename in validfilenames:
    stain_type=validfilename.split('_')[1]
    if stain_type=='HE':
        continue
    else:
        valid_count=valid_count+1
        num=validfilename.split('_')[0]
        filename1=validfilename
        filename2=num+'_HE_val.jpg'
        lmkname1=filename1.split('.')[0]+'.xlsx'
        lmkname2=filename2.split('.')[0]+'.xlsx'
        try:
            lmk1 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname1),header=None,index_col=None))
            lmk1=lmk1[1:,1:].astype('float')
            if filename1=='57_PGR_val.jpg':
                lmk1=lmk1[0:-7,:]
            lmk2 =np.array(pd.read_excel(os.path.join(validimgpath, lmkname2),header=None,index_col=None))
            lmk2=lmk2[1:,1:].astype('float')
            if filename1=='57_PGR_val.jpg':
                lmk2=lmk2[0:-7,:]
            for idx in idxs.squeeze().tolist()[-80:-20]:
                imgidx=np.int32(np.floor(idx/200))
                kpsidx=idx-imgidx*200
                if valid_count==imgidx:
                    lmk2[kpsidx,0]=0
                    lmk2[kpsidx,1]=0
            name = ['X', 'Y']
            outlmk1 = pd.DataFrame(columns=name, data=np.asarray(lmk1)[:,[0,1]])
            outlmk1.to_excel(savekpspath+lmkname1)
            outlmk2 = pd.DataFrame(columns=name, data=np.asarray(lmk2)[:,[0,1]])
            outlmk2.to_excel(savekpspath+lmkname2)
        except:
            pass
        shutil.copy(os.path.join(validimgpath, filename1),os.path.join(savekpspath, filename1))
        shutil.copy(os.path.join(validimgpath, filename2),os.path.join(savekpspath, filename2))