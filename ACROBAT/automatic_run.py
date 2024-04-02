
# ############################################train
# import os
# import pdb
# #logs=['5feDec18']#['4acFeb28','5feDec18','51fFeb21','94eFeb04','a23Jan31','ab2Feb24']#['820Dec08','9ceSep14','45eSep12','625Dec19','9ecSep14','060Jan21','47fDec02','911Sep17','341Dec07','414Sep15','155Dec02','597Sep16','658Dec17','889Dec10','aceDec11','b93Sep18','bf0Jan21','c88Dec05','d4aDec01','d9eJan21','de8Sep14','dceJan23','e48Dec15','e98Dec22','fcbDec11','8dbJan14','3afDec01','7caJan21','5feDec18','1f6Dec05']
# #for log in logs:
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
    # #time.sleep(3)
    # #os.system(exit(0))

# # folderpath=r"/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_multi_siftmap_kp_supervised/s4_deformable_reg_siftflow_as_loss_1024/weights_gelin/"
# # logpath=r'/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_multi_siftmap_kp_supervised/s4_deformable_reg_siftflow_as_loss_1024/logs/'
# # weightfiles=os.listdir(folderpath)
# # for weightfile in weightfiles:
    # # logname=weightfile.split('_')[0]
    # # if logname[-4] is not '0' or '1':
        # # logname=logname[0:-4]+'1'+logname[-3:]
        # # weightname=logname+'_'+str(weightfile.split('_')[-1])
        # # os.rename(os.path.join(folderpath,weightfile),os.path.join(folderpath,weightname))
    # # logfile=logname+'.log'
    # # #pdb.set_trace()
    # # if os.path.exists(os.path.join(logpath,logfile)):
        # # continue
    # # else:
        # # with open(os.path.join(logpath,logfile),'w+') as f:
            # # f.write(r'[2022/01/05 16:24:05] start=0, train=400, val=251, host=amax, batch=2')
            # # f.close()
        # # os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
            # # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
            # # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
# for validate_num in range(239,245):
    # os.system('python main_recursive_DLFSFG_multiscale_1024_refine_for_evaluation.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c e6eMay08 --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num))
    
    
    
    
    
# ############################################test
# import os
# import pdb
# import shutil
# import csv
# weightpath="/data/wxy/association/Maskflownet_association_1024/weights/"
# nums=[]
# with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'evaluation':
            # nums.append(num)
            # # if (row[1].split('_')[0]+'_'+row[1].split('_')[1])==classes[0]:
                # # nums_index.append(len(nums)-1)
# for validate_num in range(251):
    # print(nums[validate_num])
    # weightfilepath=weightpath+str(validate_num)+'/'
    # if os.path.exists(weightfilepath):
        # weightname=os.listdir(weightfilepath)[0].split('-')[0]
        # print(weightname)
        # os.system('python main_recursive_DLFSFG_multiscale_1024_refine_for_evaluation_evaluation.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c ' +weightname+ ' --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num)+' --valid')
    # else:
        # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1.csv',"/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/"+str(nums[validate_num])+'_1.csv')
        # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1.jpg',"/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/"+str(nums[validate_num])+'_1.jpg')
        # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_2.jpg',"/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/"+str(nums[validate_num])+'_2.jpg')
        # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1_warpped.csv',"/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/"+str(nums[validate_num])+'_1_warpped.csv')
        # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1_warpped.jpg',"/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist/"+str(nums[validate_num])+'_1_warpped.jpg')
# #logs=['5feDec18']#['4acFeb28','5feDec18','51fFeb21','94eFeb04','a23Jan31','ab2Feb24']#['820Dec08','9ceSep14','45eSep12','625Dec19','9ecSep14','060Jan21','47fDec02','911Sep17','341Dec07','414Sep15','155Dec02','597Sep16','658Dec17','889Dec10','aceDec11','b93Sep18','bf0Jan21','c88Dec05','d4aDec01','d9eJan21','de8Sep14','dceJan23','e48Dec15','e98Dec22','fcbDec11','8dbJan14','3afDec01','7caJan21','5feDec18','1f6Dec05']
# #for log in logs:
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
    # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
    # #time.sleep(3)
    # #os.system(exit(0))

# # folderpath=r"/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_multi_siftmap_kp_supervised/s4_deformable_reg_siftflow_as_loss_1024/weights_gelin/"
# # logpath=r'/data4/slice_s4_deformable_reg_siftflow_sparse_as_loss/s4_deformable_reg_siftflow_as_loss_multi_siftmap_kp_supervised/s4_deformable_reg_siftflow_as_loss_1024/logs/'
# # weightfiles=os.listdir(folderpath)
# # for weightfile in weightfiles:
    # # logname=weightfile.split('_')[0]
    # # if logname[-4] is not '0' or '1':
        # # logname=logname[0:-4]+'1'+logname[-3:]
        # # weightname=logname+'_'+str(weightfile.split('_')[-1])
        # # os.rename(os.path.join(folderpath,weightfile),os.path.join(folderpath,weightname))
    # # logfile=logname+'.log'
    # # #pdb.set_trace()
    # # if os.path.exists(os.path.join(logpath,logfile)):
        # # continue
    # # else:
        # # with open(os.path.join(logpath,logfile),'w+') as f:
            # # f.write(r'[2022/01/05 16:24:05] start=0, train=400, val=251, host=amax, batch=2')
            # # f.close()
        # # os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
            # # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
            # # #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
# # for validate_num in range(180,252):
    # # os.system('python main_recursive_DLFSFG_multiscale_1024_refine_for_evaluation.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c e6eMay08 --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num))
    






# ############################################checkout test
# import os
# import pdb
# import shutil
# import csv
# weightpath="/data/wxy/association/Maskflownet_association_1024/weights/"
# savepath="/data/wxy/association/Maskflownet_association_1024/kps_for_submit/refine_without_dist_6kps/"
# nums=[]
# with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'evaluation':
            # nums.append(num)
            # # if (row[1].split('_')[0]+'_'+row[1].split('_')[1])==classes[0]:
                # # nums_index.append(len(nums)-1)
# for validate_num in range(251):
    # # print(nums[validate_num])
    # if str(nums[validate_num])+'_1_warpped.csv' not in os.listdir(savepath):
        # print(nums[validate_num])
        # weightfilepath=weightpath+str(validate_num)+'/'
        # if os.path.exists(weightfilepath):
            # weightname=os.listdir(weightfilepath)[0].split('_')[0]
            # print(weightname)
            # os.system('python main_recursive_DLFSFG_multiscale_1024_refine_for_evaluation_evaluation.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c ' +weightname+ ' --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num)+' --valid')
        # else:
            # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1.csv',savepathstr(nums[validate_num])+'_1.csv')
            # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1.jpg',savepath+str(nums[validate_num])+'_1.jpg')
            # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_2.jpg',savepath+str(nums[validate_num])+'_2.jpg')
            # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1_warpped.csv',savepath+str(nums[validate_num])+'_1_warpped.csv')
            # shutil.copy('/data/wxy/association/Maskflownet_association_1024/kps_for_submit/final_path/'+str(nums[validate_num])+'_1_warpped.jpg',savepath+str(nums[validate_num])+'_1_warpped.jpg')





# ############################################test 6kp
# import os
# import pdb
# import shutil
# import csv
# weightpath="/data/wxy/association/Maskflownet_association_1024/weights/"
# nums=[]
# with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    # reader = csv.reader(f)
    # for row in reader:
        # if reader.line_num == 1:
            # continue
        # num = int(row[0])
        # if row[5] == 'evaluation':
            # nums.append(num)
            # # if (row[1].split('_')[0]+'_'+row[1].split('_')[1])==classes[0]:
                # # nums_index.append(len(nums)-1)
# for validate_num in range(251):
    # weightfilepath=weightpath+str(validate_num)+'/'
    # if os.path.exists(weightfilepath):
        # weightname=os.listdir(weightfilepath)[0].split('_')[0]
        # print(weightname)
        # os.system('python main_validate_for_ANHIR_train_for_evaluation_evaluation.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c ' +weightname+ ' --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num)+' --valid')
        
        
        
        ############################################test 6kp
import os
import pdb
import shutil
import csv
weightpath="/data/wxy/association/Maskflownet_association_1024/weights/"
nums=[]
with open(os.path.join("/data/wxy/Pixel-Level-Cycle-Association-main/data/matrix_sequence_manual_validation.csv"), newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
        num = int(row[0])
        if row[5] == 'evaluation':
            nums.append(num)
            # if (row[1].split('_')[0]+'_'+row[1].split('_')[1])==classes[0]:
                # nums_index.append(len(nums)-1)
for validate_num in range(251):
    weightfilepath=weightpath+str(validate_num)+'/'
    if os.path.exists(weightfilepath):
        os.system('python main_validate_for_ANHIR_train_for_evaluation_evaluation_for_e6e.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 1 -c e6eMay08-2235 --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --validate_num '+str(validate_num)+' --valid')