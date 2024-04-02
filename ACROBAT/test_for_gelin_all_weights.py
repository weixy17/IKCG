import os
import pdb
#logs=['5feDec18']#['4acFeb28','5feDec18','51fFeb21','94eFeb04','a23Jan31','ab2Feb24']#['820Dec08','9ceSep14','45eSep12','625Dec19','9ecSep14','060Jan21','47fDec02','911Sep17','341Dec07','414Sep15','155Dec02','597Sep16','658Dec17','889Dec10','aceDec11','b93Sep18','bf0Jan21','c88Dec05','d4aDec01','d9eJan21','de8Sep14','dceJan23','e48Dec15','e98Dec22','fcbDec11','8dbJan14','3afDec01','7caJan21','5feDec18','1f6Dec05']
#for log in logs:
    #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
    #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
    #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+log+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
    #time.sleep(3)
    #os.system(exit(0))

folderpath="/data/wxy/association/Maskflownet_association_1024/weights_gelin_final/"
logpath="/data/wxy/association/Maskflownet_association_1024/logs/"
weightfiles=os.listdir(folderpath)
error_files=[]
for weightfile in weightfiles:
    logname=weightfile.split('_')[0]
    # pdb.set_trace()
    # if logname[-4] is not '0' or '1':
        # logname=logname[0:-4]+'1'+logname[-3:]
        # weightname=logname+'_'+str(weightfile.split('_')[-1])
        # os.rename(os.path.join(folderpath,weightfile),os.path.join(folderpath,weightname))
    logfile=logname+'.log'
    # pdb.set_trace()
    if os.path.exists(os.path.join(logpath,logfile)):
        error_files.append(weightfile)
        os.system('python main_validate_for_ANHIR_train.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --valid')
        # continue
    else:
        with open(os.path.join(logpath,logfile),'w+') as f:
            f.write(r'[2022/01/05 16:24:05] start=0, train=400, val=251, host=amax, batch=2')
            f.close()
        # os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 1024 --valid')
        os.system('python main_validate_for_ANHIR_train.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --clear_steps --weight 1000 --batch 1 --relative UM --prep 1024 --valid')
            #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 512 --valid')
            #os.system('python main.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c '+logname[0:-5]+' --weight 1000 --batch 2 --relative UM --prep 256 --valid')
print(error_files)
pdb.set_trace()