# import os
# import pdb
# runid='795Feb03-1544'
# # runid='99cFeb03-1553'
# runid='470Feb04-1631'
# runid='11fFeb10-2002'
# # runid='795Feb03-1544'
# # runid0='470Feb04-1631'
# # runid0='11fFeb10-2002'
# runid0='f5dFeb11-1947'
# aas=[]
# writelns=[]
# valid_count=0
# with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid0+"-partial.log", 'r') as f:
    # for lnf in f:
        # if 'steps= ' in lnf:
            # steps0=lnf.split('steps= ')[-1].split(' ')[0]
            # aa0=float(lnf.split('aa=')[-1].split('(')[0])
            # ma0=float(lnf.split('ma=')[-1].split(',')[0])
            # m75tha0=float(lnf.split('m75tha=')[-1].split(',')[0])
            # m90tha0=float(lnf.split('m90tha=')[-1].split(',')[0])
            # am0=float(lnf.split('am=')[-1].split(',')[0].split('(')[0])
            # mm0=float(lnf.split('mm=')[-1].split(',')[0])
            # am75th0=float(lnf.split('am75th=')[-1].split(',')[0].split('(')[0])
            # m75thm75th0=float(lnf.split('m75thm75th=')[-1].split(',')[0])
            # am90th0=float(lnf.split('am90th=')[-1].split(',')[0].split('(')[0])
            # m90thm90th0=float(lnf.split('m90thm90th=')[-1].split(',')[0])
            # large10=float(lnf.split('large1=')[-1].split(',')[0].split('(')[0])
            # large20=float(lnf.split('large2=')[-1].split(',')[0].split('(')[0])
            # large30=float(lnf.split('large3=')[-1].split(',')[0].split('(')[0])
            # large40=float(lnf.split('large4=')[-1].split(',')[0].split('(')[0])
            # large50=float(lnf.split('large5=')[-1].split(',')[0].split('(')[0])
            # mMI0=float(lnf.split('mMI=')[-1].split(',')[0].split('(')[0])
            # mm75th0=float(lnf.split('mm75th=')[-1].split(',')[0])
            # mm90th0=float(lnf.split('mm90th=')[-1].split(',')[0])
            # if 1:#aa0<85.6 and aa0>84 and large50<500:
                # valid_steps=[]
                # with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial.log", 'r') as fi:
                    # for ln in fi:
                        # if 'steps= ' in ln:
                            # steps=ln.split('steps= ')[-1].split(' ')[0]
                            # # pdb.set_trace()
                            # aa=float(ln.split('aa=')[-1].split('(')[0])
                            # ma=float(ln.split('ma=')[-1].split(',')[0])
                            # m75tha=float(ln.split('m75tha=')[-1].split(',')[0])
                            # m90tha=float(ln.split('m90tha=')[-1].split(',')[0])
                            # am=float(ln.split('am=')[-1].split(',')[0].split('(')[0])
                            # mm=float(ln.split('mm=')[-1].split(',')[0])
                            # am75th=float(ln.split('am75th=')[-1].split(',')[0].split('(')[0])
                            # m75thm75th=float(ln.split('m75thm75th=')[-1].split(',')[0])
                            # am90th=float(ln.split('am90th=')[-1].split(',')[0].split('(')[0])
                            # m90thm90th=float(ln.split('m90thm90th=')[-1].split(',')[0])
                            # large1=float(ln.split('large1=')[-1].split(',')[0].split('(')[0])
                            # large2=float(ln.split('large2=')[-1].split(',')[0].split('(')[0])
                            # large3=float(ln.split('large3=')[-1].split(',')[0].split('(')[0])
                            # large4=float(ln.split('large4=')[-1].split(',')[0].split('(')[0])
                            # large5=float(ln.split('large5=')[-1].split(',')[0].split('(')[0])
                            # mMI=float(ln.split('mMI=')[-1].split(',')[0].split('(')[0])
                            # mm75th=float(ln.split('mm75th=')[-1].split(',')[0])
                            # mm90th=float(ln.split('mm90th=')[-1].split(',')[0])
                            
                            
                            # if aa<aa0 and ma<ma0 and m75tha<m75tha0 and m90tha<m90tha0 and am<am0 and mm<mm0 and am75th<am75th0 and m75thm75th<m75thm75th0 and \
                                # am90th<am90th0 and m90thm90th<m90thm90th0 and large1<large10 and large2<large20 and large3<large30 and large4<large40 and large5<large50\
                                 # and mMI>mMI0 and mm75th<mm75th0 and mm90th<mm90th0: 
                                 # valid_steps.append(steps)
                # if len(valid_steps)>0:
                    # print('{}_{}.params'.format(runid0,steps0))
                    # print(valid_steps)
            








import os
import pdb
runid='795Feb03-1544'
# runid='99cFeb03-1553'
runid='56cFeb07-1804'
# runid='7d9Feb07-1704'
runid='781Feb10-2141'
runid='53cFeb10-2144'
runid='a3eFeb11-1934'
runid='795Feb03-1544'
runid='470Feb04-1631'
runid='d70Feb12-1944'
runid='9f6Jan27-2118'
runid='159Feb02-1239'
runid='7e6Jan31-1230'
runid='6a6Feb12-2057'
runid='ecdFeb01-1204'
runid='a4aFeb13-2311'
runid='bf1Feb25-1000'
aas=[]
writelns=[]
valid_count=0
with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial.log", 'r') as fi:
    for ln in fi:
        # pdb.set_trace()
        if 'steps= ' in ln:
            steps=ln.split('steps= ')[-1].split(' ')[0]
            # aa=float(ln.split('aa=')[-1].split('(')[0])
            # ma=float(ln.split('ma=')[-1].split(',')[0])
            # m75tha=float(ln.split('m75tha=')[-1].split(',')[0])
            # m90tha=float(ln.split('m90tha=')[-1].split(',')[0])
            # am=float(ln.split('am=')[-1].split(',')[0].split('(')[0])
            # mm=float(ln.split('mm=')[-1].split(',')[0])
            # am75th=float(ln.split('am75th=')[-1].split(',')[0].split('(')[0])
            # m75thm75th=float(ln.split('m75thm75th=')[-1].split(',')[0])
            # am90th=float(ln.split('am90th=')[-1].split(',')[0].split('(')[0])
            # m90thm90th=float(ln.split('m90thm90th=')[-1].split(',')[0])
            # large1=float(ln.split('large1=')[-1].split(',')[0].split('(')[0])
            # large2=float(ln.split('large2=')[-1].split(',')[0].split('(')[0])
            # large3=float(ln.split('large3=')[-1].split(',')[0].split('(')[0])
            # large4=float(ln.split('large4=')[-1].split(',')[0].split('(')[0])
            # large5=float(ln.split('large5=')[-1].split(',')[0].split('(')[0])
            # mMI=float(ln.split('mMI=')[-1].split(',')[0].split('(')[0])
            # mm75th=float(ln.split('mm75th=')[-1].split(',')[0])
            # mm90th=float(ln.split('mm90th=')[-1].split(',')[0])
            
            
            # pdb.set_trace()
            # if aa < 89.9447 and ma<57.7633 and m75tha<97.1213 and m90tha<209.7879 and am<68.4549 and mm<45.7838 and am75th<105.6889 and m75thm75th<116.5427\
                # and am90th<166.2152 and m90thm90th<380.8161 and large1<67.7344 and large2<37.2186 and large3<28.2852 and large4<17.5468 and large5<10.7378\
                # and mMI<0.6868 and mm75th<65.8447 and mm90th<102.6337: 
                # print('{}_{}.params'.format(runid,steps))
            # if aa>85.9877 and ma>57.4920 and m75tha>107.9389 and m90tha>187.9111 and am>63.6482 and mm>42.8760 and am75th>98.1717 and m75thm75th>119.1516:#\
                # # and am90th>164.9835 and m90thm90th>331.9462 and large1>67.7743 and large2>35.8246 and large3>26.8769 and large4>17.2332 and large5>12.4234\
                # # and mMI>0.6907 and mm75th>66.3808 and mm90th>104.8832:
                # print('{}_{}.params'.format(runid,steps))
            # if  aa<85.8754 and ma<56.3152 and m75tha<104.9639 and m90tha<173.4659 and am<63.1955 and mm<42.0784 and am75th<102.4315 and m75thm75th<119.0981 and \
                # am90th<166.3719 and m90thm90th<374.9741 and large1<151.6773 and large2<179.3865 and large3<211.6754 and large4<246.3295 and large5<465.5742\
                 # and mMI>0.6928 and mm75th<59.1864 and mm90th<88.7973: 
            # if aa < 85.6136 and ma<51.5540 and m75tha<96.9581 and m90tha<161.6615 and am<62.9823 and mm<37.0883 and am75th<102.3691 and m75thm75th<110.7930\
                # and am90th<158.2981 and m90thm90th<387.7183 and large1<145.7081 and large2<175.2708 and large3<198.8108 and large4<239.9681 and large5<476.6863\
                # and mMI>0.6965 and mm75th<56.7577 and mm90th<87.2959: 
            # if aa<85.2752 and ma<51.7373 and m75tha<97.3096 and m90tha<163.3097 and am<62.5317 and mm<35.0975 and am75th<101.7653 and m75thm75th<112.0078 and\
                    # am90th<158.8456 and m90thm90th<381.2526 and large1<145.5875 and large2<175.4230 and large3<199.5896 and large4<239.7782 and large5<472.3169\
                     # and mMI>=0.6968 and mm75th<58.0416 and mm90th<85.5852:
                    # print('{}_{}.params'.format(runid,steps))
            # output=os.popen('python TMI_rebuttle_main_large_displacement_visualization.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c {}:{} --clear_steps --weight 1000 --batch 1'.format(runid,steps)).read()
            if os.path.exists("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/"+runid+'_'+steps+'.params'):
                output=os.popen('python TMI_rebuttle_valid_multiscale_512.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c {}:{} --clear_steps --weight 1000 --batch 1'.format(runid,steps)).read()
                # output=os.popen('python TMI_rebuttle_valid_multiscale_1024.py MaskFlownet_S.yaml --dataset_cfg ACROBAT.yaml -g 1 -c {}:{} --clear_steps --weight 1000 --batch 1'.format(runid,steps)).read()
                writeln=output.split('\n')[-2].split('aa=')[-1]
                writeln='steps= {}'.format(steps)+' aa='+writeln
                print(writeln)
                writelns.append(writeln)
            
            # # if not(round(aa)<318 and round(aa)>312 and round(ma)<=259 and round(m75tha)<=371 and round(m90tha)<=645\
                    # # and round(am)<=242 and round(mm)<=181 and round(am75th)<=404 and round(m75thm75th)<=455\
                    # # and round(large1)<=738 and round(large2)<=875 and round(large3)<=1128 and round(large4)<=1676\
                    # # and round(large5)<=1997 and round(large5)>=1600 and round(am90th)<=630 and round(m90thm90th)<=1218\
                    # # and (mMI>=0.55605)):
            
            # # if not(aa<86 and mm90th<102):
            # # # if (round(aa) in aas):
                # # os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
                # # os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.states'.format(runid,steps))
                # # print('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
            # # else:
                # # aas.append(round(aa))
                
                
                
            # # valid_count=valid_count+1
            # # if valid_count>=2:
                # # break
        else:
            writelns.append(ln)
with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_corrected.log", 'w') as file:
    for i in range(len(writelns)):
        file.write(writelns[i])
        file.write('\n')