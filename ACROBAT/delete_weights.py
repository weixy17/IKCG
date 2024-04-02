import os
import pdb
runid='795Feb03-1544'
# runid='99cFeb03-1553'
runid='f54Feb01-1158'
runid='994Jan31-1235'
runid='56cFeb07-1804'
runid='037Feb17-1217'
# runid='432Feb18-1240'
# runid='5e6Feb18-1239'
# runid='262Feb18-1256'
runid='095Feb19-1616'
# runid='da9Feb19-1618'
runid='863Feb28-0914'
runid='be0Mar07-0023'
runid='b21Mar07-0012'
runid='ac6Feb25-1053'
runid='af1Mar01-1056'
runid='21dMar07-0028'
runid='c34Mar07-0030'
runid='eb1Feb26-2302'
runid='af1Mar01-1056'
runid='471Mar07-0029'
runid='e7aMar06-2151'
runid='7f5Mar08-1019'
runid='849Mar08-0935'
runid='136Mar08-1017'
aas=[]
writelns=[]
# with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_delete_kps.log", 'r') as fi:
with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial.log", 'r') as fi:
    for ln in fi:
        if 'steps= ' in ln:
            steps=ln.split('steps= ')[-1].split(' ')[0]
            aa=float(ln.split('aa=')[-1].split('(')[0])
            ma=float(ln.split('ma=')[-1].split(',')[0])
            m75tha=float(ln.split('m75tha=')[-1].split(',')[0])
            m90tha=float(ln.split('m90tha=')[-1].split(',')[0])
            am=float(ln.split('am=')[-1].split(',')[0].split('(')[0])
            mm=float(ln.split('mm=')[-1].split(',')[0])
            am75th=float(ln.split('am75th=')[-1].split(',')[0].split('(')[0])
            m75thm75th=float(ln.split('m75thm75th=')[-1].split(',')[0])
            am90th=float(ln.split('am90th=')[-1].split(',')[0].split('(')[0])
            m90thm90th=float(ln.split('m90thm90th=')[-1].split(',')[0])
            large1=float(ln.split('large1=')[-1].split(',')[0].split('(')[0])
            large2=float(ln.split('large2=')[-1].split(',')[0].split('(')[0])
            large3=float(ln.split('large3=')[-1].split(',')[0].split('(')[0])
            large4=float(ln.split('large4=')[-1].split(',')[0].split('(')[0])
            large5=float(ln.split('large5=')[-1].split(',')[0].split('(')[0])
            mMI=float(ln.split('mMI=')[-1].split(',')[0].split('(')[0])
            mm75th=float(ln.split('mm75th=')[-1].split(',')[0])
            mm90th=float(ln.split('mm90th=')[-1].split(',')[0])
            # pdb.set_trace()
            # if aa < 89.9447 and ma<57.7633 and m75tha<97.1213 and m90tha<209.7879 and am<68.4549 and mm<45.7838 and am75th<105.6889 and m75thm75th<116.5427\
                # and am90th<166.2152 and m90thm90th<380.8161 and large1<67.7344 and large2<37.2186 and large3<28.2852 and large4<17.5468 and large5<10.7378\
                # and mMI<0.6868 and mm75th<65.8447 and mm90th<102.6337: 
                # print('{}_{}.params'.format(runid,steps))
            # if aa>85.9877 and ma>57.4920 and m75tha>107.9389 and m90tha>187.9111 and am>63.6482 and mm>42.8760 and am75th>98.1717 and m75thm75th>119.1516:#\
                # # and am90th>164.9835 and m90thm90th>331.9462 and large1>67.7743 and large2>35.8246 and large3>26.8769 and large4>17.2332 and large5>12.4234\
                # # and mMI>0.6907 and mm75th>66.3808 and mm90th>104.8832:
                # print('{}_{}.params'.format(runid,steps))
            
            
            
            
            # pdb.set_trace()
            
            # if not(round(aa)<318 and round(aa)>312 and round(ma)<=259 and round(m75tha)<=371 and round(m90tha)<=645\
                    # and round(am)<=242 and round(mm)<=181 and round(am75th)<=404 and round(m75thm75th)<=455\
                    # and round(large1)<=738 and round(large2)<=875 and round(large3)<=1128 and round(large4)<=1676\
                    # and round(large5)<=1997 and round(large5)>=1600 and round(am90th)<=630 and round(m90thm90th)<=1218\
                    # and (mMI>=0.55605)):
            # if not aa<90:
            # if not(aa>82 and aa<85.5232 and ma<55.4723 and m75tha<104.2676 and m90tha<174.8413 and am<63.0435 and mm<41.8051 and am75th<101.9874 and m75thm75th<119.7238\
                # and am90th<165.7467 and m90thm90th<379.3284 and large1<151.0668 and large2<178.6926 and large3<211.1173 and large4<245.5146 and large5<463.0018\
                # and mMI>0.6933 and mm75th<60.7892 and mm90th<88.4293):
            # if (round(aa) in aas):
            # if not (aa<85.5232 and ma<55.4723 and m75tha<104.2676 and m90tha<174.8413 and am<63.0435 and mm<41.8051 and am75th<101.9874 and m75thm75th<119.7238\
                # and am90th<165.7467 and m90thm90th<379.3284 and large1<151.0668 and large2<178.6926 and large3<211.1173 and large4<245.5146 and large5<463.0018\
                # and mMI>0.6933 and mm75th<60.7892 and mm90th<88.4293):# and dist_means<50:
            # if (aa<=83.8434 and ma<=53.8193 and m75tha<=96.6041 and m90tha<=181.9390 and am<=61.8473 and mm<=39.3552 and am75th<=100.0717\
                # and m75thm75th<=105.3005 and am90th<=161.0267 and m90thm90th<=377.5123 and large1<=145.8417 and large2<=174.5453\
                # and large3<=198.6060 and large4<=234.0385 and large5<=410.9312 and mMI>=0.6990 and mm75th<=59.5743 and mm90th<=93.7984):
            # if (aa>=95):
            # pdb.set_trace()
            if not (aa<86):
            # if (int(steps)!=1245 and int(steps)!=1262):
            # pdb.set_trace()
            
                os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
                os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.states'.format(runid,steps))
                print('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
                writelns.append(ln)
            else:
                aas.append(round(aa))
weights_files=os.listdir("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/")
weights_files=[temp for temp in weights_files if temp.split('_')[0]==runid]
# for temp in weights_files:
    # steps=int(temp.split('_')[-1].split('.')[0])
    # if steps>6709:
        # os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
        # os.popen('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.states'.format(runid,steps))
        # print('rm -rf /ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/{}_{}.params'.format(runid,steps))
                
weights_files_steps=[temp.split('_')[-1].split('.')[0] for temp in weights_files]
with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_selected_delete_kps.log", 'w') as file:
# with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_selected.log", 'w') as file:
    # with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_delete_kps.log", 'r') as fi:
    with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial.log", 'r') as fi:
        for ln in fi:
            if 'steps= ' in ln:
                steps=ln.split('steps= ')[-1].split(' ')[0]
                if steps in weights_files_steps:
                    file.write(ln)
            else:
                file.write(ln)
# weights_files=os.listdir("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/weights_TMI_rebuttle/")
# weights_files=[temp for temp in weights_files if temp.split('_')[0]==runid]
# weights_files_steps=[temp.split('_')[-1].split('.')[0] for temp in weights_files]
# with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial_selected.log", 'w') as file:
    # with open("/ssd2/wxy/IPCG_Acrobat/association/Maskflownet_association_1024/logs/"+runid+"-partial.log", 'r') as fi:
        # for ln in fi:
            # if 'steps= ' not in ln:
                # file.write(ln)
    # for temp in writelns:
        # file.write(temp)