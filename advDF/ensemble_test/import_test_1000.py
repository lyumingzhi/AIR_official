import torchvision
import cv2
# from advDF import SimSwap
import torch 
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import pandas as pd
# from advDF.ensemble_test import insight_arcface_backbones
from torch.nn import DataParallel
import advDF.ensemble_test.attacks as attacks
import advDF.ensemble_test.loadAttackModel as loadAttackModel
from advDF.ensemble_test import options
# from advDF.ensemble_test.loadTargetModel import get_Simswap
# import  advDF.ensemble_test.loadTargetModel as loadTargetModel
# from advDF.ensemble_test.Simswap import Simswap
# from advDF.ensemble_test.AgileGAN import AgileGAN

# from advDF.ensemble_test.Faceshifter import Faceshifter
# from advDF.ensemble_test.Megagan import Megagan
import sys
import random

class Tester():
    def __init__(self,opt) -> None:
        self.simswap=None
        # self.agilegan=AgileGAN()
        self.faceshifter=None
        self.megagan=None
        # self.model={'megagan':self.megagan,'simswap':self.simswap,'agilegan':self.agilegan,'faceshifter':self.faceshifter}

        # self.model={'simswap':self.simswap,'megagan':self.megagan,'agilegan':self.agilegan,'faceshifter':self.faceshifter}
        # self.model={'megagan':self.megagan,'agilegan':self.agilegan,'faceshifter':self.faceshifter}
        # self.model={'simswap':self.simswap,'agilegan':self.agilegan,'faceshifter':self.faceshifter}
        # self.model={'simswap':self.simswap,'megagan':self.megagan,'agilegan':self.agilegan}
        # self.model={'simswap':self.simswap}
        # self.model={'megagan':self.megagan}
        self.model={}
        for model in opt.source_model:
            if model=='simswap':
                import advDF.ensemble_test.Faceshifter as Faceshifter
                import advDF.ensemble_test.Megagan as Megagan
                self.faceshifter=Faceshifter.Faceshifter()
                self.megagan=Megagan.Megagan(opt)
                self.model={'megagan':self.megagan,'faceshifter':self.faceshifter}
            if model=='megagan':
                import advDF.ensemble_test.Simswap as Simswap
                import advDF.ensemble_test.Faceshifter as Faceshifter
                self.simswap=Simswap.Simswap(opt)
                self.faceshifter=Faceshifter.Faceshifter()
                self.model={'simswap':self.simswap,'faceshifter':self.faceshifter}
            if model=='faceshifter':
                import advDF.ensemble_test.Simswap as Simswap
                import advDF.ensemble_test.Megagan as Megagan
                self.simswap=Simswap.Simswap(opt)
                self.megagan=Megagan.Megagan(None)
                self.model={'simswap':self.simswap,'megagan':self.megagan}
                # self.model={'megagan':self.megagan}
            
    def run_attack(self,img_s_path,img_t_path,attacker,img_t_paths=None):
        img_s=cv2.imread(img_s_path)[...,::-1].transpose(2,0,1).copy()
        img_t=cv2.imread(img_t_path)[...,::-1].transpose(2,0,1).copy()

        assert (img_s is not None) and (img_t is not None)
        with torch.no_grad():
            img_s=transforms.ToTensor()(img_s.transpose(1,2,0).copy()/255).unsqueeze(0).cuda().float()
            img_t=transforms.ToTensor()(img_t.transpose(1,2,0).copy()/255).unsqueeze(0).cuda().float()
            original_output={}
            for name in self.model.keys():
                assert img_s.dim()==4 and img_t.dim()==4
                print(img_s.size(),img_t.size())
                output=self.model[name](img_s,img_t)   
                original_output[name]=self.model[name].depreprocess(output)
                print(name)
                assert original_output[name].dim()==4

        #     output=self.simswap(img_s,img_t)
        #     original_output_simswap=self.simswap.depreprocess(output)
            
        #     output=self.agilegan(img_s)
        #     original_output_agilegan=self.agilegan.depreprocess(output)

        #     output=self.faceshifter(img_s,img_t)
        #     original_output_faceshifter=self.faceshifter.depreprocess(output)
            
        #     output=self.megagan(img_s,img_t)
        #     original_output_megagan=self.megagan.depreprocess(output)
        # original_output={'simswap':original_output_simswap,'faceshifter':original_output_faceshifter,'megagan':original_output_megagan,'agilegan':original_output_agilegan}
        # output=output.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        

        source_adv,noise_adv,output_adv,adv_id_index, additive_noise, relighted_img=attacker.ensemble_test(img_s=img_s,img_d=img_t,y=original_output,image1_path=img_s_path,image2_path=img_t_path,target_3pimage=None)
        
        print('max error',torch.max(torch.abs(source_adv-relighted_img)))
        assert torch.max(torch.abs(source_adv-relighted_img)) <=0.05001
        img_s=cv2.imread(img_s_path)[...,::-1].transpose(2,0,1).copy()
        img_s=transforms.ToTensor()(img_s.transpose(1,2,0).copy()/255).unsqueeze(0).cuda().float()

        if opt.reshape_to_min:
            source_adv=noise_adv.cuda()+img_s.cuda()
        test_attack_outputs=[]
        # for img_t_path in img_t_paths:
        img_t=cv2.imread(img_t_path)[...,::-1].transpose(2,0,1).copy()

        assert (img_s is not None) and (img_t is not None)
        with torch.no_grad():
            
            img_t=transforms.ToTensor()(img_t.transpose(1,2,0).copy()/255).unsqueeze(0).cuda().float()
            test_attack_output={}
            for name in self.model.keys():
                
                # assert opt.relighting==False
                output=self.model[name](img_s+noise_adv.cuda(),img_t)
                output=self.model[name].depreprocess(output)
                test_attack_output[name]=output
                
            # test_attack_outputs.append(test_attack_output)
      
            
        for name,value in test_attack_output.items():
            test_attack_output[name]=test_attack_output[name].detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
            original_output[name]=original_output[name].detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
            output_adv[name]=output_adv[name].detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        img_s=img_s.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        img_t=img_t.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        source_adv=source_adv.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        noise_adv=torch.clamp(torch.abs(noise_adv)*10,min=0,max=1).detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        
        if opt.relighting:
            relighted_img=relighted_img.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
        
        # print(original_output['simswap'].shape,output_adv['simswap'].shape)
        # exit()
        return img_s,img_t,original_output,test_attack_output,output_adv,source_adv,noise_adv,adv_id_index, relighted_img

        # print(np.max(output),np.min(output))
        # assert cv2.imwrite('advDF/ensemble_test/test_output.jpg',output*255)
def pad_to_largest(image_list):
    # max_height=max(row1.shape[1],row2.shape[1],row3.shape[1],row4.shape[1],row5.shape[1])
    # max_width=max(row1.shape[2],row2.shape[2],row3.shape[2],row4.shape[2],row5.shape[2])
    assert image_list[0].shape[2]==3
    max_height=max(image_list,key=lambda x: x.shape[0]).shape[0]
    max_width=max(image_list,key=lambda x: x.shape[1]).shape[1]
    # print('max',max_height,max_width)
    processed_image_list=[]
    for img in image_list:
        # print(img.shape)
        img=transforms.ToTensor()(img.copy())
        # print(img.shape)
        padding=(int((max_height-img.shape[1] )/2),int((max_width-img.shape[2])/2),int((max_height-img.shape[1] )/2), int((max_width-img.shape[2] )/2))
        # print('padding',padding,img.size()[2],max_width)
        img=torchvision.transforms.functional.pad(img, padding).numpy().transpose(1,2,0).copy()
        # print(img.shape)
        processed_image_list.append(
            img
            )
    return processed_image_list

def get_id(file_name,src_id):
    
    id=src_id.loc[src_id['source_img']==file_name].iloc[0,1]
    return id
def get_file(id,src_id):
    # src_id=pd.read_excel('advDF/ensemble_test/src_id.xlsx', engine='openpyxl')
    file_name=src_id.loc[src_id['id']==id].iloc[0,0]
    return file_name
if __name__=='__main__':
    # opt=options.BaseOptions().parse()
    # print(opt)
    # t=Test(opt)
    # t.inference('advDF/SimSwap/crop_224/8.jpg','advDF/SimSwap/crop_224/4.jpg')
    opt=options.BaseOptions().parse()
    if not opt.source_model:
        raise SystemExit('Please pass at least one --source_model, e.g. --source_model simswap')
    if opt.max_pairs is not None and opt.max_pairs <= 0:
        raise SystemExit('--max_pairs must be a positive integer')
    if opt.pair_start < 0:
        raise SystemExit('--pair_start must be >= 0')
    if opt.pair_end is not None and opt.pair_end <= opt.pair_start:
        raise SystemExit('--pair_end must be greater than --pair_start')
    if opt.save_every <= 0:
        raise SystemExit('--save_every must be a positive integer')

    images_path=[]
    valid_ext={'.jpg', '.jpeg', '.png', '.bmp'}
    for root,dirs,files in os.walk(opt.dir):
        for file_path in files:
            if os.path.splitext(file_path.lower())[1] in valid_ext:
                images_path.append(file_path)
    image_name_set=set(images_path)
    if not images_path:
        raise SystemExit(f'No input images found in {opt.dir}. Put images there or pass --dir /path/to/images.')

    test_img_records=pd.read_excel(os.path.join('advDF/ensemble_test/input_pair_index.xlsx'), engine='openpyxl')
    all_test_image_indexes=[]
    for i in range(len(test_img_records)):
        all_test_image_indexes.append((test_img_records.iloc[i,0],test_img_records.iloc[i,1]))
    pair_end=opt.pair_end if opt.pair_end is not None else len(all_test_image_indexes)
    if opt.pair_start >= len(all_test_image_indexes):
        raise SystemExit(f'--pair_start {opt.pair_start} is outside input_pair_index.xlsx with {len(all_test_image_indexes)} pairs')
    pair_end=min(pair_end, len(all_test_image_indexes))
    test_image_indexes=all_test_image_indexes[opt.pair_start:pair_end]
    if opt.max_pairs is not None:
        test_image_indexes=test_image_indexes[:opt.max_pairs]
    if not test_image_indexes:
        raise SystemExit('No pairs selected. Check --pair_start, --pair_end, and --max_pairs.')

    missing_pairs=[]
    for filename1, filename2 in test_image_indexes:
        source_name=str(filename1)+'.jpg'
        target_name=str(filename2)+'.jpg'
        if source_name not in image_name_set or target_name not in image_name_set:
            missing_pairs.append((source_name, target_name))
    if missing_pairs:
        preview=', '.join([f'{a}/{b}' for a,b in missing_pairs[:5]])
        message=f'{len(missing_pairs)} selected pair(s) reference missing images under {opt.dir}; first missing pair(s): {preview}'
        if opt.fail_on_missing_pairs:
            raise SystemExit(message)
        print('Warning:', message)

    if opt.dry_run:
        available_pairs=len(test_image_indexes)-len(missing_pairs)
        print(f'Dry run OK: found {len(images_path)} images in {opt.dir}; pair_start={opt.pair_start}; pair_end={pair_end}; selected_pairs={len(test_image_indexes)}; available_pairs={available_pairs}; source_model={opt.source_model}; output_path={opt.output_path}')
        raise SystemExit(0)

    os.makedirs(opt.output_path, exist_ok=True)

    T=Tester(opt)

    df={'pair_0':[],'pair_1':[],'loss':[]}
    writer = pd.ExcelWriter(os.path.join(opt.output_path,opt.lossType+'result_loss.xlsx'), engine='openpyxl')


    arcface_for_attack,retinanet,face_recogntion_models,face_recogntion_models,faceParsing_model,fairFace_net,landmark_model=loadAttackModel.load_attack_model(opt)
    attacker=attacks.LinfPGDAttack(model=T.model,device='cuda',
            top_model=arcface_for_attack,fr_model=retinanet,opt=opt,face_recogntion_models=face_recogntion_models,faceParsing_model=faceParsing_model,fairFace_model=fairFace_net,landmark_model=landmark_model)
        
    full_output={}
    full_comparision_output={}
    full_relighted_output={}

    generated_num=0
    process_pair_num=0
    result_code=0
    written_files=[]
    if_save=0

    def save_batch(name, code):
        if name not in full_output or len(full_output[name]) == 0:
            return False
        full_output_list=np.concatenate(full_output[name],axis=1)*255
        full_comparision_output_list=np.concatenate(full_comparision_output[name],axis=1)*255
        if opt.relighting:
            full_relighted_output_list=np.concatenate(full_relighted_output[name],axis=1)*255
        assert os.path.exists(opt.output_path)
        comparison_path=os.path.join(opt.output_path, opt.lossType+'_comparision_result'+str(code)+'_'+name+'.png')
        result_path=os.path.join(opt.output_path, opt.lossType+'_result_admin'+str(code)+'_'+name+'.png')
        source_path=os.path.join(opt.output_path,opt.lossType+'_source_img'+str(code)+'_'+name+'.png')
        assert cv2.imwrite(comparison_path,full_comparision_output_list,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        assert cv2.imwrite(result_path,full_output_list,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        assert cv2.imwrite(source_path,full_output_list[3*int(full_output_list.shape[0]/7):4*int(full_output_list.shape[0]/7),:,:],[int(cv2.IMWRITE_PNG_COMPRESSION),0])
        written_files.extend([comparison_path, result_path, source_path])
        if opt.relighting:
            relighted_path=os.path.join(opt.output_path, opt.lossType+'_relightted_'+str(code)+'_'+name+'.png')
            assert cv2.imwrite(relighted_path,full_relighted_output_list,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            written_files.append(relighted_path)
        full_output[name]=[]
        full_comparision_output[name]=[]
        if opt.relighting:
            full_relighted_output[name]=[]
        return True


    # # for num1 in range(10):
    # # for num1 in [30000]:
    # # for num1 in [29959]:
    # # for num1 in [3291]:
    # for num1 in [3,3291,2338,3056,138,2942,1903,3157,1544,29890,29959, 643,2363,2869,2936,2611,2570,3052,2854,1291,    4424,8347,2121,8023,8445,6691,5100,5582,4707,8846,7796, 5580,7863]:
    # # for num1 in [29914,29959,29890]:
    # # for num1 in [29941,29991,29971,29967,26612]:
    #     # for num2 in range(10):
    #     # for num2 in [29959]:
    #     # for num2 in [3291]:
    #     for num2 in [3,3291,2338,3056,138,2942,1903,3157,1544,29890,29959, 643,2363,2869,2936,2611,2570,3052,2854,1291,    4424,8347,2121,8023,8445,6691,5100,5582,4707,8846,7796, 5580,7863]:
    #     # for num2 in [30000]:
    #     # for num2 in [29514,29959,29890]:
    #     # for num2 in [29941,29991,29971,29967,26612]:
    src_id=pd.read_excel('advDF/ensemble_test/src_id.xlsx', engine='openpyxl')

    # pre_assign_test_file=[3,3291,2338,3056,138,2942,1903,3157,1544,29890,29959, 643,2363,2869,2936,2611,2570,3052,2854,1291,    4424,8347,2121,8023,8445,6691,5100,5582,4707,8846,7796, 5580,7863]
    # pre_assign_test_id=[get_id(str(num)+'.jpg',src_id) for num in pre_assign_test_file]
    
    ######################################
    # pre_assign_test_file=[]
    # pre_assign_test_id=[]
    # imgs_to_test=pre_assign_test_file+[]
    # id_to_test=pre_assign_test_id+[]

    # while len(imgs_to_test)<1000:
    #     random_file=random.randrange(30000)
    #     if get_id(str(random_file)+'.jpg',src_id) in imgs_to_test:
    #         continue
    #     id_to_test.append(get_id(str(random_file)+'.jpg',src_id))
    #     imgs_to_test.append(random_file)
        
    # print(imgs_to_test)
    # print(id_to_test)
    ######################################
    # print(test_image_indexes)
    #############################################
    # for filename1 in imgs_to_test:
    #     # num1=int(filename1[:-4])
    #     num1=filename1
    #     num2=num1
    #     while num2==num1:
    #         num2=random.randrange(len(imgs_to_test))
    ############################################
    
    for (filename1, filename2) in test_image_indexes:
        num1=filename1
        num2=filename2
        if str(num1)+'.jpg' not in image_name_set or str(num2)+'.jpg' not in image_name_set:
            continue
        else:
            process_pair_num+=1
            # if process_pair_num<=50*14:
            #     result_code=14
            #     continue

            image1_path=str(num1)+'.jpg'
            image2_path=str(num2)+'.jpg'
            image1_path=os.path.join(opt.dir,image1_path)
            image2_path=os.path.join(opt.dir,image2_path)

            print("start to infer.",process_pair_num,image1_path)

            row1,row2,original_outputs,test_attack_outputs,output_adv,row6,row7,adv_id_index, relighted_img=T.run_attack(image1_path,image2_path,attacker)

            df['pair_0'].append(num1)
            df['pair_1'].append(num2)
            df['loss'].append(adv_id_index)


            for name, value in original_outputs.items():

                one_column=pad_to_largest([row1,row2,original_outputs[name],test_attack_outputs[name],output_adv[name],row6,row7])
                comparision_rows=pad_to_largest([original_outputs[name],test_attack_outputs[name],output_adv[name]])

                if opt.relighting:
                    relighted_imgs=pad_to_largest([relighted_img])

                output=np.concatenate(one_column,axis=0)
                comparision_output=np.concatenate(comparision_rows,axis=0)

                if opt.relighting:
                    relighted_output=np.concatenate(relighted_imgs,axis=0)
                # full_output.append(output)
                if name not in full_output:
                    assert name not in full_comparision_output
                    full_output[name]=[]
                    full_comparision_output[name]=[]
                    
                    full_relighted_output[name]=[]
            
                full_output[name].append(output)
                full_comparision_output[name].append(comparision_output)
                if opt.relighting:
                    full_relighted_output[name].append(relighted_output)
                
                print("done.",str(num1),str(num2))

                print('size of list',len(full_output[name]))
                # if  len(full_output[name])%50==0 or len(full_output[name])==99:
                # if len(full_output[name])%1==0 or len(full_output[name])==99:
                if len(full_output[name])%opt.save_every==0:
                    if save_batch(name, result_code):
                        generated_num+=opt.save_every
                        if_save=1
                    # exit(0)

                elif generated_num==1050*len(original_outputs) and (len(full_output[name])+generated_num/len(original_outputs))%(33*33)==0:
                    if save_batch(name, result_code):
                        if_save=1
                    # exit()
            if if_save==1:
                if_save=0
                result_code+=1
    for name in list(full_output.keys()):
        if save_batch(name, result_code):
            result_code+=1
    print(result_code)
    print('pair_start', opt.pair_start)
    print('pair_end', pair_end)
    print('selected pair', len(test_image_indexes))
    print('process pair',process_pair_num)
    if process_pair_num == 0:
        raise SystemExit('No selected pairs were processed. Check --dir, input_pair_index.xlsx, or use --dry_run for diagnostics.')
    df=pd.DataFrame(df)
    result_xlsx=os.path.join(opt.output_path,opt.lossType+'result_loss.xlsx')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    manifest={
        'status': 'complete',
        'pair_index': 'advDF/ensemble_test/input_pair_index.xlsx',
        'pair_start': int(opt.pair_start),
        'pair_end': int(pair_end),
        'selected_pairs': int(len(test_image_indexes)),
        'processed_pairs': int(process_pair_num),
        'missing_pairs': int(len(missing_pairs)),
        'source_model': opt.source_model,
        'lossType': opt.lossType,
        'testType': opt.testType,
        'testAttackType': opt.testAttackType,
        'relighting': bool(opt.relighting),
        'total_mask': bool(opt.total_mask),
        'hard_constraint': bool(opt.hard_constraint),
        'save_every': int(opt.save_every),
        'result_xlsx': os.path.relpath(result_xlsx, opt.output_path),
        'output_files': sorted([os.path.relpath(path, opt.output_path) for path in written_files]),
    }
    with open(os.path.join(opt.output_path, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
# from advDF.SimSwap.model_for_attack.resnet import resnet_face18
# import advDF.SimSwap.retinanet.model as face_recognition_model
# from advDF.SimSwap.options.test_options import TestOptions
# from advDF.SimSwap.attacks import LinfPGDAttack
# # from advDF.SimSwap.model_for_attack import *


# from advDF.SimSwap import insight_arcface_backbones
# from advDF.SimSwap.faceParsing import get_mask
# from advDF.SimSwap.FairFaceNet import FairFaceNet
# from advDF.SimSwap import get_landmark 

