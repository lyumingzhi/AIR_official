#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
# sys.path.append('./faceParsing/')
# print(sys.path)
from advDF.ensemble_test.faceParsing.logger import setup_logger
from advDF.ensemble_test.faceParsing.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from advDF.ensemble_test.path_utils import require_file
label2index={'hair':17,'skin':1,'u_lip':12,'l_lip':13,'neck':14,'nose':10,'mouth':11,'r_ear':8,'l_ear':7,'eye_g':6,'r_eye':5,'l_eye':4,'r_brow':3,'l_brow':2,'background':0,'ear_r':9,'neck_l':15,'cloth':16,'hat':18}

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='output/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi==4 :
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im=vis_im.reshape(vis_im.shape[1],vis_im.shape[2],vis_im.shape[3]).transpose(1,2,0)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        face_region=np.where((vis_parsing_anno<=15 )&(vis_parsing_anno>=1))
        vis_parsing_anno[face_region[0],face_region[1]]=100
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        exit()

    # return vis_im

def evaluate(src_path=None,tgt_path=None,packagepth='',src_name=None,tgt_name=None, cp='model_final_diss.pth',return_mask=False):

    src_add=os.path.join(src_path,src_name)
    tgt_add=os.path.join(tgt_path,tgt_name)
    if not os.path.exists(packagepth):
        os.makedirs(packagepth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(packagepth,'res/cp/', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # for image_path in os.listdir(dspth):
        
        img = Image.open(tgt_add)
        image = img.resize((224, 224), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        print(np.unique(parsing))

        save_pth=osp.join(packagepth,'res/test_res/', tgt_name[:-4]+'_mask'+tgt_name[-4:])
        # print('save_path',save_pth)
        # exit(0)
        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_pth
        if return_mask:
            return parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_pth)
    return osp.join(packagepth,'res/test_res/', tgt_name[:-4]+'_mask.png')



# def return_mask(input_img,net):
#     # mask=evaluate(src_path,tgt_path,src_name=src_name,tgt_name=tgt_name,packagepth='./faceParsing/',cp='79999_iter.pth',return_mask=True)
#     to_tensor = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     with torch.no_grad():
#         # for image_path in os.listdir(dspth):
        
#         input_img=to_tensor(input_img)
#         out = net(input_img)[0]
#         parsing = out.squeeze(0).cpu().numpy().argmax(0)
#         # print(parsing)
#         print(np.unique(parsing))

#         # save_pth=osp.join('./faceParsing/','res/test_res/', tgt_name[:-4]+'_mask'+tgt_name[-4:])
#         # print('save_path',save_pth)
#         # exit(0)
#         # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_pth
#         return parsing_maps( parsing, stride=1, save_im=True, save_path=None)
    
def get_mask_address(src_path,tgt_path,src_name,tgt_name):
    mask_address=evaluate(src_path,tgt_path,src_name=src_name,tgt_name=tgt_name,packagepth='./faceParsing/',cp='79999_iter.pth')
    return mask_address
# def create_faceParsing_model(cp):
#     n_classes = 19
#     net = BiSeNet(n_classes=n_classes)
#     # net.cuda()
#     save_pth = osp.join('./faceParsing/','res/cp/', cp)
#     net.load_state_dict(torch.load(save_pth))
#     net.eval()
#     return net

class FPModel():
    def __init__(self,cp):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        # net.cuda()
        save_pth = require_file(osp.join('advDF/ensemble_test/faceParsing/', 'res/cp/', cp), 'BiSeNet face parsing checkpoint')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
        self.net=net
        pass
    def set_device(self,device):
        self.net.to(device)
    # def return_vis_parsing_anno(self,input_image)
    def return_mask(self,input_img,total_mask=True,region_to_mask=[],to_return_vis_anno=False):
        # mask=evaluate(src_path,tgt_path,src_name=src_name,tgt_name=tgt_name,packagepth='./faceParsing/',cp='79999_iter.pth',return_mask=True)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_normalization= transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            # for image_path in os.listdir(dspth):
            if type(input_img)==str:
                input_img = Image.open(input_img)
                input_img = input_img.resize((224, 224), Image.BILINEAR)
                input_img = to_tensor(input_img)
                input_img = torch.unsqueeze(input_img, 0)
            
                # input_img = input_img.cuda()

            # image=input_img.detach().cpu().numpy()
            try:
                input_img=to_tensor(input_img).float().cuda()
            except:
                # if input_img.dim()>3:
                #     input_img=input_img.squeeze(0)
                print(input_img.size())
                input_img=to_normalization(input_img).float().cuda()
            # print(input_img.size())
            out = self.net(input_img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))
            # print(parsing.shape)
            # exit(0)
            # save_pth=osp.join('./faceParsing/','res/test_res/', tgt_name[:-4]+'_mask'+tgt_name[-4:])
            # print('save_path',save_pth)
            # exit(0)
            # vis_parsing_maps(image, parsing, stride=1, save_im=True)
            return self.parsing_maps( parsing, stride=1, save_im=True, save_path=None,total_mask=total_mask, region_to_mask=region_to_mask,to_return_vis_anno=False)
    def eye_region(self,eyeb_region,vis_parsing_anno,enlarge_scale):
        # print('eye',eyeb_region[0])
        # exit(0)
        try:
            min_xindex=np.min(eyeb_region[0])-enlarge_scale
            max_xindex=np.max(eyeb_region[0])+enlarge_scale
            min_yindex=np.min(eyeb_region[1])-enlarge_scale
            max_yindex=np.max(eyeb_region[1])+enlarge_scale
        except:
            # min_xindex=vis_parsing_anno[vis_parsing_anno.shape[0]/8*3:vis_parsing_anno.shape[0]/8*4,vis_parsing_anno.shape[1]/5*1:vis_parsing_anno.shape[1]/5*3]
            min_xindex=vis_parsing_anno.shape[0]/8*3
            max_xindex=vis_parsing_anno.shape[0]/8*4
            min_yindex=vis_parsing_anno.shape[1]/5*1
            max_yindex=vis_parsing_anno.shape[1]/5*3
        return int(min_xindex),int(max_xindex),int(min_yindex),int(max_yindex)
    def parsing_maps(self, parsing_anno, stride, save_im=False, save_path='./output/parsing_map_on_im.jpg',total_mask=True,region_to_mask=[],to_return_vis_anno=False):
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        mask=np.ones(vis_parsing_anno.shape)
        # mask_region=np.where((vis_parsing_anno<=15 )&(vis_parsing_anno>=1))
        # mask_region=np.where((vis_parsing_anno!=12)&(vis_parsing_anno!=13)&(vis_parsing_anno!=5)&(vis_parsing_anno!=6)&(vis_parsing_anno!=7)&(vis_parsing_anno!=4))
        # mask_region=np.where((vis_parsing_anno==19))
        
        min_xindex,max_xindex,min_yindex,max_yindex=self.eye_region(np.where((vis_parsing_anno>=2)&(vis_parsing_anno<7)),vis_parsing_anno,8)
        # mask_region=np.where((vis_parsing_anno==1)|(vis_parsing_anno==0)|(vis_parsing_anno==12)|(vis_parsing_anno==13)|(vis_parsing_anno==10))
        
        mask_region=None
        for region in region_to_mask:
            r_index=label2index[region]
            if mask_region==None:
                mask_region=list(np.where((vis_parsing_anno==r_index)))
            else:
                tmp_mask_region=np.where((vis_parsing_anno==r_index))
                for dim_index in range(len(mask_region)):
                    mask_region[dim_index]=np.concatenate((mask_region[dim_index],tmp_mask_region[dim_index]))
        # mask_region=np.where((vis_parsing_anno==1)|(vis_parsing_anno==0)|(vis_parsing_anno==10))

        if total_mask:
            # print('total mask')
            # exit()
            mask[mask_region[0],mask_region[1]]=0
        else:
            # print('not total mask')
            mask[mask_region[0],mask_region[1]]=0.1 
        # min_xindex,max_xindex,min_yindex,max_yindex=eye_region(np.where((vis_parsing_anno==6)&(vis_parsing_anno==5)&(vis_parsing_anno==4)),vis_parsing_anno,30)
        # print(max(0,min_xindex),min(mask.shape[0],max_xindex),max(0,min_yindex),min(mask.shape[1],max_yindex))
        
        mask[max(0,min_xindex):min(mask.shape[0],max_xindex),max(0,min_yindex):min(mask.shape[1],max_yindex)]=np.ones((min(mask.shape[0],max_xindex)-max(0,min_xindex),min(mask.shape[1],max_yindex)-max(0,min_yindex)))
        if not to_return_vis_anno:
            return mask
        else:
            return vis_parsing_anno 
# if __name__ == "__main__":
#     # evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='79999_iter.pth')
#     evaluate( cp='79999_iter.pth')

