from copyreg import constructor
from re import L
import sys
from unittest import result


# from advDF.DPR.utils.utils_SH import *
from advDF.DPR.model.defineHourglass_1024_gray_skip_matchFeature import HourglassNet,HourglassNet_1024

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2

import random
import math
import torch.nn as nn
import torch

import torchvision.transforms as transforms
import cv2 
from advDF.ensemble_test.path_utils import require_file
class DPR(nn.Module):
    def __init__(self,device):
        super(DPR,self).__init__()
        self.my_network_512= HourglassNet(16)
        self.my_network = HourglassNet_1024(self.my_network_512, 16)
        self.my_network.load_state_dict(torch.load(require_file(os.path.join('advDF/DPR/trained_model/', 'trained_model_1024_03.t7'), 'DPR relighting checkpoint')))
        self.my_network.cuda()
        self.my_network.train(False)
        self.sh=torch.nn.Parameter(torch.Tensor([0,0,0,0,0,0,0,0,0]).view(1,9,1,1),requires_grad=True)
        
        self.tanh=torch.nn.Tanh()
        self.device=device
    def get_sh_input(self,original_sh=None,constraint=None):
        if original_sh is None:
            sh_base=torch.Tensor([1,0,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
            sh_base+=self.tanh(self.sh*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
            return sh_base
        else:
            if original_sh is not None:
                sh_base=original_sh.view(1,9,1,1).to(self.device)*torch.Tensor([1,1,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
                # print(self.sh.size())
                # exit(0)
                sh_base+=self.tanh(self.sh*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
 
            sh_base=torch.clamp(sh_base,min=-1,max=1)
            if constraint is not None and original_sh is not None:
                # print('min',original_sh-torch.abs(original_sh)*constraint,'max',torch.abs(original_sh)*constraint)
                sh_base=torch.clamp(sh_base, min=original_sh-torch.abs(original_sh)*constraint, max=original_sh+torch.abs(original_sh)*constraint)
            return sh_base
    def extract_sh(self,x):
        x=transforms.Resize((1024,1024))(x)
        Lab=rgb_to_lab(x)
        inputL=Lab[0,0,:,:].clone().unsqueeze(0).unsqueeze(0)/255
        sh_base=torch.Tensor([1,0,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
        outputImg, _, outputSH, _  = self.my_network(inputL, sh_base, 0)

        print(sh_base,outputSH)
        # exit()
        return outputSH
    def forward(self,input,original_sh=None,tmp_sh=None,constraint=None,inter_grad={}):
        # input=input[:,:1,:,:]


        # print(input.size())
        # exit(0)
        # input=torch.zeros(input.size()).to(self.device)
        original_size_of_input=input.size()[-2:]
        input=transforms.Resize((1024,1024))(input)
        Lab=rgb_to_lab(input)
        
        print('rgb to lab')
        # print('Lab',torch.sum(Lab))

        # self.sh=torch.nn.Parameter(torch.Tensor([0,0,0,0,0,0,0,0,0]).view(1,9,1,1),requires_grad=True).to(self.device)
        # sh_base=torch.Tensor([0,0,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
        # sh_base=torch.Tensor([1,0,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
        
        if tmp_sh is not None:
            # sh_base+=self.tanh(tmp_sh*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))

            # sh_base+=self.tanh(tmp_sh*torch.Tensor([1,1,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
            sh_base=tmp_sh
            # print(sh_base)
            # exit()
        elif original_sh is not None:
            sh_base=original_sh.view(1,9,1,1).to(self.device)*torch.Tensor([1,1,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
            
            
            
            def print_grad(name):
                
                def grad_func(grad):
                    inter_grad[name]=grad
                    print('biggest grad for '+name+'  is ',torch.max(torch.abs(grad)))
                return grad_func

            sh_clone=self.sh.clone()
            sh_clone.register_hook(print_grad('attack'))

            # sh_base+=self.tanh(self.sh*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
            sh_base+=self.tanh(sh_clone*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))

            
            
        else:
            sh_base=torch.Tensor([1,0,0,0,0,0,0,0,0]).view(1,9,1,1).to(self.device)
            sh_base+=self.tanh(self.sh*torch.Tensor([0,0,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
            # sh_base+=self.tanh(self.sh*torch.Tensor([1,1,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device))
            # sh_base+=self.sh*torch.Tensor([1,1,1,1,1,1,1,1,1]).view(1,9,1,1).to(self.device)
        sh_base=torch.clamp(sh_base,min=-1,max=1)
        if constraint is not None and original_sh is not None:
            # print('min',original_sh-torch.abs(original_sh)*constraint,'max',torch.abs(original_sh)*constraint)
            sh_base=torch.clamp(sh_base, min=original_sh-torch.abs(original_sh)*constraint, max=original_sh+torch.abs(original_sh)*constraint)
            # for i in range(9):
            
            #     print('sh',self.sh[:,i,:,:].item(),sh_base[:,i,:,:].item(),(sh_base-original_sh)[:,i,:,:].item())
        # print('sh_base',sh_base)
        # print(type(Lab[0,0,:,:]))
        # loss=torch.sum(sh_base)
        # loss.backward()
        # print('sh_base',torch.sum(torch.isnan(sh_base)),'sh',torch.sum(torch.isnan(self.sh)), 'input', torch.sum(torch.isnan(input)),'Lab', torch.sum(torch.isnan(Lab)))
        # exit()
        inputL=Lab[0,0,:,:].clone().unsqueeze(0).unsqueeze(0)/255

        # print('inputL',torch.sum(torch.isnan(inputL)))
        # print('inputL',inputL, torch.sum(inputL))
        
        outputImg, _, outputSH, _  = self.my_network(inputL, sh_base, 0)
        
        # print('outputImg',torch.sum(torch.isnan(outputImg)),'outputSH',torch.sum(torch.isnan(outputSH)))

        outputImg=outputImg[0].clone()
        # print(outputImg)
        # # print(outputImg.size())
        # exit()

        # Lab_mask=torch.zeros(Lab.size()).to(self.device)
        # Lab_mask[0,0,:,:]=torch.ones(Lab_mask[0,0,:,:].size())

        # Lab_clone=torch.zeros(Lab.size()).to(self.device).detach()
        # Lab_clone[1:,1:,:,:]+=Lab[1:,1:,:,:].clone()
        # Lab_clone=Lab_clone.clone()
        # Lab_clone[0,0,:,:]=outputImg[0,:,:].clone()*255
        

        # Lab_mask=torch.zeros(Lab.size()).to(self.device)
        # Lab_mask[0,0,:,:]=torch.ones(Lab_mask[0,0,:,:].size())
        # # Lab_clone=Lab_clone.detach()
        # Lab=Lab.clone()-Lab.clone()*Lab_mask+Lab_clone

        # Lab_clone=Lab.clone()
        # Lab_clone[0,0,:,:]=Lab_clone[0,0,:,:].detach()
        # Lab_clone[0,0,:,:]=outputImg[0,:,:].clone()*255

        Lab=Lab.clone()
        Lab[0,0,:,:]=outputImg[0,:,:].clone()*255

        # print(Lab[0].permute(1,2,0))
        # exit()

        # Lab[0,0,:,:]=torch.zeros(Lab[0,0,:,:].size())
        # outputImg=outputImg.expand(1,3,-1,-1)
        # print(Lab)
        # exit()
        result_rgb=lab_to_rgb(Lab)
        print('lab to rgb')
        # print('labto rgb', torch.sum(torch.isnan(result_rgb)),'Lab_clone',torch.sum(torch.isnan(Lab)))
        # print('result ',result_rgb)
        # exit()
        # print(result_rgb[0].permute(1,2,0)*255)
        # cv2.imwrite('advDF/ensemble_test/result_rgb.jpg',result_rgb[0].detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        result_rgb=transforms.Resize(original_size_of_input)(result_rgb)
        return result_rgb
def rgb_to_lab(srgb_input):
    result_lab=[]
    for i in range(srgb_input.size()[0]):
        srgb=srgb_input[i,:,:,:].clone()
        srgb_pixels=srgb.reshape((3,-1)).permute(1,0)
        # srgb_pixels = torch.reshape(srgb, [-1, 3])


        linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
        exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        
        rgb_to_xyz = torch.tensor([
                    #    X        Y          Z
                    [0.412453, 0.212671, 0.019334], # R
                    [0.357580, 0.715160, 0.119193], # G
                    [0.180423, 0.072169, 0.950227], # B
                ]).type(torch.FloatTensor).cuda()
        
        xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
        

        # XYZ to Lab
        xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).cuda())

        epsilon = 6.0/29.0

        linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).cuda()

        exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).cuda()

        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
        # convert to lab
        fxfyfz_to_lab = torch.tensor([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ]).type(torch.FloatTensor).cuda()
        lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()
        #return tf.reshape(lab_pixels, tf.shape(srgb))
        # return torch.reshape(lab_pixels, srgb.shape)
        # return lab_pixels
        
        lab_pixels=lab_pixels*torch.Tensor([255/100,1,1]).view(1,3).cuda()+torch.Tensor([0,128,128]).cuda()
        lab_pixels=lab_pixels.permute(1,0).view(srgb.size())

        result_lab.append(lab_pixels.clone().unsqueeze(0))

        # # print(lab_pixels.permute(1,0).view(srgb.size()))
        # print(np.floor(srgb_input[0].detach().cpu().numpy().transpose(1,2,0)*255).shape)
        # lab=cv2.cvtColor(np.floor(srgb_input[0].detach().cpu().numpy().transpose(1,2,0)*255).astype('uint8'), cv2.COLOR_BGR2LAB)
        # assert cv2.imwrite('advDF/ensemble_test/dpr_test.jpg',np.floor(srgb_input[0].detach().cpu().numpy().transpose(1,2,0)*255))
        # print(lab[:10,:10,:])
        
    return torch.cat(result_lab,dim=0)
def lab_to_rgb(lab_input):
    result_rgb=[]
    for i in range(lab_input.size()[0]):
        lab=lab_input[i,:,:,:].clone()
        lab_pixels=lab.reshape((3,-1)).permute(1,0)
        lab_pixels=lab_pixels*torch.Tensor([100/255,1,1]).view(1,3).cuda()+torch.Tensor([0,-128,-128]).view(1,3).cuda()
        # lab_pixels = torch.reshape(lab, [-1, 3])
        # convert to fxfyfz
        lab_to_fxfyfz = torch.tensor([
            #   fx      fy        fz
            [1/116.0, 1/116.0,  1/116.0], # l
            [1/500.0,     0.0,      0.0], # a
            [    0.0,     0.0, -1/200.0], # b
        ]).type(torch.FloatTensor).cuda()
        fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda(), lab_to_fxfyfz)

        # convert to xyz
        epsilon = 6.0/29.0
        linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).cuda()
        exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).cuda()


        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

        # denormalize for D65 white point
        xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).cuda())


        xyz_to_rgb = torch.tensor([
            #     r           g          b
            [ 3.2404542, -0.9692660,  0.0556434], # x
            [-1.5371385,  1.8760108, -0.2040259], # y
            [-0.4985314,  0.0415560,  1.0572252], # z
        ]).type(torch.FloatTensor).cuda()

        rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
        # avoid a slightly negative number messing up the conversion
        #clip
        rgb_pixels[rgb_pixels > 1] = 1
        rgb_pixels[rgb_pixels < 0] = 0

        linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).cuda()
        exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).cuda()
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        result_rgb.append(srgb_pixels.clone().permute(1,0).view(lab.size()).unsqueeze(0))
        # return torch.reshape(srgb_pixels, lab.shape)
    return torch.cat(result_rgb,dim=0)

if __name__=='__main__':
    relighting_net=DPR('cuda').cuda()
    x=cv2.imread('./advDF/One-Shot-Face-Swapping-on-Megapixels/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/100.jpg')[...,::-1].transpose(2,0,1).copy()
    cv2.imwrite('./advDF/ensemble_test/original_input_for_relighting.jpg',x.transpose(1,2,0)[...,::-1])
    x=transforms.ToTensor()(x.transpose(1,2,0).copy()/255).unsqueeze(0).cuda().float()
    sh=relighting_net.extract_sh(x)
    
    # relighting_net.sh.data=sh
    # print(relighting_net.sh)
    # sh[:,2:,...]=torch.Tensor([0.2,-0.05,-0.05,0,-0.15,0,0.05]).view(sh[:,2:,...].size())
    sh[:,2:,...]=torch.Tensor([0.2,-0.05,-0.05,0,-0.15,0,0.05]).view(sh[:,2:,...].size())+3*torch.Tensor([ 0.2/3,0.35/3,0.3/3,0.2/3,0.3/3,0.2/3,0.35/3]).view(sh[:,2:,...].size())
    relighting_net.sh.data=sh
    re_x=relighting_net(x,original_sh=sh,constraint=1000)
    re_x=re_x.detach().cpu().squeeze(0).permute(1,2,0).numpy()[...,::-1]
    cv2.imwrite('./advDF/ensemble_test/relighted_sample.jpg',re_x*255)
    
