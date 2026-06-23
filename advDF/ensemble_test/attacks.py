import copy
import re
from statistics import mode

from numpy.lib.function_base import average, gradient, interp
from cv2 import transform
import numpy as np
from collections.abc import Iterable
from numpy.random import random
from scipy.stats import truncnorm
import advDF.ensemble_test.TVLoss as TVLoss
from torch.optim import optimizer
from torchvision import transforms
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import os
# import defenses.smoothing as smoothing
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import cv2
import pytorch_colors as colors
import advDF.ensemble_test.TVLoss
import sys
from advDF.ensemble_test.noise_func import Noise_Func
import advDF.ensemble_test.TVLoss as TVLoss
from advDF.ensemble_test.DPR import DPR
from advDF.ensemble_test.contrast_func import Contrast_func
from advDF.ensemble_test.sh_distribution_model import SH_distribution, SH_NormalDistribution
import math

np.set_printoptions(threshold=np.inf)

# from   faceParsing import get_mask
transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Resize((256,256))
    ])
transformer_for_megagan= transforms.Compose([transforms.Resize((256,256)),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
preprocess_for_different_size=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Resize((224,224)),
    # transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]
)
normalize_transform=transforms.Compose([transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
# transformer_Arcface = transforms.Compose([
        
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
transformer_Arcface=transforms.Compose([])
detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])
mask_transform=transforms.Compose([transforms.GaussianBlur(21,200)])
def preprocess_for_attack_arcface(image):
    # image=F.interpolate(image,scale_factor=0.5)
    image=F.interpolate(image,(112,112))
    image=F.pad(image,(8,8,8,8),'constant',0)
    image=image[:,0,:,:]
    image = torch.stack((image, torch.fliplr(image)))
    return image
class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.02
    , k=150, a=0.02 , momentum=1, feat = None,
     top_model=None,fr_model=None,opt=None,face_recogntion_models=None,faceParsing_model=None,fairFace_model=None,landmark_model=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size

        IFGSM k=20 a=0.01 e=0.05
        saimese l1 k=20 a=0.003 e=0.05
        saimese latent l1 k=20 a=0.003 e=0.05
        saimese transfer l1 k=2 a=0.05 e=0.05 
        """

        self.opt=opt
        self.model = model
        self.input_size=(1,3,256,256)

        self.relighting_net=None
        self.noise_func=None
        # self.model={'encoder':encoder,'swapper':swapper,'generator':generator}
        # if self.opt.lossType=='output_id' or self.opt.lossType=='internal_id' or self.opt.lossType=='manipulate_id' or self.opt.lossType=='ensemble_wb_test':
        #     self.top_model=top_model.to(device).eval()
        # else:
        #     self.top_model=None
        self.top_model=top_model.to(device).eval()
        if self.opt.lossType=='fd_loss':
            self.fr_model=fr_model.to(device).eval()
        else:
            self.fr_model=None
        self.faceParsing_model=faceParsing_model
        self.faceParsing_model.set_device(device)
        if self.opt.lossType=='face_classification_loss' or self.opt.testAttackType=='ensemble_loss':
            self.fairFace_model=fairFace_model.to(device)
        else:
            self.fairFace_model=None
        if self.opt.lossType=='landmark_loss':
            self.landmark_model=landmark_model.to(device).eval()
        else:
            self.landmark_model=None
        self.face_recogntion_models=[model.to(device) for model in face_recogntion_models]
        self.epsilon = epsilon
        # self.noise_func=Noise_Func(1.0,0.0)
        self.k = k
        self.a = a
        self.momentum=momentum
        # self.loss_fn = nn.MSELoss().to(device)
        self.loss_fn = nn.CosineSimilarity().to(device)
        self.cosloss_fn=nn.CosineSimilarity().to(device)
        self.l2loss_fn=nn.MSELoss().to(device)
        self.TVLoss=TVLoss.TVLoss().to(device)
        self.device = device

        self.TI_kernel=None
        
        self.hard_constrain=opt.hard_constraint
        self.total_mask=opt.total_mask
        # Feature-level attack? Which layer?
        self.feat = feat
        
        # PGD or I-FGSM?
        self.rand = True
    
    def get_img_gradient(self,src):
        src=src.clone().detach()
        # src_numpy=src.detach().cpu().numpy()[0].transpose(1,2,0)*255
        # cv2.imwrite('advDF/ensemble_test/src_test.jpg',src_numpy)
        # print(src_numpy)
        
        # v=torch.Tensor([[0.25,0.5,0,-0.5,-0.25],[0.5,1,0,-1,-0.5],[1,2,0,-2,-1],[0.5,1,0,-1,-0.5],[0.25,0.5,0,-0.5,-0.25]]).expand(3,-1,-1).expand(3,-1,-1,-1)
        # v=v.view((3,3,5,5)).cuda()
        # G_x=F.conv2d(src,v,padding=int(v.size()[-1]/2))    
        # h=torch.Tensor([[0.25,0.5,1,0.5,0.25],[0.5,1,2,1,0.5],[0,0,0,0,0],[-0.5,-1,-2,-1,-0.5],[-0.25,-0.5,-1,-0.5,-0.25]]).expand(3,-1,-1).expand(3,-1,-1,-1)
        # h=h.view((3,3,5,5)).cuda()
        # G_y=F.conv2d(src,h,padding=int(h.size()[-1]/2))   


        # a=torch.Tensor([[0.25,0.5,0,-0.5,-0.25],[0.5,1,0,-1,-0.5],[1,2,0,-2,-1],[0.5,1,0,-1,-0.5],[0.25,0.5,0,-0.5,-0.25]])
        # a=a.view((5,5))
        # v=torch.zeros((3,3,5,5)).to(self.device)
        # for i in range(3):
        #     v[i,i,:,:]=a

        # G_x=F.conv2d(src,v,padding=int(v.size()[-1]/2))

        # b=torch.Tensor([[0.25,0.5,1,0.5,0.25],[0.5,1,2,1,0.5],[0,0,0,0,0],[-0.5,-1,-2,-1,-0.5],[-0.25,-0.5,-1,-0.5,-0.25]])
        # b=b.view((5,5))
        # h=torch.zeros((3,3,5,5)).to(self.device)
        # for i in range(3):
        #     h[i,i,:,:]=b
        # G_y=F.conv2d(src,h,padding=int(h.size()[-1]/2))
        
        a= torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]])
        a= torch.Tensor([[0.5,0.5,0.5,0.5,0.5],[0.5,1,1,1,0.5],[0.5,1,-16,1,0.5],[0.5,1,1,1,0.5],[0.5,0.5,0.5,0.5,0.5]])
        # a= torch.Tensor([[1/3,1/3,1/3,1/3,1/3,1/3,1/3],[1/3,0.5,0.5,0.5,0.5,0.5,1/3],[1/3,0.5,1,1,1,0.5,1/3],[1/3,0.5,1,-24,1,0.5,1/3],[1/3,0.5,1,1,1,0.5,1/3],[1/3,0.5,0.5,0.5,0.5,0.5,1/3],[1/3,1/3,1/3,1/3,1/3,1/3,1/3]])
        # a=torch.zeros((11,11))
        # theta=int(a.size()[-1]/2)/math.sqrt(3)
        # for i in range(a.size()[-2]):
        #     for j in range(a.size()[-1]):
        #         a[i,j]=1/(2*math.pi*theta*theta)*math.exp(-(abs(i-int(a.size()[-2]/2))**2+abs(j-int(a.size()[-1]/2))**2)/(2*(theta**2)))
        # a=a/torch.sum(a)
        # a=a.view((11,11))
        # a=a.view((3,3))
        a=a.view((5,5))
        # a=a.view(7,7)

        # v=torch.zeros((3,3,5,5)).to(self.device)
        # for i in range(3):
        #     v[i,i,:,:]=a

        # v=torch.zeros((3,3,11,11)).to(self.device)
        v=torch.zeros((3,3,5,5)).to(self.device)
        # v=torch.zeros((3,3,7,7)).to(self.device)
        for i in range(3):
            v[i,i,:,:]=a

        G_x=F.conv2d(src,v,padding=int(v.size()[-1]/2),bias=None)

        # b=torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]])

        # b=b.view((3,3))
        # h=torch.zeros((3,3,3,3)).to(self.device)
        # for i in range(3):
        #     h[i,i,:,:]=b
        # G_y=F.conv2d(src,h,padding=int(h.size()[-1]/2))
        
        
        G=G_x.to(self.device)
      ###################################
        # G=((G-torch.min(G.view(3,-1),dim=1)[0].view(1,3,1,1).to(self.device)))
        # G=(G/torch.max(G.view(3,-1),dim=1)[0].view(1,3,1,1).to(self.device).detach()-0.5)*2
        ##############################################
        G=(G/(torch.max(torch.abs(G).view(3,-1),dim=1)[0].view(1,3,1,1).to(self.device)))
        ##        ######################
        # G=torch.clamp(G,min=0,max=1) 
        
        G=torch.clamp(G,min=-1,max=1) 
        # print(G.size())
        # exit(0)
        # src_numpy=np.ceil(torch.abs(G.detach()).cpu().numpy()[0].transpose(1,2,0)*255)

        # img_size=src_numpy.shape[0]*src_numpy.shape[1]*src_numpy.shape[2]
        # num_bin=50
        # (unique, counts)=np.unique(G.detach().cpu().numpy()[0],return_counts=True)
        # frequencies=np.asarray((unique,counts)).T
        # print(frequencies)

        # cv2.imwrite('advDF/ensemble_test/id_dy_test.jpg',src_numpy)
        # print(src_numpy)
        # exit()
        return G

    def integrated_gradient(self,src,tgt,y,steps,tau=0.05):
        src=src.detach()
        tgt=tgt.detach()
        for name in self.model.keys():
            y[name]=y[name].clone().detach()
        gradients=[]
        # baseline=torch.rand(src.size()).to(self.device)*0.1
        baseline=torch.zeros(src.size()).to(self.device)
        scaled_inputs = [baseline + (float(i) / steps) * (src - baseline)+(torch.rand(baseline.size())-0.5).to(self.device)*tau for i in
                     range(0, steps + 1)]
        # scaled_inputs = [baseline + (float(i) / steps) * (src - baseline) for i in
        #              range(0, steps + 1)]
        # scaled_inputs = torch.cat(scaled_inputs,0)
        # scaled_inputs = scaled_inputs.to(self.device) + torch.rand(scaled_inputs.size()).to(self.device)*tau
        # # scaled_inputs = torch.from_numpy(scaled_inputs)
        # scaled_inputs = scaled_inputs.to(self.device, dtype=torch.float)
        # scaled_inputs.requires_grad_(True)
        # # tgt=tgt.expand(scaled_inputs.size()[0],-1,-1,-1)

        
        
        grads=[]
        final_loss=0
        baseline_loss=0
        original_loss=0
        for index,scaled_input in enumerate(scaled_inputs):
            loss=0
            tmp_outputs={}
            scaled_input=scaled_input.clone().detach()
            # tgt=tgt.clone().detach()
            for name in self.model.keys():
                # y[name]=y[name].clone().detach()
                self.model[name].zero_grad()
            for name in self.model.keys():                
                scaled_input.requires_grad=True
                scaled_input.grad=None
                
                assert scaled_input.dim()==4 and tgt.dim()==4
                # print(scaled_input.size(),tgt.size())
                tmp_output=self.model[name](scaled_input.clone(),tgt.clone())
                tmp_output=self.model[name].depreprocess(tmp_output)
                assert tmp_output.dim()==4
                tmp_outputs[name]=tmp_output
            # if index==0:
            #     baseline_loss=2000*self.ensemble_face_recognition_loss(tmp_outputs,y).detach().cpu()
            # elif index==(steps):
            #     original_loss=2000*self.ensemble_face_recognition_loss(tmp_outputs,y).detach().cpu()
            # loss+=20000*self.ensemble_face_recognition_loss(tmp_outputs,y)
            loss+=-20000*self.output_pixel_loss(tmp_outputs,y)

            loss.backward()
            # loss=loss.detach()
            grads.append(scaled_input.grad.detach()) 
            final_loss=loss.detach().data

            del scaled_input, tmp_outputs,loss
        # for step in range(steps):
        #     tmp_src=(src-baseline)*step/steps+baseline
        #     tmp_src.requires_grad=True
        #     loss=0
        #     for name in self.model.keys():
        #         self.model[name].zero_grad()
        #         assert tmp_src.dim()==4 and tgt.dim()==4
        #         tmp_output=self.model[name](tmp_src,tgt)
        #         tmp_output=self.model[name].depreprocess(tmp_output)
        #         assert tmp_output.dim()==4
        #         loss+=-self.ensemble_face_recognition_loss(tmp_output,y)
        #     loss.backward() 
            # gradients.append(tmp_src.grad.item())
        
            
        avg_grads = torch.mean(torch.cat(grads,0), dim=0)
        delta_X = scaled_inputs[-1] - scaled_inputs[0]
        integrated_grad = delta_X * avg_grads
        IG=integrated_grad.cpu().detach().numpy()
        # print('IG',np.sum(IG))
        # print('output Loss',(original_loss-baseline_loss))
        # exit(0)

        # average_gradient=torch.sum(torch.Tensor(gradient))/len(gradient)
        # integrated_gradient=average_gradient*(src-baseline)
        
        del delta_X, avg_grads,scaled_inputs,src,tgt,y,baseline,grads,integrated_grad
        
        
        return IG,final_loss.cpu().detach().numpy()
    def VT(self,src,tgt,y,steps,eps,tau=0.1,beta=1.5):
        src=src.clone()
        tgt=tgt.clone()
        for name in self.model.keys():
            y[name]=y[name].clone().detach()
        
        neighbors=[src+torch.rand(src.size()).to(self.device)*eps*beta*2-eps*beta for i in range(steps)]
        neighbors.append(src)
        # print(neighbors)
        gradients=[]
        src_grad=0
        for index,neighbor in enumerate(neighbors):
            loss=0
            tmp_outputs={}
            neighbor=neighbor.clone().detach()
            tgt=tgt.clone().detach()
            for name in self.model.keys():
                # y[name]=y[name].clone().detach()
                self.model[name].zero_grad()
            for name in self.model.keys():                
                neighbor.requires_grad=True
                neighbor.grad=None
                
                assert neighbor.dim()==4 and tgt.dim()==4
                # print(scaled_input.size(),tgt.size())
                tmp_output=self.model[name](neighbor.clone(),tgt.clone())
                tmp_output=self.model[name].depreprocess(tmp_output)
                assert tmp_output.dim()==4
                tmp_outputs[name]=tmp_output
            # if index==0:
            #     baseline_loss=2000*self.ensemble_face_recognition_loss(tmp_outputs,y).detach().cpu()
            # elif index==(steps):
            #     original_loss=2000*self.ensemble_face_recognition_loss(tmp_outputs,y).detach().cpu()
            loss+=2000*self.ensemble_face_recognition_loss(tmp_outputs,y)
            loss.backward()
            loss=loss.detach()
            if index==len(neighbor)-1:
                src_grad=neighbor.grad.detach()
            else:
                gradients.append(neighbor.grad.detach()) 
            final_loss=loss.cpu().detach().numpy()
        avg_grads = torch.mean(torch.cat(gradients,0), dim=0)
        grad_variance=(avg_grads-src_grad)

        del avg_grads,neighbors,src,tgt,y,loss,tmp_outputs,gradients
        return grad_variance.cpu().detach().numpy(),src_grad.cpu().detach().numpy(),final_loss

    
    def output_pixel_loss(self,attacked_output,original_output):
        if isinstance(attacked_output,dict):
            assert isinstance(attacked_output,dict)
            assert isinstance(original_output,dict)
            attacked_outputs=attacked_output
            original_outputs=original_output
            loss=0
            # for i, (attacked_output_name,original_output_name) in enumerate(zip(attacked_outputs.keys(),original_outputs.keys())):
            for i, attacked_output_name in enumerate(attacked_outputs.keys()):
                # assert attacked_output_name==original_output_name
                assert attacked_output_name in list(original_outputs)
                original_output_name=attacked_output_name
                attacked_outputs[attacked_output_name]=attacked_outputs[attacked_output_name].clone()
                original_outputs[original_output_name]=original_outputs[original_output_name].clone()
                result_size=max(attacked_outputs[attacked_output_name].size()[-1],original_outputs[original_output_name].size()[-1])
                attacked_output=transforms.Resize(result_size)(attacked_outputs[attacked_output_name])
                original_output=transforms.Resize(result_size)(original_outputs[original_output_name])
                # loss+=self.l2loss_fn(attacked_output,original_output+0.00000001)
                l1loss_fn = nn.L1Loss()
                loss+=l1loss_fn(attacked_output,original_output+0.00000001)
                
            return loss
        else:
            attacked_output=attacked_output.clone()
            original_output=original_output.clone()
            result_size=max(attacked_output.size()[-1],original_output.size()[-1])
            attacked_output=transforms.Resize(result_size)(attacked_output)
            original_output=transforms.Resize(result_size)(original_output)
        return self.l2loss_fn(attacked_output,original_output)
    def output_id_loss(self,attacked_output,original_output,default=True):
        if self.opt.testType=='df':
            raise('it is not implemented yet')
        attacked_output=attacked_output.clone()
        original_output=original_output.clone()
        if default:
            attacked_output=transformer_Arcface(attacked_output.squeeze(0))
            original_output=transformer_Arcface(original_output.squeeze(0))        
            attacked_output=F.interpolate(attacked_output.unsqueeze(0),scale_factor=0.5)
            original_output=F.interpolate(original_output.unsqueeze(0),scale_factor=0.5)
            # print(attacked_output.size())
            # print('y ',original_output.size())
            output_id=self.model['simswap'].model.netArc(attacked_output)
            origin_id=self.model['simswap'].model.netArc(original_output)
            loss = self.loss_fn(output_id, origin_id )
        else:
            attacked_output=preprocess_for_attack_arcface(attacked_output)
            original_output=preprocess_for_attack_arcface(original_output)
            output_id=self.top_model(attacked_output)
            origin_id=self.top_model(original_output)
            loss = self.loss_fn(output_id, origin_id )
        return loss
    def internal_id_loss(self,attacked_input,objected_input,default=True):
        attacked_input=attacked_input.clone()
        objected_input=objected_input.clone()
        if objected_input.dim()==3:
            objected_input=objected_input.unsqueeze(0)
        if default:
            # origin_img_id_downsample = F.interpolate(objected_input, scale_factor=0.5)
            # img_id_downsample = F.interpolate(attacked_input, scale_factor=0.5)
            origin_img_id_downsample = F.interpolate(objected_input, (112,112))
            img_id_downsample = F.interpolate(attacked_input, (112,112))


            origin_id=self.model['simswap'].model.netArc(origin_img_id_downsample)
            latend_id = self.model['simswap'].model.netArc(img_id_downsample)
            # print('latent id',latend_id.size())
            # latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
            latend_id = latend_id.to('cuda')
            # origin_id=origin_id/torch.norm(origin_id,p=2,dim=1,keepdim=True)
            loss = self.cosloss_fn(latend_id, origin_id )
        else:
            origin_img_id_downsample=preprocess_for_attack_arcface(objected_input)
            img_id_downsample=preprocess_for_attack_arcface(attacked_input)
            latend_id=self.top_model(img_id_downsample)
            origin_id=self.top_model(origin_img_id_downsample)
            # print(img_id_downsample.size(),oorigin_img_id_downsamplerigin_id.size())
            loss=torch.sum(self.cosloss_fn(latend_id,origin_id))/latend_id.size()[0]
        return loss
    def ensemble_face_recognition_loss(self,attacked_input,objected_input,i=None):
        if ((self.opt.lossType!='ensemble_wb_test' and self.opt.testType!='df') or (self.opt.lossType=='ensemble_wb_test' and self.opt.testType=='id') )and (not isinstance(attacked_input,dict)):
            attacked_input=attacked_input.clone()
            objected_input=objected_input.clone()

            attacked_input= F.interpolate(attacked_input, (112,112))
            objected_input= F.interpolate(objected_input, (112,112))

            loss=0
            if self.face_recogntion_models!=None:
                for fr_model in self.face_recogntion_models:
                    attacked_feat=fr_model(attacked_input)
                    objected_feat=fr_model(objected_input)
                    assert attacked_input.size()==objected_input.size()
                    
                        
                    loss+=self.cosloss_fn(attacked_feat,objected_feat)
            return loss
        else:
            # loss=0
            loss=[]
            for name in attacked_input.keys():
                one_attacked_input =attacked_input[name].clone()
                one_objected_input=objected_input[name].clone()
                # print(one_attacked_input.size(),name)
                one_attacked_input= F.interpolate(one_attacked_input, (112,112))
                one_objected_input= F.interpolate(one_objected_input, (112,112))

                
                if self.face_recogntion_models!=None:
                    for fr_model_index,fr_model in enumerate(self.face_recogntion_models):
                        if self.opt.left_one_out and fr_model_index==len(self.face_recogntion_models)-1:
                            break
                        attacked_feat=fr_model(one_attacked_input)
                        objected_feat=fr_model(one_objected_input)
                        assert attacked_feat.size()==objected_feat.size()                        
                        # loss+=self.cosloss_fn(attacked_feat,objected_feat)
                        # if i==0:
                        #     attacked_feat+=torch.tensor(np.random.uniform(-0.01, 0.01, attacked_feat.shape).astype('float32')).to(self.device)
                        # loss+=self.cosloss_fn(attacked_feat,objected_feat)**2 
                        loss.append(self.cosloss_fn(attacked_feat,objected_feat)**2 )
            for i in range( len( loss)):
                print(i,'loss:',loss[i])
            loss_sum=sum(loss)

            if not self.opt.left_one_out:
                return loss_sum, loss
            else:
                last_fr_loss=[]
                for name in attacked_input.keys():
                    one_attacked_input =attacked_input[name].clone()
                    one_objected_input=objected_input[name].clone()
                    # print(one_attacked_input.size(),name)
                    one_attacked_input= F.interpolate(one_attacked_input, (112,112))
                    one_objected_input= F.interpolate(one_objected_input, (112,112))

                    
                    if self.face_recogntion_models!=None:
                        for fr_model_index,fr_model in enumerate(self.face_recogntion_models):
                            if self.opt.left_one_out and fr_model_index==len(self.face_recogntion_models)-1:
                                attacked_feat=fr_model(one_attacked_input)
                                objected_feat=fr_model(one_objected_input)
                                assert attacked_feat.size()==objected_feat.size()                        
                                # loss+=self.cosloss_fn(attacked_feat,objected_feat)
                                # if i==0:
                                #     attacked_feat+=torch.tensor(np.random.uniform(-0.01, 0.01, attacked_feat.shape).astype('float32')).to(self.device)
                                # loss+=self.cosloss_fn(attacked_feat,objected_feat)**2 
                                last_fr_loss.append(self.cosloss_fn(attacked_feat,objected_feat)**2 )
                            else:
                                continue
                last_fr_loss=sum(last_fr_loss)

                return loss_sum, loss, last_fr_loss
    def ensemble_face_recognition_loss_mutli_sources(self,attacked_input,objected_input,image1_path):
        
        multi_src_images_path='advDF/ensemble_test/celeba/'+image1_path.split('/')[-1].split('.')[0]
        multi_src_images=[]
        try:
            # print(multi_src_images_path)
            # exit()
            for root,dirs,files in os.walk(multi_src_images_path):
                for file in files:
                    multi_src_images.append(os.path.join(multi_src_images_path,file))
                    # print(file)
                    # exit(0)
        except:
            print('this path does not exist')     
            exit()   
        if (self.opt.lossType!='ensemble_wb_test' and self.opt.testType!='df') or (self.opt.lossType=='ensemble_wb_test' and self.opt.testType=='id'):
            attacked_input=attacked_input.clone()
            objected_input=objected_input.clone()

            attacked_input= F.interpolate(attacked_input, (112,112))
            objected_input= F.interpolate(objected_input, (112,112))

            loss=0
            if self.face_recogntion_models!=None:
                for fr_model in self.face_recogntion_models:
                    attacked_feat=fr_model(attacked_input)
                    objected_feat=fr_model(objected_input)
                    assert attacked_input.size()==objected_input.size()
                    loss+=self.cosloss_fn(attacked_feat,objected_feat)
            return loss
        else:

            
            loss=0

            for name in attacked_input.keys():
                one_attacked_input =attacked_input[name].clone()
                one_objected_input=objected_input[name].clone()
                # print(one_attacked_input.size(),name)
                one_attacked_input= F.interpolate(one_attacked_input, (112,112))
                one_objected_input= F.interpolate(one_objected_input, (112,112))

                
                if self.face_recogntion_models!=None:
                    for fr_model in self.face_recogntion_models:
                        attacked_feat=fr_model(one_attacked_input)
                        objected_feat=fr_model(one_objected_input)
                        assert attacked_feat.size()==objected_feat.size()                        
                        # loss+=self.cosloss_fn(attacked_feat,objected_feat)
                        # loss+=self.cosloss_fn(attacked_feat,objected_feat)**2 
                        tmp_loss=self.cosloss_fn(attacked_feat,objected_feat)**2 
                        # multi_src_feats=[objected_feat]
                        for multi_source_image_path in multi_src_images:
                            # print('other source img',multi_source_image_path)
                            multi_source_image=cv2.imread(multi_source_image_path)
                            multi_source_image=transforms.ToTensor()(multi_source_image).unsqueeze(0).to(self.device)
                            multi_source_image=F.interpolate(multi_source_image,(112,112))
                            
                            multi_source_feat=fr_model(multi_source_image)
                            # multi_src_feats.append(multi_source_feat)
                            
                            assert attacked_feat.size()==multi_source_feat.size()
 
                            # loss+=self.cosloss_fn(attacked_feat,multi_source_feat.detach())**2 

                            cur_loss=self.cosloss_fn(attacked_feat,objected_feat)**2
                            if  cur_loss>tmp_loss:
                                tmp_loss=cur_loss
                        loss+=tmp_loss

                        # for feat1 in multi_src_feats:
                        #     for feat2 in multi_src_feats:
                        #         print('inter loss',self.cosloss_fn(feat1,feat2.detach())**2)
                        #     print('attacked output and src feat',self.cosloss_fn(attacked_feat,feat1.detach())**2)
                        #     break
                    # exit()
            return loss
    def ensemble_face_recognition_style_loss(self,attacked_input,objected_input):
        if self.opt.lossType!='ensemble_wb_test' and self.opt.testType!='df':
            attacked_input=attacked_input.clone()
            objected_input=objected_input.clone()

            attacked_input= F.interpolate(attacked_input, (112,112))
            objected_input= F.interpolate(objected_input, (112,112))

            loss=0
            if self.face_recogntion_models!=None:
                for fr_model in self.face_recogntion_models:
                    attacked_feat,attacked_intermediate_features=fr_model(attacked_input)
                    objected_feat,objected_intermediate_features=fr_model(objected_input)
                    for attacked_intermediate_feat, objected_intermediate_feat in zip(attacked_intermediate_features,objected_intermediate_features):    
                        loss+=self.l2loss_fn(attacked_intermediate_feat,objected_intermediate_feat.detach())
                        if loss>1e+10:
                            print(attacked_intermediate_feat)
                            print(attacked_intermediate_feat.size())
                            exit()
            return loss
        else:
            loss=0
            for name in attacked_input.keys():
                one_attacked_input=attacked_input[name].clone()
                one_objected_input=objected_input[name].clone()

                one_attacked_input= F.interpolate(one_attacked_input, (112,112))
                one_objected_input= F.interpolate(one_objected_input, (112,112))

                
                
                if self.face_recogntion_models!=None:
                    for fr_model in self.face_recogntion_models:
                        attacked_feat,attacked_intermediate_features=fr_model(one_attacked_input)
                        objected_feat,objected_intermediate_features=fr_model(one_objected_input)
                        index=0
                        for attacked_intermediate_feat, objected_intermediate_feat in zip(attacked_intermediate_features,objected_intermediate_features):  
                            index+=1 
                            attacked_feat_size=attacked_intermediate_feat.size()                             
                            attacked_gram=torch.mm(attacked_intermediate_feat.view(attacked_feat_size[0]*attacked_feat_size[1],-1),
                                attacked_intermediate_feat.view(attacked_feat_size[0]*attacked_feat_size[1],-1).t()).div(attacked_feat_size[0]*attacked_feat_size[1]*attacked_feat_size[2]*attacked_feat_size[3])
                            
                            objected_feat_size=objected_intermediate_feat.size()
                            assert objected_feat_size==attacked_feat_size
                            objected_gram=torch.mm(objected_intermediate_feat.view(objected_feat_size[0]*objected_feat_size[1],-1),
                                objected_intermediate_feat.view(objected_feat_size[0]*objected_feat_size[1],-1).t()).div(objected_feat_size[0]*objected_feat_size[1]*objected_feat_size[2]*objected_feat_size[3])
                            # print(objected_intermediate_feat[:,:3,:,:].view(3,-1).size(),objected_intermediate_feat[:,:3,:,:].view(3,-1).max(1,keepdim=True)[0].size())
                            
                            # map=objected_intermediate_feat[:,3:6,:,:].view(3,-1)-objected_intermediate_feat[:,3:6,:,:].view(3,-1).min(1,keepdim=True)[0]
                            # if index==1:
                            #     print(map.max(1,keepdim=True)[0])
                            # map=map/map.max(1,keepdim=True)[0]
                            # map=map.view(3,attacked_feat_size[2],attacked_feat_size[3]).detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
                            
                            # cv2.imwrite(os.path.join(self.opt.output_path,'feature_map_'+str(index)+'_1.jpg'),map)
                            
                            # print(objected_gram)
                        
                            
                            loss+=self.l2loss_fn(attacked_gram,objected_gram.detach())
                            if loss>1e+10:
                                print(attacked_intermediate_feat)
                                print(attacked_intermediate_feat.size())
                                exit()
                        # exit(0)
            return loss
    def ensemble_face_recognition_inter_feature_loss(self,attacked_input,objected_input):
        if self.opt.lossType!='ensemble_wb_test' and self.opt.testType!='df':
            attacked_input=attacked_input.clone()
            objected_input=objected_input.clone()

            attacked_input= F.interpolate(attacked_input, (112,112))
            objected_input= F.interpolate(objected_input, (112,112))

            loss=0
            if self.face_recogntion_models!=None:
                for fr_model in self.face_recogntion_models:
                    attacked_feat,attacked_intermediate_features=fr_model(attacked_input)
                    objected_feat,objected_intermediate_features=fr_model(objected_input)
                    for attacked_intermediate_feat, objected_intermediate_feat in zip(attacked_intermediate_features,objected_intermediate_features):    
                        loss+=self.l2loss_fn(attacked_intermediate_feat,objected_intermediate_feat.detach())
                        if loss>1e+10:
                            print(attacked_intermediate_feat)
                            print(attacked_intermediate_feat.size())
                            exit()
            return loss
        else:
            for name in attacked_input.keys():
                one_attacked_input=attacked_input[name].clone()
                one_objected_input=objected_input[name].clone()

                one_attacked_input= F.interpolate(one_attacked_input, (112,112))
                one_objected_input= F.interpolate(one_objected_input, (112,112))

                loss=0
                
                if self.face_recogntion_models!=None:
                    for fr_model in self.face_recogntion_models:
                        attacked_feat,attacked_intermediate_features=fr_model(one_attacked_input)
                        objected_feat,objected_intermediate_features=fr_model(one_objected_input)
                        for attacked_intermediate_feat, objected_intermediate_feat in zip(attacked_intermediate_features,objected_intermediate_features):                               
                            loss+=self.l2loss_fn(attacked_intermediate_feat,objected_intermediate_feat.detach())
                            if loss>1e+10:
                                print(attacked_intermediate_feat)
                                print(attacked_intermediate_feat.size())
                                exit()
                return loss
    def discriminator_loss(self,attacked_output,default=True):
        if self.opt.testType=='df':
            raise('it is not implemented yet')
        attacked_output=attacked_output.clone()
        loss=self.model['simswap'].model.get_GANLoss(attacked_output).squeeze(0)
        print('discrimination loss',loss)
        return loss
    def mapulate_id_loss(self,attacked_input,default=False):
        origin_img_id_downsample=preprocess_for_attack_arcface(attacked_input)
        latend_id=self.top_model(origin_img_id_downsample)
        # targeted_id=torch.ones(latend_id.size()).to(self.device)
        # loss=-self.loss_fn(latend_id,targeted_id)

        loss=torch.sum(torch.norm(latend_id),dim=-1).squeeze(0)
        return loss
    # def mapulate_output_loss(self,attacked_output,original_output=None,latend_id=None,img_att=None):
    def mapulate_output_loss(self,attacked_output,original_output=None,img_id=None,img_att=None):
        if self.opt.testType=='df':
            raise('it is not implemented yet')
        attacked_output=attacked_output.clone()
        if original_output!=None:
            original_output=original_output.clone().detach()
        # if latend_id!=None and img_att!=None:
        #     latend_id=latend_id.clone()
        #     img_att=img_att.clone()
        # target_output=torch.zeros(attacked_output.size()).to(self.device)
        # target_output=torch.ones(attacked_output.size()).to(self.device)
        
        # target_output=torch.rand(attacked_output.size()).to(self.device)
        
        # dump_latend_id=latend_id.detach()+0.5*torch.rand(latend_id.size()).cuda()
            ############## Forward Pass ######################
        # target_output = self.model(None, img_att, dump_latend_id, dump_latend_id, True)
        
        # target_output=1-original_output.clone().detach_()
        # print(target_output[:10,:10,:])
        # print(original_output[:10,:10,:])
        

        # target_output=torch.mean(attacked_output.detach(),dim=(attacked_output.dim()-2,attacked_output.dim()-1),keepdim=True).expand_as(attacked_output)
        
        # target_output=torch.var(attacked_output.detach(),dim=(attacked_output.dim()-2,attacked_output.dim()-1),keepdim=True).expand_as(attacked_output)
        
        #blur attack
        blur_transformer=transforms.Compose([transforms.GaussianBlur(21,200)])
        target_output=blur_transformer(attacked_output.clone().detach())
        loss=-self.l2loss_fn(attacked_output,target_output)
        # print(target_output.size())
        # exit(0)

        # loss=-self.l2loss_fn(attacked_output,target_output)
        # loss=-torch.sum(torch.var(attacked_output,dim=(attacked_output.dim()-2,attacked_output.dim()-1),keepdim=True)).to(self.device)
        return loss
    def face_detection_loss(self,attacked_output):
        if self.opt.testType=='df':
            raise('it is not implemented yet')
        attacked_output=attacked_output.clone()
        # attacked_output=torch.zeros(attacked_output.size()).cuda()
        scores,classification,transformed_anchors=self.fr_model(attacked_output)
        # print('fr scores, ',scores,'classification',classification)
        # exit()
        return -scores.sum()
    def face_classification_loss(self,attacked_input,objected_input,predict_range):
        attacked_input=attacked_input.clone()
        objected_input=objected_input.clone()
        attacked_scores_7,attacked_scores_4=self.fairFace_model(attacked_input)
        objected_scores_7,objected_scores_4=self.fairFace_model(objected_input.detach())
        original_race_output=objected_scores_7[:7]
        original_gender_output=objected_scores_7[7:9]
        original_age_output=objected_scores_7[9:18]
        loss=0
        if 'race' in predict_range:
            # original_race_pred_7=torch.argmax(original_race_output)
            # loss+=torch.exp(attacked_scores_7[0+original_race_pred_7])/torch.sum(torch.exp(attacked_scores_7[:7]))

            # origninal_race_pred_4=torch.argmax(objected_scores_4[:4])
            # loss+=torch.exp(attacked_scores_4[0+origninal_race_pred_4])/torch.sum(torch.exp(attacked_scores_4[:4]))
            loss+=-torch.exp(attacked_scores_7[1])/torch.sum(torch.exp(attacked_scores_7[:7]))
            loss+=-torch.exp(attacked_scores_4[1])/torch.sum(torch.exp(attacked_scores_4[:4]))
        if 'gender' in predict_range:
            original_gender_pred=torch.argmax(original_gender_output)
            loss+=torch.exp(attacked_scores_7[7+original_gender_pred])/torch.sum(torch.exp(attacked_scores_7[7:9]))
        if 'age' in predict_range:
            original_age_pred=torch.argmax(original_age_output)
            loss+=torch.exp(attacked_scores_7[9+original_age_pred])/torch.sum(torch.exp(attacked_scores_7[9:18]))

        
        # loss=self.l2loss_fn(attacked_input,objected_input)
        return loss
    def face_classification_loss_ver2(self,attacked_inputs,objected_inputs):
        loss=0
        for model_name in attacked_inputs:
            attacked_input=attacked_inputs[model_name].clone()
            objected_input=objected_inputs[model_name].clone()
            attacked_age_result,attacked_gender_result=self.fairFace_model(attacked_input)
            original_age_result,original_gender_result=self.fairFace_model(objected_input.detach())
            
            
            loss+=-torch.sum(attacked_age_result[[i for i in range(attacked_gender_result.size()[0])],torch.argmin(original_age_result)])
            loss+=-torch.sum(attacked_gender_result[[i for i in range(attacked_gender_result.size()[0])],torch.argmin(original_gender_result)])
        return loss
    def face_classification_loss_ver3(self,attacked_inputs,objected_inputs):
        loss=0
        for model_name in attacked_inputs:
            attacked_input=attacked_inputs[model_name].clone()
            objected_input=objected_inputs[model_name].clone()
            attacked_gender_result,attacked_age_result=self.fairFace_model.return_result(attacked_input)
            original_gender_result,original_age_result=self.fairFace_model.return_result(objected_input.detach())
            
            
            # loss+=-torch.sum(attacked_age_result[0,0])
            loss+=-torch.sum(attacked_gender_result[[i for i in range(attacked_gender_result.size()[0])],torch.argmin(original_gender_result)])
        return loss
    def face_classification_loss_ver4(self,attacked_inputs,objected_inputs):
        loss=0
        for model_name in attacked_inputs:
            attacked_input=attacked_inputs[model_name].clone()
            objected_input=objected_inputs[model_name].clone()
            attacked_logits,attacked_probs=self.fairFace_model(attacked_input)
            original_logits,original_probs=self.fairFace_model(objected_input.detach())
            
            
            # loss+=-torch.sum(attacked_age_result[0,0])
            loss+=-torch.sum(attacked_logits[[i for i in range(attacked_logits.size()[0])],torch.argmin(original_logits)])
        return loss
    def face_classification_loss_ver5(self,attacked_inputs,objected_inputs):
        loss=0
        for model_name in attacked_inputs:
            attacked_input=attacked_inputs[model_name].clone()
            objected_input=objected_inputs[model_name].clone()


            
            attacked_input= F.interpolate(attacked_input, (112,112))
            objected_input= F.interpolate(objected_input, (112,112))

            
            if self.face_recogntion_models!=None:
                attacked_feats=[]
                objected_feats=[]
                for fr_model in self.face_recogntion_models[:2]:
                    attacked_feat=fr_model(attacked_input)
                    objected_feat=fr_model(objected_input)
                    assert attacked_feat.size()==objected_feat.size()  
                    attacked_feats.append(attacked_feat)
                    objected_feats.append(objected_feat)

                attacked_feats=torch.cat(attacked_feats,dim=1)
                objected_feats=torch.cat(objected_feats,dim=1)
                attacked_logits,attacked_probs=self.fairFace_model(attacked_feats)
                original_logits,original_probs=self.fairFace_model(objected_feats.detach())
                print(original_logits)
                # exit()


            
                # loss+=-torch.sum(attacked_age_result[0,0])
                # loss+=-torch.sum(attacked_logits[[i for i in range(attacked_logits.size()[0])],torch.argmin(original_logits)])
                print(torch.argmax(original_logits))
                print(attacked_logits[[i for i in range(attacked_logits.size()[0])],torch.argmax(original_logits)])
                # loss+=torch.sum(attacked_logits[[i for i in range(attacked_logits.size()[0])],torch.argmax(original_logits)])
                loss+=torch.sum(attacked_logits[:,:1])
                return loss
            else:
                print('face recognition models are not avaliable')
                exit()

    def sp_face_region_loss(self,attacked_input,objected_input,attack_mask):
        attacked_input=attacked_input.clone()
        objected_input=objected_input.clone()
        # self_swap_img=self.get_attacked_output_from_attacked_source(attacked_input,objected_input,self.model)
        self_swap_img=self.model['simswap'](attacked_input,objected_input)
        attack_mask=transforms.Resize(self_swap_img.size(-1))(attack_mask)
        # print(objected_input.size(),attack_mask.size(),self_swap_img.size())
        loss=self.l2loss_fn(objected_input.detach()*attack_mask,self_swap_img*attack_mask)
        if torch.isnan(loss):
            print(torch.isnan(attacked_input),torch.isnan(objected_input))
        return loss
    def get_feature(self,name,features):
            def hook(model,input, output):
                features[name]=output.detach()
            return hook
    def feature_match_loss(self,attacked_output,original_output,original_input,attacked_input=None,img_att=None):
        if self.opt.testType=='df':
            if isinstance(attacked_output,dict):
                for model_name in attacked_output:
                    attacked_output[model_name]=attacked_output[model_name].clone()
                    original_output[model_name]=original_output[model_name].clone()
            else:
                attacked_output=attacked_output.clone()
                original_output=original_output.clone()
            loss=0
            for name in self.model:
                if name=='megagan':
                    # attacked_feature,original_feature=self.model[name].get_DFeature(attacked_output,original_output)
                    # for feaD1,feaD2 in zip(attacked_feature,original_feature):
                    #     for fea1,fea2 in zip(feaD1,feaD2):
                    #         loss+=self.loss_fn(fea1,fea2)
                    attacked_feature=self.model[name].get_inter_feats(attacked_input.clone(),img_att.clone())
                    original_feature=self.model[name].get_inter_feats(original_input.clone(),img_att.clone())
                    # for feaD1,feaD2 in zip(attacked_feature,original_feature):
                        # for fea1,fea2 in zip(feaD1,feaD2):
                    # print(attacked_feature.size())
                    # exit()
                    l1loss_fn= nn.L1Loss()
                    loss+=l1loss_fn(attacked_feature,original_feature)
                    # loss+=self.l2loss_fn(attacked_feature,original_feature)
                if name=='simswap':
                    attacked_feature=self.model[name].get_inter_feats(attacked_input.clone(),img_att.clone())
                    original_feature=self.model[name].get_inter_feats(original_input.clone(),img_att.clone())
                    for feaD1,feaD2 in zip(attacked_feature,original_feature):
                        # for fea1,fea2 in zip(feaD1,feaD2):
                        print(feaD1.size(),feaD2.size())
                        l1loss_fn=nn.L1Loss()
                        loss+=l1loss_fn(feaD1,feaD2)
                        # loss+=self.l2loss_fn(feaD1,feaD2)
                if name=='faceshifter':
                    attacked_feature=self.model[name].get_inter_feats(attacked_input.clone(),img_att.clone())
                    original_feature=self.model[name].get_inter_feats(original_input.clone(),img_att.clone())
                    for feaD1,feaD2 in zip(attacked_feature,original_feature):
                        # for fea1,fea2 0in zip(feaD1,feaD2):
                        l1loss_fn= nn.L1Loss()
                        loss+=l1loss_fn(feaD1,feaD2)
                        # loss+=self.l2loss_fn(feaD1,feaD2)
            return loss
        # elif isinstance(attacked_output,dict)   :
        #     loss=0
        #     for model_name in attacked_output:
        #         attacked_output[model_name]=attacked_output[model_name].clone()
        #         original_output[model_name]=original_output[model_name].clone()
        #         if model_name=='megagan':
        #             attacked_feature,original_feature=self.model.get_DFeature(attacked_output[model_name],original_output[model_name])
        #             for feaD1,feaD2 in zip(attacked_feature,original_feature):
        #                 for fea1,fea2 in zip(feaD1,feaD2):
        #                     loss+=self.loss_fn(fea1,fea2)

        #         if model_name=='simswap':
        #             pass

        else:
            attacked_output=attacked_output.clone()
            original_output=original_output.clone()

            attacked_feature,original_feature=self.model.get_DFeature(attacked_output,original_output)
            loss=0
            for feaD1,feaD2 in zip(attacked_feature,original_feature):
                for fea1,fea2 in zip(feaD1,feaD2):
                    loss+=self.loss_fn(fea1,fea2)
            return loss
    def landmark_loss(self,attacked_input,objected_input):
        attacked_input=attacked_input.clone()
        objected_input=objected_input.clone()
        attacked_landmark=self.landmark_model(attacked_input)
        objected_landmark=self.landmark_model(objected_input)
        return self.l2loss_fn(attacked_landmark,objected_landmark)
    def get_attacked_output_from_attacked_source(self,img_id,img_att,model):
        
        if self.opt.lossType!='ensemble_wb_test' and self.opt.testType!='df':
            # img_fake=self.model(img_id,img_att)
            # return img_fake

            img_fakes={}
            for name, fs_model in model.items():
                # img_fakes.append(self.model[name](img_id.img_att))
                
                img_fakes[name]=self.model[name].depreprocess(self.model[name](img_id,img_att))
            assert isinstance(img_fakes,dict)
            return img_fakes
        elif self.opt.lossType=='ensemble_wb_test' and (self.opt.testType=='df' or self.opt.testType=='id'):
            img_fakes={}
            for name, fs_model in model.items():
                # img_fakes.append(self.model[name](img_id.img_att))
                
                img_fakes[name]=self.model[name].depreprocess(self.model[name](img_id,img_att))
            assert isinstance(img_fakes,dict)
            return img_fakes
        # elif self.opt.lossType=='ensemble_wb_test' and self.opt.testType=='id':
        #     img_fake=self.model(img_id,img_att)
        #     return img_fake
    def grad_of_preprocess(self,grad):
       
        shift=nn.Conv2d(grad.size()[1],grad.size()[1],kernel_size=13,stride=1,padding=6,bias=False).to(self.device)
        
        kernel=torch.zeros((shift.weight.data.size()))

        kernel=torch.zeros((shift.weight.data.size()))
        # kernelB=torch.zeros((shiftB.weight.data.size()))
        center_x=int((kernel.size()[-2]-1)/2)
        center_y=int((kernel.size()[-1]-1)/2)
        for i in range(grad.size()[1]):
            kernel[i,i,:,:]=torch.ones((kernel.size()[-2],kernel.size()[-1]))/torch.sum(torch.ones((kernel.size()[-2],kernel.size()[-1])))
            #################################
            
                
            # for j in range(kernel.size()[-2]):
            #     for k in range(kernel.size()[-1]):
            #         kernel[i,i,j,k]=(1-abs(j-center_x)/kernel.size()[-2])*(1-abs(k-center_y)/kernel.size()[-1])

                    # sigma=kernel.size()[-2]/math.sqrt(3)

                    # kernel[i,i,j,k]=1/(2*math.pi*sigma*sigma)*math.exp(-((j-center_x)*(j-center_x)+(k-center_y)*(k-center_y))/(2*sigma*sigma))
                    # kernel[i,i,j,k]=1/(max(abs(j-center_x),abs(k-center_y))+1)
            # for j in range(int(kernel.size()[-2])):
            #     for k in range(int(kernel.size()[-1])):
            #         kernel[i,i,j,k]=(1- abs(j-int(kernel.size()[-2]/2))/(int(kernel.size()[-2]/2)+1))*(1-abs(k-int(kernel.size()[-1]/2))/(int(kernel.size()[-1]/2)+1))

            kernel[i,i,:,:]=kernel[i,i,:,:]/torch.sum(kernel[i,i,:,:])
        # print('kernel',kernel[0,0,...])
        # exit()
        # kernel=torch.cat(kernel,dim=0)
        # shift.weight.data=torch.ones((shift.weight.data.size())).to(self.device)/(torch.sum(torch.ones((shift.weight.data.size()))).to(self.device)/grad.size()[1])
        shift.weight.data=kernel.to(self.device)
        resizes=[]
        # for i in range(100):
        #     resizes.append(transforms.Compose([transforms.Resize(200+int((random.random()-0.5)*100/2)*2)]))
        # normalize=transforms.Normalize([0,0,0],[0.5,0.5,0.5])
        normalizes=[]
        # print('before',grad)
        for i in range(100):
            normalizes.append(0.5+(random.random()-0.5)*0.1)
        grad=shift(grad)
        # print(grad)
        # grad_sum=0
        # for resize in resizes:
        #     grad_sum+=resize(grad)
        # grad=grad_sum/len(resizes)
        # grad=normalize(grad)

        grad_sum=0
        for denormalize in normalizes:
            grad_sum+=grad/denormalize
        grad=grad_sum/len(normalizes)
        return grad
        pass
    def grad_of_dynamic_preprocess(self,grad,gradient_of_input):
        
            # print('gradient of input',torch.mean(torch.abs(gradient_of_input)))
            # gradient_of_input=torch.clamp(torch.abs(gradient_of_input)*10,min=0,max=1)
            gradient_of_input=torch.abs(gradient_of_input)
            # gradient_of_input=torch.ones(gradient_of_input.size()).to(self.device)
            if self.TI_kernel==None:
                shiftA=nn.Conv2d(grad.size()[1],grad.size()[1],kernel_size=13,stride=1,padding=6,bias=False).to(self.device)
                shiftB=nn.Conv2d(grad.size()[1],grad.size()[1],kernel_size=13,stride=1,padding=6,bias=False).to(self.device)
                kernelA=torch.zeros((shiftA.weight.data.size()))
                kernelB=torch.zeros((shiftB.weight.data.size()))
                center_x=int((kernelA.size()[-2]-1)/2)
                center_y=int((kernelA.size()[-1]-1)/2)
                for i in range(grad.size()[1]):
                    for j in range(kernelA.size()[-2]):
                        for k in range(kernelA.size()[-1]):
                            kernelA[i,i,j,k]=max(abs(j-center_x),abs(k-center_y))/(max(abs(j-center_x),abs(k-center_y))+1)
                            kernelB[i,i,j,k]=1/(max(abs(j-center_x),abs(k-center_y))+1)
                            # kernel[i,i,:,:]=torch.ones((kernel.size()[-2],kernel.size()[-1]))/torch.sum(torch.ones((kernel.size()[-2],kernel.size()[-1])))
                            # for j in range(int(kernel.size()[-2])):
                            #     for k in range(int(kernel.size()[-1])):
                            #         kernel[i,i,j,k]=(1- abs(j-int(kernel.size()[-2]/2))/(int(kernel.size()[-2]/2)+1))*(1-abs(k-int(kernel.size()[-1]/2))/(int(kernel.size()[-1]/2)+1))

                    # kernelA[i,i,:,:]=kernelA[i,i,:,:]/torch.sum(kernelA[i,i,:,:])
                    # kernelB[i,i,:,:]=kernelB[i,i,:,:]/torch.sum(kernelB[i,i,:,:])
            # print('kernel',kernel[0,0,...])
            # exit()
            # kernel=torch.cat(kernel,dim=0)
            # shift.weight.data=torch.ones((shift.weight.data.size())).to(self.device)/(torch.sum(torch.ones((shift.weight.data.size()))).to(self.device)/grad.size()[1])
                shiftA.weight.data=kernelA.to(self.device)
                shiftB.weight.data=kernelB.to(self.device)
                self.TI_kernel=(shiftA,shiftB)
            resizes=[]
            # for i in range(100):
            #     resizes.append(transforms.Compose([transforms.Resize(200+int((random.random()-0.5)*100/2)*2)]))
            # normalize=transforms.Normalize([0,0,0],[0.5,0.5,0.5])
            normalizes=[]
            # print('before',grad)
            for i in range(100):
                normalizes.append(0.5+(random.random()-0.5)*0.1)
            # grad=shift(grad)
            gradA=self.TI_kernel[0](grad)
            gradB=self.TI_kernel[1](grad)
            processed_grad=gradA*gradient_of_input+gradB
            

            kernel_norm_term=torch.zeros((grad.size()[1])).to(self.device)
            for i in range(self.TI_kernel[0].weight.data.size()[0]):
                kernel_norm_term[i]+=torch.sum(self.TI_kernel[0].weight.data[i,i,:,:])
            kernel_norm_term=kernel_norm_term.view(1,3,1,1).expand(gradient_of_input.size())
            kernel_norm_term=self.TI_kernel[0].weight.data.size()[-2]*self.TI_kernel[0].weight.data.size()[-1]*gradient_of_input+kernel_norm_term*(1-gradient_of_input)

            # kernel_norm_term_A = torch.zeros((grad.size()[1])).to(self.device)
            # for i in range(self.ATI_kernel[0].weight.data.size()[0]):
            #     kernel_norm_term_A[i] += torch.sum(self.ATI_kernel[0].weight.data[i,i,:,:])

            # kernel_norm_term_B = torch.zeros((grad.size()[1])).to(self.device)
            # for i in range(self.ATI_kernel[1].weight.data.size()[0]):
            #     kernel_norm_term_B[i] += torch.sum(self.ATI_kernel[1].weight.data[i,i,:,:])

            # kernel_norm_term_A = kernel_norm_term_A.view(1,3,1,1).expand(gradient_of_input.size())
            # kernel_norm_term_B = kernel_norm_term_B.view(1,3,1,1).expand(gradient_of_input.size())
            # kernel_norm_term = kernel_norm_term_B + kernel_norm_term_A * (gradient_of_input)


            processed_grad=processed_grad/kernel_norm_term

            # print(grad)
            # grad_sum=0
            # for resize in resizes:
            #     grad_sum+=resize(grad)
            # grad=grad_sum/len(resizes)
            # grad=normalize(grad)

            grad_sum=0
            for denormalize in normalizes:
                grad_sum+=processed_grad/denormalize
            processed_grad=grad_sum/len(normalizes)
            return processed_grad
            pass
    

    
    def rgb2yCbCr(self,input_im):
        original_size=input_im.size()
        im_flat=input_im.contiguous().view(-1,3).float()
        mat=torch.tensor([[0.299,0.587,0.114],[-0.14713,-0.28886,0.436],[0.615,-0.51499,-0.10001]]).to(self.device)
        temp=im_flat.mm(mat)
        out=temp.view(original_size)
        return out
    def yCbCr2rgb(self,input_im):
        original_size=input_im.size()
        im_flat=input_im.contiguous().view(-1,3).float()
        mat=torch.tensor([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]]).to(self.device)
        temp=im_flat.mm(mat)
        out=temp.view(original_size)
        return out
    def simulate_normalization(self,tmpX,simn):
        normalized_Xs=[]
        # simulated_normalizer=[transforms.Compose([transforms.Normalize((np.array([0.485, 0.456, 0.406])+(np.array([0.5,0.5,0.5])-np.array([0.485, 0.456, 0.406]))*k/simn).tolist(),(np.array([0.229, 0.224, 0.225])+(np.array([0.5,0.5,0.5])-np.array([0.229, 0.224, 0.225]))*k/simn).tolist())]) for k in range(simn)]
        # simulated_normalizer=[transforms.Compose([transforms.CenterCrop(256),
        #                                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        #                                         transforms.CenterCrop(237-19)])]
        # simulated_normalizer=[transforms.Compose([transforms.CenterCrop(200+int(random.random()*100)), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),transforms.CenterCrop(200+int(random.random()*100))]) for k in range(simn)]
        simulated_normalizer=[transforms.Compose([transforms.Resize((200+int((random.random()-0.5)*100/2)*2)), 
                    # transforms.Normalize((0.5+((random.random()-0.5)*0.1),0.5+((random.random()-0.5)*0.1),0.5+((random.random()-0.5)*0.1)),(0.5+((random.random()-0.5)*0.1),0.5+((random.random()-0.5)*0.1),0.5+((random.random()-0.5)*0.1))),
                    transforms.Normalize((0+((random.random()-0.5)*0.1),0+((random.random()-0.5)*0.1),0+((random.random()-0.5)*0.1)),(1+((random.random()-0.5)*0.1),1+((random.random()-0.5)*0.1),1+((random.random()-0.5)*0.1))),
                    transforms.CenterCrop(200+int((random.random()-0.5)*100/2)*2)]) for k in range(simn)]

        # simulated_normalizer=[transforms.Compose([transforms.Resize()])]
        if torch.sum(torch.isnan(tmpX))>0:
            print('before clone tmpX has its problem')
        for normalizer in simulated_normalizer:
            # print('normalizer: ',normalizer)
            if torch.sum(torch.isnan(normalizer(tmpX.clone())))>0:
                print("in preprocess",normalizer)
                # print('grad',torch.sum(tmpX.grad))
                if torch.sum(torch.isnan(tmpX))>0:
                    print('tmpX has its problem')
                    exit(0)
                for i, t in enumerate(normalizer.transforms):
                    tmpX=t(tmpX.clone())
                    if torch.sum(torch.isnan(tmpX))>0:
                        print('index',i,t)
                        exit()
                exit(0)
            normalized_Xs.append(normalizer(tmpX.clone()))
        return normalized_Xs
    def flatten2matrix(self,flv,input_size):
        mapped_index=[]
        flv=flv.detach().cpu()
        size=1
        for i in range(len(input_size)):
            size*=input_size[i]
        for i in range(len(input_size)):
            size/=input_size[i]
            if i==(len(input_size)-1):
                mapped_index.append(flv.long().tolist())
            else:
                ith_index=torch.floor(flv/size)
                mapped_index.append(ith_index.long().tolist())
                # print(flv)
                # print(ith_index)
                flv=flv-ith_index.float()*size
        return mapped_index
    def admin_perturb_with_different_size(self,img_s,img_d,y,image1_path,image2_path):
        img_s=transformer(img_s)
        img_d=transformer(img_d)
        # print('img_d',img_d.shape)
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()
        
        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  
        original_X=X.clone()
        original_X=preprocess_for_different_size(original_X)
        img_d=preprocess_for_different_size(img_d)
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        attack_optimizer=optim.SGD([{'params': [X], 'lr': self.a}])
        scheduler = StepLR(attack_optimizer, step_size=50, gamma=0.1)
        # X.requires_grad=True
        with torch.enable_grad():
            for i in range(self.k):
                record=False
                if i%(100-1)==0 and i!=0:
                    record=True
                attack_optimizer.zero_grad()

                y=y.detach()
                X = X.cuda()
                X.requires_grad=True

                self.model.netArc.zero_grad()
                self.top_model.zero_grad()
                self.model.zero_grad()

                tmpX=preprocess_for_different_size(X)
                tmpX_list=self.simulate_normalization(tmpX,10)
                loss=0

                output_list=[]
                for tmpX in tmpX_list:
                    img_att = img_att.cuda()
                    # img_id_downsample = F.interpolate(tmpX, scale_factor=0.5)
                    img_id_downsample=F.interpolate(tmpX,(112,112))
                    # print('img_id_downsample ',img_id_downsample.size())
                    latend_id = self.model.netArc(img_id_downsample)
                    latend_id = latend_id
                    latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                    latend_id = latend_id.to('cuda')
                    # latend_id=0.1*torch.ones(latend_id.size()).to(self.device) # to test 0 id
                    output = self.model(tmpX, img_att, latend_id, latend_id, True)
                    if record:
                        output_list.append(output.clone().detach().cpu()[0])
                
                    
                    if self.opt.lossType=='output_dist':
                        loss+=50*self.output_pixel_loss(output,y)
                    elif self.opt.lossType=='output_id':
                        loss+=1000*self.output_id_loss(output,y,False)
                    elif self.opt.lossType=='internal_id':
                        loss+=2000*self.internal_id_loss(tmpX,original_X.detach())
                        loss+=2000*self.internal_id_loss(tmpX,output)
                    elif self.opt.lossType=='manipulate_id':
                        loss+=200*self.mapulate_id_loss(tmpX)
                    elif self.opt.lossType=='disc_loss':
                        loss+=1000*self.discriminator_loss(output)
                    elif self.opt.lossType=='fr_loss':
                        loss+=10*self.face_detection_loss(output)
                    elif self.opt.lossType=='manipulate_output' or self.opt.lossType=='mo_blur' or self.opt.lossType=='inverted_output' or self.opt.lossType=='increase_var_output':
                        loss+=1000*self.mapulate_output_loss(output,original_output=y,latend_id=latend_id,img_att=img_att)
                    elif self.opt.lossType=='fm_loss':
                        loss+=100*self.feature_match_loss(output,y)
                    elif self.opt.lossType=='ensemble_fr_loss':
                        loss+=2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                    else:
                        print('there is no matched lossType')
                        exit()
                loss=-loss/len(tmpX_list)
                loss.backward()
                grad = X.grad
                
                # X_adv = X + self.a * grad/torch.norm(grad)
                # X_adv = X + self.a * grad.sign()
                attack_optimizer.step()
                scheduler.step()

                # print(X-img_id)
                # X_adv=X
                eta = torch.clamp(X - img_id, min=-self.epsilon, max=self.epsilon)
                print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                # print('eta',torch.mean(eta))
                if torch.norm(eta,p=1)<0.0005:
                    break
                X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                # X.grad=None
                # print('input size',X.size())
                # exit()

                if record:
                    output_list=torch.cat(output_list,dim=1).detach()
                    output_list=output_list.permute(1,2,0)
                    output_list=output_list.numpy()[:,:,::-1]*255
                    cv2.imwrite(os.path.join(self.opt.output_path,image1_path+' '+image2_path+' output_iter_'+str(i)+'.jpg'),output_list)

            self.model.zero_grad()

            test_output=self.get_attacked_output_from_attacked_source(img_id,img_att,self.model)
            result_index=self.internal_id_loss(X,test_output)
        return X.detach().cpu(), X.detach().cpu() - img_id.detach().cpu(),output.detach().cpu(), result_index.detach().cpu().item()
    def admin_perturb_with_different_size_YUV(self,img_s,img_d,y,image1_path,image2_path):
        img_s=transformer(img_s)
        img_d=transformer(img_d)
        # print('img_d',img_d.shape)
        # exit()
        # img_s=self.convert_to_YUV(img_s.view(-1,img_s.shape[1], img_s.shape[2], img_s.shape[0]))
        img_id = self.rgb2yCbCr(img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda())
        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  
        

        delta_x=torch.zeros(X.size()).cuda()
        original_X=X.clone()
        original_X=preprocess_for_different_size(colors.yuv_to_rgb(original_X))
        img_d=preprocess_for_different_size(img_d)
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        attack_optimizer=optim.SGD([{'params': [delta_x], 'lr': self.a}])
        scheduler = StepLR(attack_optimizer, step_size=50, gamma=0.1)

        attack_mask=torch.ones(X.size()).cuda()
        attack_mask[:,0,:,:]=torch.zeros(attack_mask[:,0,:,:].size()).cuda()
        # X.requires_grad=True
        with torch.enable_grad():
            for i in range(self.k):
                record=False
                if i%(100-1)==0 and i!=0:
                    record=True
                attack_optimizer.zero_grad()

                y=y.detach()
                X = X.cuda()
                delta_x=delta_x.cuda()
                X.requires_grad=True
                delta_x.requires_grad=True
                X_with_noise=X+delta_x*attack_mask

                X_with_noise=self.yCbCr2rgb(X_with_noise.clone())
                self.model.netArc.zero_grad()
                self.top_model.zero_grad()
                self.model.zero_grad()

                tmpX=preprocess_for_different_size(X_with_noise)
                tmpX_list=self.simulate_normalization(tmpX,10)
                loss=0

                output_list=[]
                for tmpX in tmpX_list:
                    img_att = img_att.cuda()
                    # img_id_downsample = F.interpolate(tmpX, scale_factor=0.5)
                    img_id_downsample=F.interpolate(tmpX,(112,112))
                    # print('img_id_downsample ',img_id_downsample.size())
                    latend_id = self.model.netArc(img_id_downsample)
                    latend_id = latend_id
                    latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                    latend_id = latend_id.to('cuda')
                    # latend_id=0.1*torch.ones(latend_id.size()).to(self.device) # to test 0 id
                    output = self.model(tmpX, img_att, latend_id, latend_id, True)
                    if record:
                        output_list.append(output.clone().detach().cpu()[0])
                
                    
                    if self.opt.lossType=='output_dist':
                        loss+=50*self.output_pixel_loss(output,y)
                    elif self.opt.lossType=='output_id':
                        loss+=1000*self.output_id_loss(output,y,False)
                    elif self.opt.lossType=='internal_id':
                        loss+=2000*self.internal_id_loss(tmpX,original_X.detach())
                        loss+=2000*self.internal_id_loss(tmpX,output)
                    elif self.opt.lossType=='manipulate_id':
                        loss+=200*self.mapulate_id_loss(tmpX)
                    elif self.opt.lossType=='disc_loss':
                        loss+=1000*self.discriminator_loss(output)
                    elif self.opt.lossType=='fr_loss':
                        loss+=10*self.face_detection_loss(output)
                    elif self.opt.lossType=='manipulate_output' or self.opt.lossType=='mo_blur' or self.opt.lossType=='inverted_output' or self.opt.lossType=='increase_var_output':
                        loss+=1000*self.mapulate_output_loss(output,original_output=y,latend_id=latend_id,img_att=img_att)
                    elif self.opt.lossType=='fm_loss':
                        loss+=100*self.feature_match_loss(output,y)
                    elif self.opt.lossType=='ensemble_fr_loss':
                        loss+=2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                    else:
                        print('there is no matched lossType')
                        exit()
                loss=-loss/len(tmpX_list)
                loss.backward()
                # grad = X.grad
                grad=delta_x.grad
                # X_adv = X + self.a * grad/torch.norm(grad)
                # X_adv = X + self.a * grad.sign()
                attack_optimizer.step()
                scheduler.step()
                # X_adv=X
                # print(grad)
                # print(delta_x)
                eta = torch.clamp(delta_x, min=-self.epsilon, max=self.epsilon)
                print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                # exit()
                # print('eta',torch.mean(eta))
                if torch.norm(eta,p=1)<0.0005:
                    print('eta is too small')
                    break
                # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                delta_x.data= self.rgb2yCbCr((torch.clamp(img_id + self.yCbCr2rgb(eta.detach()), min=-1, max=1)-img_id )).detach()
                # X.grad=None
                # print('input size',X.size())
                # exit()

                if record:
                    output_list=torch.cat(output_list,dim=1).detach()
                    output_list=output_list.permute(1,2,0)
                    output_list=output_list.numpy()[:,:,::-1]*255
                    cv2.imwrite(os.path.join(self.opt.output_path,image1_path+' '+image2_path+' output_iter_'+str(i)+'.jpg'),output_list)
            

            self.model.zero_grad()

            test_output=self.get_attacked_output_from_attacked_source(img_id,img_att,self.model)
            result_index=self.internal_id_loss(X_with_noise,test_output)

        attack_source_image=X_with_noise
        return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),output.detach().cpu(), result_index.detach().cpu().item()
    
    def admin_perturb_with_different_size_facemask(self,img_s,img_d,y,image1_path,image2_path,target_3pimage=None):
        img_s=transformer(img_s)
        img_d=transformer(img_d)
        # print('img_d',img_d.shape)
        # exit()
        # img_s=self.convert_to_YUV(img_s.view(-1,img_s.shape[1], img_s.shape[2], img_s.shape[0]))
        # img_id = self.rgb2yCbCr(img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda())
        img_id=img_s.view(-1,img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()
        # img_s = transformer_Arcface(img_s)
        if self.rand:
            # X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-0.01, 0.01, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  
        

        delta_x=torch.zeros(X.size()).cuda()
        original_X=X.clone()
        original_X=preprocess_for_different_size((original_X))
        img_d=preprocess_for_different_size(img_d)
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        if self.opt.lossType=='push_to_3p_fr_loss':
            target_3pimage=transformer(target_3pimage)
            target_3pimage=preprocess_for_different_size(target_3pimage)
            target_3pimage=target_3pimage.view(-1, target_3pimage.shape[0], target_3pimage.shape[1], target_3pimage.shape[2]).cuda()


        attack_optimizer=optim.SGD([{'params': [delta_x], 'lr': self.a}])
        scheduler = StepLR(attack_optimizer, step_size=50, gamma=0.1)


        attack_mask=torch.Tensor(self.faceParsing_model.return_mask(img_id,self.total_mask,['background','skin']))
        attack_mask=transforms.Resize(X.size()[-2])((attack_mask.expand(X.size()[0],3,attack_mask.size()[0],attack_mask.size()[1]))).cuda()
        attack_mask=mask_transform(attack_mask)*10
        # print('attack mask size',attack_mask.size(),X.size())
        # exit(0)
        # cv2.imwrite('attack_mask.png',attack_mask.detach().cpu().numpy()[0,:,:,:].transpose(1,2,0))
        original_attack_mask=attack_mask.clone()
        # print(attack_mask.detach().cpu().numpy())
        # exit()
        if self.total_mask==False:
            eta_mask=attack_mask.clone()
            eta_mask=torch.div(1,eta_mask)
    
        with torch.enable_grad():
            for i in range(self.k):
                record=False
                if i%(100-1)==0 and i!=0:
                    record=True
                attack_optimizer.zero_grad()

                y=y.detach()
                X = X.cuda()
                delta_x=delta_x.cuda()
                X.requires_grad=True
                delta_x.requires_grad=True
                # if torch.equal(attack_mask,original_attack_mask)==False:
                #     print('attack mask is updated')
                #     exit()
                if self.total_mask:
                    X_with_noise=X+delta_x*attack_mask
                else:
                    X_with_noise=X+delta_x
                # X_with_noise=self.yCbCr2rgb(X_with_noise.clone())
                self.model.netArc.zero_grad()
                self.top_model.zero_grad()
                self.model.zero_grad()

                tmpX=preprocess_for_different_size(X_with_noise)
                tmpX_list=self.simulate_normalization(tmpX,10)
                loss=0

                output_list=[]
                for tmpX in tmpX_list:
                    img_att = img_att.cuda()
                    # img_id_downsample = F.interpolate(tmpX, scale_factor=0.5)
                    img_id_downsample=F.interpolate(tmpX,(112,112))
                    # print('img_id_downsample ',img_id_downsample.size())
                    latend_id = self.model.netArc(img_id_downsample)
                    latend_id = latend_id
                    latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                    latend_id = latend_id.to('cuda')
                    # latend_id=0.1*torch.ones(latend_id.size()).to(self.device) # to test 0 id
                    output = self.model(tmpX, img_att, latend_id, latend_id, True)
                    if record:
                        output_list.append(output.clone().detach().cpu()[0])
                
                    
                    if self.opt.lossType=='output_dist':
                        loss+=50*self.output_pixel_loss(output,y)
                    elif self.opt.lossType=='output_id':
                        loss+=1000*self.output_id_loss(output,y,False)
                    elif self.opt.lossType=='internal_id':
                        loss+=2000*self.internal_id_loss(tmpX,original_X.detach())
                        # loss+=2000*self.internal_id_loss(tmpX,output)
                    elif self.opt.lossType=='manipulate_id':
                        loss+=200*self.mapulate_id_loss(tmpX)
                    elif self.opt.lossType=='disc_loss':
                        loss+=1000*self.discriminator_loss(output)
                    elif self.opt.lossType=='fd_loss':
                        loss+=10*self.face_detection_loss(output)
                    elif self.opt.lossType=='manipulate_output' or self.opt.lossType=='mo_blur' or self.opt.lossType=='inverted_output' or self.opt.lossType=='increase_var_output':
                        loss+=1000*self.mapulate_output_loss(output,original_output=y,latend_id=latend_id,img_att=img_att)
                    elif self.opt.lossType=='fm_loss':
                        loss+=100*self.feature_match_loss(output,y)
                    elif self.opt.lossType=='face_classification_loss':
                        loss+=-1000*self.face_classification_loss(tmpX,original_X.detach(),self.opt.facial_cls_type)
                    elif self.opt.lossType=='ensemble_fr_loss':
                        # loss+=2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        loss+=-200*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        
                        # loss+=-2000*self.ensemble_face_recognition_loss(tmpX,img_att.detach())
                    elif self.opt.lossType=='push_to_3p_fr_loss':
                        loss+=200*self.ensemble_face_recognition_loss(tmpX,target_3pimage.detach())
                        loss+=-200*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='landmark_loss':
                        loss+=100000*self.landmark_loss(tmpX,original_X.detach())
                    else:
                        print('there is no matched lossType')
                        exit()

                loss=-loss/len(tmpX_list)

                
                print('attack loss: ',loss)
                if self.hard_constrain:
                    loss.backward()
                    # grad = X.grad
                    grad=delta_x.grad
                    
                    attack_optimizer.step()
                    scheduler.step()
                    if self.total_mask==True:
                        eta=torch.clamp(delta_x,min=-self.epsilon,max=self.epsilon)
                    if self.total_mask==False:
                        eta = torch.clamp(delta_x*eta_mask, min=-self.epsilon, max=self.epsilon)
                        eta=torch.div(eta,eta_mask)
                    print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    if torch.norm(eta,p=1)<0.0005:
                        print('eta is too small')
                        break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    delta_x.data= (torch.clamp(img_id + eta.detach(), min=-1, max=1)-img_id ).detach()
                    
                else: #soft constraint
                    # loss+=100000*torch.sum(torch.square(torch.log(self.epsilon*eta_mask-delta_x+1))+torch.square(torch.log(self.epsilon*eta_mask+delta_x+1)))
                    
                    loss+=torch.sum(torch.exp((torch.square(delta_x)-self.epsilon*self.epsilon)))+100*self.TVLoss(delta_x)
                    # loss+= torch.sum(1000000*torch.nn.LeakyReLU(negative_slope=0.00001)((torch.sort(torch.square(torch.flatten(delta_x)))[0][:-100]-self.epsilon*self.epsilon)))+100*self.TVLoss(delta_x)
                    # loss=torch.sum(1000000*torch.nn.LeakyReLU(negative_slope=0.00001)((torch.square(delta_x)-self.epsilon*self.epsilon)))+100*self.TVLoss(delta_x)
                    
                    if False:
                        keypoints=torch.where((torch.square(delta_x)-self.epsilon*self.epsilon)>0,torch.ones(delta_x.size()).to(delta_x.device),torch.zeros(delta_x.size()).to(delta_x.device))
                        print('num of keypoints',torch.sum(keypoints))
                        if torch.sum(keypoints)>=1000:
                            delta_x.data= (torch.clamp(img_id + keypoints, min=-1, max=1)-img_id ).detach()
                            if self.total_mask:
                                X_with_noise=X+delta_x*attack_mask
                            else:
                                X_with_noise=X+delta_x
                            break

                    # print(torch.sum(torch.exp(100*(torch.square(delta_x)-self.epsilon*self.epsilon))))
                    # exit(0)
                    loss.backward()
                    # grad = X.grad
                    
                    # delta_x.grad=delta_x.grad/(max(torch.norm(delta_x.grad,p=2),1)/2)
                    grad=delta_x.grad

                    attack_optimizer.step()
                    scheduler.step()
                    eta=delta_x
                    
                    # eta = torch.clamp(delta_x*eta_mask, min=-self.epsilon, max=self.epsilon)
                    # eta=torch.div(eta,eta_mask)
                    print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    if loss>1e+9:
                        print('loss explods')
                        break
                    if torch.norm(eta,p=1)<0.0005:
                        print('eta is too small')
                        break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    delta_x.data= (torch.clamp(img_id + eta.detach(), min=-1, max=1)-img_id ).detach()
                if record:
                    output_list=torch.cat(output_list,dim=1).detach()
                    output_list=output_list.permute(1,2,0)
                    output_list=output_list.numpy()[:,:,::-1]*255
                    cv2.imwrite(os.path.join(self.opt.output_path,image1_path+' '+image2_path+' output_iter_'+str(i)+'.jpg'),output_list)
            

            self.model.zero_grad()

            test_output=self.get_attacked_output_from_attacked_source(img_id,img_att,self.model)
            result_index=self.internal_id_loss(X_with_noise,test_output)
        # X_with_noise=img_id.detach()+delta_x.detach()
        attack_source_image=X_with_noise
        # print((attack_source_image.detach().cpu().numpy() - img_id.detach().cpu().numpy()))
        # exit()
        return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),output.detach().cpu(), result_index.detach().cpu().item()
    def admin_perturb_with_different_size_facemask_noiseFunc(self,img_s,img_d=None,y=None,image1_path=None,image2_path=None,target_3pimage=None,original_input=None):
        # print(origin_img_s.size(),img_s.shape,origin_img_t.size(),img_d.shape,y.size())
        # exit(0)
        # img_s = torch.from_numpy(img_s.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
        # img_d = torch.from_numpy(img_d.copy().transpose((2, 0, 1))).float().mul_(1/255.0).cuda()
        
        # img_s=transformer(img_s).permute(1,2,0).numpy()
        # img_d=transformer(img_d).permute(1,2,0).numpy()
        # print(img_s.shape)
        # img_s=torch.Tensor(img_s.copy()[:,:,::-1].copy()).permute(2,0,1).cuda()
        # img_d=torch.Tensor(img_d.copy()[:,:,::-1].copy()).permute(2,0,1).cuda()
        try:
            img_s=img_s.cuda().float()
            img_d=img_d.cuda().float()
        except:
            pass
        print('imgs',img_s.size())
        # exit()
        assert torch.is_tensor(y)
        if y.dim()==3: y=y.view(-1,y.size()[0],y.size()[1],y.size()[2]).cuda()
        if img_s.dim()==3:img_s=img_s.view(-1,img_s.size()[0],img_s.size()[1],img_s.size()[2]).cuda()
        if (img_d is not None) and img_d.dim()==3:img_d=img_d.view(-1,img_d.size()[0],img_d.size()[1],img_d.size()[2]).cuda()
        img_id=img_s
        img_att=img_d
        
        if self.rand:
            # X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-0.01, 0.01, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
        noise_func=Noise_Func(X.size(),0,0).to(self.device)

        original_X=X.clone()
        # original_X=preprocess_for_different_size((original_X))
        # img_d=preprocess_for_different_size(img_d)
        # img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        if self.opt.lossType=='push_to_3p_fr_loss':
            # target_3pimage=transformer(target_3pimage)
            # target_3pimage=preprocess_for_different_size(target_3pimage)
            # target_3pimage=transformer_for_megagan(target_3pimage)
            assert torch.is_tensor(target_3pimage) and target_3pimage.size()[1]==3
            # target_3pimage=target_3pimage.view(-1, target_3pimage.shape[0], target_3pimage.shape[1], target_3pimage.shape[2]).cuda()
            assert target_3pimage.size()==self.input_size

        attack_optimizer=optim.SGD([{'params': noise_func.parameters(), 'lr': self.a}])
        scheduler = StepLR(attack_optimizer, step_size=8, gamma=0.1)


        attack_mask=torch.Tensor(self.faceParsing_model.return_mask(img_id,self.total_mask,['background'])).to(self.device)
        attack_mask=transforms.Resize(X.size()[-2])((attack_mask.expand(X.size()[0],3,attack_mask.size()[0],attack_mask.size()[1]))).cuda()
        # attack_mask=mask_transform(attack_mask)
        # print('attack mask size',attack_mask.size(),X.size())
        # exit(0)
        # cv2.imwrite('attack_mask.png',attack_mask.detach().cpu().numpy()[0,:,:,:].transpose(1,2,0))
        original_attack_mask=attack_mask.clone()
        # print(attack_mask.detach().cpu().numpy())
        # exit()
        if self.total_mask==False:
            eta_mask=attack_mask.clone()
            eta_mask=torch.div(1,eta_mask)
    
        # X=transformer_for_megagan(X).cuda()
        original_y=y.clone()
        # y=transformer_for_megagan(y).cuda()
        # img_att=transformer_for_megagan(img_att).cuda()
        # assert( X.size()==self.input_size)and(y.size()==self.input_size)and(img_att.size()==self.input_size)
        # assert(y.size()==self.input_size)and(img_att.size()==self.input_size)

        with torch.enable_grad():
            for i in range(self.k):
                record=False
                if i%(100-1)==0 and i!=0:
                    record=True
                attack_optimizer.zero_grad()
                
                y=y.detach()
                X = X.cuda()
                # delta_x=delta_x.cuda()
                # X.requires_grad=True
                # delta_x.requires_grad=True
                # if torch.equal(attack_mask,original_attack_mask)==False:
                #     print('attack mask is updated')
                #     exit()
                if self.total_mask:
                    # X_with_noise=X+delta_x*attack_mask
                    X_with_noise=noise_func(X,attack_mask)
                else:
                    X_with_noise=noise_func(X)
                
                if torch.sum(torch.isnan(X_with_noise))>0:
                    print('X with noise has its problem')
                    if torch.sum(torch.isnan(X))>0:
                        print('X also has its problem')
                    else:
                        print('noise has problem',noise_func.a.data,noise_func.b.data,torch.sum(torch.isnan(noise_func.delta_x)))
                # print(X_with_noise.size())
                # exit()
                # self.model.encoder.zero_grad()
                # self.model.encoder.zero_grad()
                # self.model.encoder.zero_grad()
                tmpX=X_with_noise.clone()
                # tmpX=transforms.Resize(self.input_size[-1])(transformer_for_megagan(X_with_noise))
                tmpX_list=self.simulate_normalization(tmpX,10)
                loss=0
                
                # output1=self.model(img_id,img_att,origin_img_s,origin_img_t)                # print(torch.sum(tmpX.cuda()-origin_img_s.cuda()))
                # print(torch.sum(img_att.cuda()-origin_img_t.cuda()))
                # print(torch.sum(output1.cuda()-original_y.cuda()))
                # exit()
                output_list=[]
                for tmpX in tmpX_list:
                    if torch.sum(torch.isnan(tmpX))>0:
                        print("in loop loss: ",loss,'iter',i)
                        # print('grad',torch.sum(tmpX.grad))
                        exit(0)
                    # output=self.model(img_id+0*X_with_noise)
                    output=self.model(tmpX,img_att)
                    # print(torch.sum(X-img_id))
                    # print(torch.sum(y-output))
                    # exit()
                    if record:
                        output_list.append(output.clone().detach().cpu()[0])
                
                    
                    if self.opt.lossType=='output_dist':
                        loss+=500*self.output_pixel_loss(output,y)
                    elif self.opt.lossType=='output_id':
                        loss+=1000*self.output_id_loss(output,y,False)
                    elif self.opt.lossType=='internal_id':
                        loss+=-2000*self.internal_id_loss(tmpX,original_X.detach())
                        # loss+=2000*self.internal_id_loss(tmpX,output)
                    elif self.opt.lossType=='manipulate_id':
                        loss+=200*self.mapulate_id_loss(tmpX)
                    elif self.opt.lossType=='disc_loss':
                        loss+=1000*self.discriminator_loss(output)
                    elif self.opt.lossType=='fd_loss':
                        loss+=10*self.face_detection_loss(output)
                    elif self.opt.lossType=='manipulate_output' or self.opt.lossType=='mo_blur' or self.opt.lossType=='inverted_output' or self.opt.lossType=='increase_var_output':
                        # loss+=100000000*self.mapulate_output_loss(output,original_output=y,latend_id=latend_id,img_att=img_att)
                        loss+=100000000*self.mapulate_output_loss(output,original_output=y,latend_id=None,img_att=img_att)
                    elif self.opt.lossType=='fm_loss':
                        loss+=100*self.feature_match_loss(output,y)
                    elif self.opt.lossType=='face_classification_loss':
                        loss+=-1000*self.face_classification_loss(tmpX,original_X.detach(),self.opt.facial_cls_type)
                    elif self.opt.lossType=='ensemble_fr_loss':
                        # loss+=2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        loss+=-2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        
                        # loss+=-2000*self.ensemble_face_recognition_loss(tmpX,img_att.detach())
                    elif self.opt.lossType=='push_to_3p_fr_loss':
                        loss+=200*self.ensemble_face_recognition_loss(tmpX,target_3pimage.detach())
                        loss+=-200*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='landmark_loss':
                        loss+=100000*self.landmark_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='ensemble_face_recognition_inter_feature_loss':
                        loss+=200*self.ensemble_face_recognition_inter_feature_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='sp_face_region_loss':
                        loss+=2000*self.sp_face_region_loss(tmpX,original_X.detach(),attack_mask)
                    else:
                        print('there is no matched lossType')
                        exit()

                loss=-loss/len(tmpX_list)

                
                print('attack loss: ',loss)
                if self.hard_constrain:
                    loss.backward()
                    # grad = X.grad
                    grad=noise_func.delta_x.grad
                    
                    attack_optimizer.step()
                    scheduler.step()
                    if self.total_mask==True:
                        eta=torch.clamp(noise_func.delta_x,min=-self.epsilon,max=self.epsilon).cuda().detach()
                        # noise_func.a.data=min(max(noise_func.a.data,-1),1)
                        # noise_func.b.data=min(max(noise_func.b.data,-1),1)
                        # noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                        # noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                        noise_func.update_func()
                    if self.total_mask==False:
                        eta = torch.clamp(noise_func.delta_x*eta_mask, min=-self.epsilon, max=self.epsilon).detach()
                        eta=torch.div(eta,eta_mask)
                    print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    if torch.norm(eta,p=1)<0.0005:
                        print('eta is too small')
                        break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    # print(img_id.device,attack_mask)
                    noise_func.delta_x.data= (torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                    # noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                else: #soft constraint
                    # loss+=100000*torch.sum(torch.square(torch.log(self.epsilon*eta_mask-delta_x+1))+torch.square(torch.log(self.epsilon*eta_mask+delta_x+1)))
                    
                    # loss+=torch.sum(torch.exp((torch.square(noise_func.delta_x)-self.epsilon*self.epsilon)))+10000*self.TVLoss(noise_func.delta_x)
                    
                    # loss+= torch.sum(1000*torch.nn.LeakyReLU(negative_slope=0.001)((torch.sort(torch.square(torch.flatten(noise_func.delta_x)))[0][:-100]-self.epsilon*self.epsilon)))+10000*self.TVLoss(noise_func.delta_x)
                    
                    # loss=torch.sum(1000000*torch.nn.LeakyReLU(negative_slope=0.00001)((torch.square(delta_x)-self.epsilon*self.epsilon)))+100*self.TVLoss(delta_x)
                    
                    if True:
                        keypoints=torch.where((torch.square(noise_func.delta_x)-self.epsilon*self.epsilon)>0,torch.ones(noise_func.delta_x.size()).to(noise_func.delta_x.device),torch.zeros(noise_func.delta_x.size()).to(noise_func.delta_x.device))
                        print('num of keypoints',torch.sum(keypoints))
                        
                        if torch.sum(keypoints)>=1000:
                            topValues,topIndex=torch.sort(torch.square(torch.flatten(noise_func.delta_x)))
                            topIndex=topIndex[-800:]
                            remap_index=self.flatten2matrix(topIndex,noise_func.delta_x.size())
                            # print((remap_index))
                            keypoints_to_attack=torch.zeros(keypoints.size()).to(keypoints.device)
                            keypoints_to_attack[remap_index[0],remap_index[1],remap_index[2],remap_index[3]]=1

                            # noise_func.delta_x.data= (torch.clamp(noise_func(img_id,attack_mask)-noise_func.delta_x.data*attack_mask + keypoints_to_attack, min=-1, max=1)-(noise_func(img_id,attack_mask)-noise_func.delta_x.data*attack_mask) ).detach()
                            noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + keypoints_to_attack.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                            if self.total_mask:
                                # noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                                # noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                                noise_func.update_func()
                                X_with_noise=noise_func(img_id,attack_mask)-noise_func.delta_x.data+noise_func.delta_x*attack_mask
                            else:
                                X_with_noise=noise_func(img_id,attack_mask)-noise_func.delta_x.data+noise_func.delta_x
                                break

                    # print(torch.sum(torch.exp(100*(torch.square(delta_x)-self.epsilon*self.epsilon))))
                    # exit(0)
                    loss.backward()
                    # grad = X.grad
                    
                    # delta_x.grad=delta_x.grad/(max(torch.norm(delta_x.grad,p=2),1)/2)
                    grad=noise_func.delta_x.grad

                    attack_optimizer.step()
                    scheduler.step()
                    eta=noise_func.delta_x
                    
                    # eta = torch.clamp(delta_x*eta_mask, min=-self.epsilon, max=self.epsilon)
                    # eta=torch.div(eta,eta_mask)
                    print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    if loss>1e+9:
                        print('loss explods')
                        break
                    if torch.norm(eta,p=1)<0.0005:
                        print('eta is too small')
                        break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    if self.total_mask:
                        noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                        noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                    noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                if record:
                    output_list=torch.cat(output_list,dim=1).detach()
                    output_list=output_list.permute(1,2,0)
                    output_list=output_list.numpy()[:,:,::-1]*255
                    try:
                        cv2.imwrite(os.path.join(self.opt.output_path,image1_path+' '+image2_path+' output_iter_'+str(i)+'.jpg'),output_list)
                    except:
                        pass        
            # print('noise: ',noise_func.sigmoid(noise_func.a).data,noise_func.sigmoid(noise_func.b).data)
            try:
                print('noise',noise_func.a.data,noise_func.b.data)
            except:
                pass
            self.model.zero_grad()

            test_output=self.get_attacked_output_from_attacked_source(X_with_noise.detach(),img_att,self.model)
            # print(img_id.size(),test_output)
            result_index=self.internal_id_loss(img_id.detach(),test_output,default=False)
        # X_with_noise=img_id.detach()+delta_x.detach()
        attack_source_image=X_with_noise
        # print(torch.sum(X_with_noise-X))
        # exit(0)
        # print((attack_source_image.detach().cpu().numpy() - img_id.detach().cpu().numpy()))
        # exit()
        print('img_id',img_id.size(),attack_source_image.size())
        return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),output.detach().cpu(), result_index.detach().cpu().item()
    
    
    def ensemble_test(self,img_s,img_d,y,image1_path,image2_path,target_3pimage=None):
        print(self.model.keys(),y.keys())
        assert torch.is_tensor(img_s)  and torch.is_tensor(img_d)
        assert (img_s.dim()==4) and (img_d.dim()==4) and (img_s.size()[1]==3) and (img_d.size()[1]==3)
        if self.opt.reshape_to_min:
            img_s_with_original_size=img_s.clone()
            img_s=transforms.Resize((112,112))(img_s)
            img_d=transforms.Resize((112,112))(img_d)
            
            
        if self.opt.lossType=='ensemble_wb_test':
            if self.opt.testType=='id':
                assert torch.is_tensor(y) and (y.dim()== 4) and (y.size()[1]==3) or isinstance(y,dict)
            else:
                assert isinstance(y,dict)
        img_id=img_s.clone()
    
        # if self.rand:
        #     # X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        #     X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-0.01, 0.01, img_id.shape).astype('float32')).to(self.device)
        #     # X = img_id.clone().detach_().to(self.device)
        # else:
        #     X = img_id.clone().detach_().to(self.device)
        #     # use the following if FGSM or I-FGSM and random seeds are fixed        
        
        X = img_id.clone().detach_().to(self.device)
        if not self.opt.gradient_encourage:
            # random_nosie=torch.from_numpy(np.random.uniform(-1/100000000,1/100000000,X.size()).astype('float32')).to(self.device)
            # random_nosie=1/100000000*torch.ones(X.size()).to(self.device)
            random_noise=torch.zeros(X.size()).to(self.device)
            # X+=random_nosie/torch.norm(random_nosie)*100
            X+=random_noise
            X=torch.clamp(X,min=0,max=1)
            print(X)
            print('Xs nan',torch.sum(torch.isnan(X)))
        # X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-0.001, 0.001, img_id.shape).astype('float32')).to(self.device)
        # delta_x=torch.zeros(X.size()).cuda()
        if not self.opt.universal_attack:
            noise_func=Noise_Func(X.size(),None,None).cuda()
        if self.opt.universal_attack:
            if self.noise_func==None:
                self.noise_func=Noise_Func(X.size(),None,None).cuda()
                noise_func=self.noise_func
            else:
                noise_func=self.noise_func



        original_X=X.clone()
        img_att=img_d.clone()
        if self.opt.lossType=='push_to_3p_fr_loss':
            assert torch.is_tensor(target_3pimage) 
            assert (target_3pimage.dim()==4) and (target_3pimage.size()[1]==3)

        if self.opt.dynamic_budget:
            self.epsilon=self.get_img_gradient(original_X)

        parameters_to_learn=[{'params': noise_func.delta_x, 'lr': self.a}]
        # attack_optimizer=optim.SGD([{'params': noise_func.parameters(), 'lr': self.a}])
        
        parameters_to_learn_relighting=[]
        if self.opt.relighting:
            if not self.opt.universal_attack:
                relighting_net=DPR(self.device).cuda()
            else:
                if self.relighting_net==None:
                    self.relighting_net=DPR(self.device).cuda()
                relighting_net=self.relighting_net
            # parameters_to_learn.append({'params': relighting_net.sh, 'lr': self.a, 'weight_decay':0})
            parameters_to_learn_relighting=[{'params': relighting_net.sh, 'lr': self.a}]
            # attack_optimizer=optim.SGD([{'params': noise_func.parameters(), 'lr': self.a},
            #                         {'params': relighting_net.sh, 'lr': self.a}])
            
            
            original_sh=relighting_net.extract_sh(original_X.clone()).detach()
            if self.opt.natural_initial_lighting:
                # print(original_sh[:,2:,...].size())
                # exit()
                original_sh[:,2:,...]=torch.Tensor([0.2,-0.05,-0.05,0,-0.15,0,0.05]).view(original_sh[:,2:,...].size())

            relighting_loss_weight=200000000

            relighting_net.sh.data=original_sh.clone().detach()

            if self.opt.relighting_encourage:
                sh_distribution=SH_distribution()
                sh_distribution=SH_NormalDistribution()
                # sh_distribution=SH_NormalDistribution(initial_mean=original_sh.clone().detach())
            

            
            # print('original_sh',original_sh)
        
        
        if self.opt.adjust_contrast:
            contrast_func=Contrast_func(original_X,10,self.device).cuda()
            parameters_to_learn.append({'params': contrast_func.parameters(), 'lr': self.a})
        
        
        attack_optimizer=optim.SGD(parameters_to_learn)

        if self.opt.relighting:
            relighting_optimizer=optim.SGD(parameters_to_learn_relighting)
        # print(parameters_to_learn)
        
        # scheduler = StepLR(attack_optimizer, step_size=4)
        scheduler_for_noise_fn=StepLR(attack_optimizer,step_size=4,gamma=0.5)##0.5
        if self.opt.relighting:
            scheduler_for_relighting=StepLR(relighting_optimizer,step_size=4,gamma=1)

        
        
        # attack_mask=torch.Tensor(self.faceParsing_model.return_mask(img_id,self.total_mask,['background']))
        # attack_mask=transforms.Resize(X.size()[-2])((attack_mask.expand(X.size()[0],3,attack_mask.size()[0],attack_mask.size()[1]))).cuda()

        attack_mask=torch.ones(X.size()).cuda()

        assert attack_mask.size()==X.size()
        original_attack_mask=attack_mask.clone()

        if self.total_mask==False:
            eta_mask=attack_mask.clone()
            eta_mask=torch.div(1,eta_mask)

        
        with torch.enable_grad():

            # variable for VT
            old_grad=np.zeros(noise_func.delta_x.size())
            old_variance=np.zeros(noise_func.delta_x.size())
            transfer_iter=0
            topk_mask=None

            if self.opt.adapt_iter or self.opt.left_one_out:
                self.k=50
            for i in range(self.k):
                record=False
                if i%(100-1)==0 and i!=0:
                    record=False # not implemented yet
                attack_optimizer.zero_grad()

                # y=y.detach()
                if isinstance(y,dict):
                    for name, value in y.items():
                        y[name].detach()
                else:
                    y=y.detach()
                X = X.cuda()

                X.requires_grad=True

                print('X is nan',torch.sum(torch.isnan(X)))
                
                sh_inter_grad={}
                if self.opt.relighting:
                    # print(X_with_noise.size())
                    relighted_X=relighting_net(X.clone(),original_sh=original_sh,inter_grad=sh_inter_grad)
                    
                    print('relighted_x',torch.sum(torch.isnan(relighted_X)))
                    
                    print('noise has problem',torch.sum(torch.isnan(noise_func.delta_x)),'sh has problem',torch.sum(torch.isnan(relighting_net.sh)))
                    
                if self.opt.adjust_contrast:
                    relighted_X=contrast_func(relighted_X.clone())
                
                if self.total_mask:
                    if self.opt.relighting or self.opt.adjust_contrast:
                        X_with_noise=noise_func(relighted_X,attack_mask)
                    else:
                        X_with_noise=noise_func(X.clone(),attack_mask)
                else:
                    X_with_noise=noise_func(X.clone())

                
                
                
                if torch.sum(torch.isnan(X_with_noise))>0:
                    print('X with noise has its problem')
                    if torch.sum(torch.isnan(X))>0:
                        print('X also has its problem')
                    else:
                        try:
                            print('noise has problem',noise_func.a.data,noise_func.b.data,torch.sum(torch.isnan(noise_func.delta_x)))
                        except:
                            print('noise has problem',torch.sum(torch.isnan(noise_func.delta_x)),'sh has problem',torch.sum(torch.isnan(relighting_net.sh)))
                # self.model.netArc.zero_grad()
                # self.model.zero_grad()
                for name, fs_model in self.model.items():
                    # if name=='simswap':
                    #     self.model[name].netArc.zero_grad()
                    #     self.model[name].zero_grad()
                    # else:
                    #     self.model[name].zero_grad()
                    self.model[name].zero_grad()

                # tmpX=preprocess_for_different_size(X_with_noise)
                tmpX=X_with_noise
                # tmpX_list=self.simulate_normalization(tmpX,3)
                tmpX_list=[tmpX]
                loss=0

                output_list=[]
                
                for tmpX in tmpX_list:
                    
                    assert torch.sum(torch.isnan(tmpX ))==0
                    if torch.sum(torch.isnan(tmpX))>0:
                        print("in loop loss: ",loss,'iter',i)
                        # print('grad',torch.sum(tmpX.grad))
                        exit(0)
                    if not self.opt.testType=='id' and (not self.opt.testAttackType=='integrated_gradient' or self.opt.relighting ) and not self.opt.testAttackType=='VT' or True:
                        output={}
                        
                        for name in self.model.keys():
                            print(name)
                            assert tmpX.dim()==4 and img_att.dim()==4
                            if self.opt.testType=='transfer_attack' and transfer_iter==0:
                                output_of_model=self.model[name](tmpX.clone(),tmpX.clone().detach())
                                output_of_model=self.model[name].depreprocess(output_of_model)
                            else:
                                # cv2.imwrite('advDF/ensemble_test/input_image2.png',img_id.detach().cpu().numpy()[0].transpose(1,2,0)[...,::-1]*255)
                                output_of_model=self.model[name](tmpX.clone(),img_att.clone())
                                output_of_model=self.model[name].depreprocess(output_of_model)
                               
                            assert output_of_model.dim()==4

                            # print('output_of_model',torch.sum(output_of_model))
                            output[name]=output_of_model

                        # output_simswap=self.model['simswap'](tmpX,img_att)
                        # output_simswap=self.model['simswap'].depreprocess(output_simswap)

                        # output_faceshifter=self.model['faceshifter'](tmpX,img_att)
                        # output_faceshifter=self.model['faceshifter'].depreprocess(output_faceshifter)

                        # output_megagan=self.model['megagan'](tmpX,img_att)
                        # output_megagan=self.model['megagan'].depreprocess(output_megagan)

                        # output_agilegan=self.model['agilegan'](tmpX)
                        # output_agilegan=self.model['agilegan'].depreprocess(output_agilegan)
                        # output={'simswap':output_simswap,'faceshifter':output_faceshifter,'megagan':output_megagan,'agilegan':output_agilegan}
                        # if record:
                            # output_list.append([output_simswap,output_faceshifter,output_megagan,output_agilegan].clone().detach().cpu()[0])
            
                    
                    if self.opt.lossType=='output_dist':
                        # loss+=5000000*self.output_pixel_loss(output,y)
                        if (self.opt.testType=='transfer_attack' and transfer_iter==0):
                            original_reconstruct_outputs={}
                            for name,item in output.items():
                                original_reconstruct_outputs[name]=item.detach()
                            loss+=5000000*self.output_pixel_loss(output,original_reconstruct_outputs)
                        elif (self.opt.testType=='transfer_attack' and transfer_iter>0) or self.opt.testType!='transfer_attack':
                            loss+=5000000*self.output_pixel_loss(output,y)
                            # print('loss random nor not',loss,'tmpX {:.10f}'.format(torch.sum(tmpX)))
                            # exit()
                    elif self.opt.lossType=='output_id':
                        loss+=1000*self.output_id_loss(output,y,False)
                    elif self.opt.lossType=='internal_id':
                        loss+=-2000*self.internal_id_loss(tmpX,original_X.detach())
                        # loss+=2000*self.internal_id_loss(tmpX,output)
                    elif self.opt.lossType=='manipulate_id':
                        loss+=200*self.mapulate_id_loss(tmpX)
                    elif self.opt.lossType=='disc_loss':
                        loss+=1000*self.discriminator_loss(output)
                    elif self.opt.lossType=='fd_loss':
                        loss+=10*self.face_detection_loss(output)
                    elif self.opt.lossType=='manipulate_output' or self.opt.lossType=='mo_blur' or self.opt.lossType=='inverted_output' or self.opt.lossType=='increase_var_output':
                        loss+=100000000*self.mapulate_output_loss(output,original_output=y,latend_id=latend_id,img_att=img_att)
                    elif self.opt.lossType=='fm_loss':
                        loss+=100*self.feature_match_loss(output,y)
                    elif self.opt.lossType=='face_classification_loss':
                        loss+=-1000*self.face_classification_loss(tmpX,original_X.detach(),self.opt.facial_cls_type)
                    elif self.opt.lossType=='ensemble_fr_loss':
                        # loss+=2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        loss+=-2000*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                        
                        # loss+=-2000*self.ensemble_face_recognition_loss(tmpX,img_att.detach())
                    elif self.opt.lossType=='push_to_3p_fr_loss':
                        loss+=200*self.ensemble_face_recognition_loss(tmpX,target_3pimage.detach())
                        loss+=-200*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='landmark_loss':
                        loss+=100000*self.landmark_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='ensemble_face_recognition_inter_feature_loss':
                        loss+=200*self.ensemble_face_recognition_inter_feature_loss(tmpX,original_X.detach())
                    elif self.opt.lossType=='sp_face_region_loss':
                        loss+=2000*self.sp_face_region_loss(tmpX,original_X.detach(),attack_mask)
                    elif self.opt.lossType=='ensemble_wb_test':
                        if self.opt.testType=='id':
                            # loss+=-200*self.ensemble_face_recognition_loss(tmpX,original_X.detach())
                            # loss+=-2000*self.internal_id_loss(tmpX,original_X.detach())

                            # # print(loss,-2000*self.internal_id_loss(tmpX,original_X.detach(),default=False))
                            # loss+=-2000*self.internal_id_loss(tmpX,original_X.detach(),default=False)
                            noise_func.delta_x.grad=torch.rand(noise_func.delta_x.size()).to(self.device)
                        elif self.opt.testType=='df':
                            # loss+=50000*self.output_pixel_loss(output,y)
                            # print('i am here')
                            if self.opt.testAttackType=='fr':
                                # loss+=-2000*self.ensemble_face_recognition_loss(output,{'simswap':original_X.detach(),'megagan':original_X.detach(),'faceshifter':original_X.detach(),'agilegan':original_X.detach()})
                                if not self.opt.left_one_out:
                                    frLoss_sum,loss_list=self.ensemble_face_recognition_loss(output,y,i)
                                else:
                                    frLoss_sum,loss_list,last_fr_loss=self.ensemble_face_recognition_loss(output,y,i)
                                    
                                loss+=-2000*frLoss_sum
                                
                                print('using y')
                                # exit()
                            
                            if self.opt.testAttackType=='style':
                                loss+=200000000*self.ensemble_face_recognition_style_loss(output,y)
                            if self.opt.testAttackType=='integrated_gradient':

                                if self.opt.relighting:
                                    def tmp_relighting_backward(relighting_net,output):
                                        # output_clone={}
                                        # X_clone=X.clone()
                                        # relighting_X_clone=relighting_net(X_clone)

                                        y_clone={}
                                        for model_name, individual_output in output.items():
                                            # output_clone[model_name]=individual_output.clone().detach()
                                            y_clone[model_name]=y[model_name].clone().detach()
                                        tmp_loss=-2000*self.ensemble_face_recognition_loss(output,y_clone)
                                        tmp_loss.backward()
                                        # print(y_clone[model_name].grad)
                                        # # print(relighting_net.sh.grad)
                                        # exit(0)
                                        # for model_name in output:
                                        #     del output[model_name],y_clone[model_name]
                                        del tmp_loss, output,y_clone
                                        # print('sh',relighting_net.sh.grad)
                                        # exit()
                                        # relighting_net.sh.grad=relighting_net.sh.grad.detach()
                                    
                                    tmp_relighting_backward(relighting_net,output)
                                    
                                IG,tmploss=self.integrated_gradient(tmpX.detach(),img_att.detach(),y,30)
                                # IG=torch.zeros(tmpX.size()).numpy()
                                # if True:
                                #     IG=torch.from_numpy(IG)

                                #     print(IG)
                                #     print(torch.min(IG),torch.max(IG))
                                #     integrated_grad=torch.abs(IG)
                                #     integrated_grad=integrated_grad-torch.min(integrated_grad)
                                #     integrated_grad=integrated_grad/torch.max(integrated_grad)
                                #     integrated_grad=integrated_grad.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
                                #     print(image1_path,image2_path)
                                #     assert cv2.imwrite('advDF/ensemble_test/Integrated_gradients_graph_abs_random/'+image1_path+'_'+image2_path+'integrated_grad.jpg',integrated_grad)
                                #     print('save gradient image')
                                #     return
                                # print(type(noise_func.delta_x.grad),torch.from_numpy(IG).dtype)
                                
                                    
                                if not self.opt.gradient_encourage:
                                    noise_func.delta_x.grad=torch.sign(torch.from_numpy(IG).to(self.device))
                                else:
                                    noise_func.delta_x.grad=torch.sign(torch.from_numpy(IG).to(self.device))
                                # del IG
                                    # noise_func.delta_x.data-=self.a*torch.from_numpy(IG).to(self.device)
                            if self.opt.testAttackType=='VT':
                                variance,src_grad,loss=self.VT(tmpX.clone(),img_att.clone(),y,20,self.k)
                                # print(noise_func.delta_x.grad)
                                # print(torch.abs(torch.from_numpy(src_grad)+torch.from_numpy(old_variance)).to(self.device))
                                new_direct=(torch.from_numpy(src_grad).to(self.device)+torch.from_numpy(old_variance).to(self.device)).to(self.device)
                                new_direct_abs=torch.abs(torch.from_numpy(src_grad).to(self.device)+torch.from_numpy(old_variance).to(self.device)).to(self.device)
                                updated_grad=self.momentum*torch.from_numpy(old_grad).to(self.device)+torch.where(new_direct_abs==0, torch.zeros(new_direct_abs.size()).to(self.device), (new_direct/new_direct_abs).float().to(self.device)).to(self.device)

                                if torch.sum(torch.isnan(updated_grad))>0:
                                    print('there is nan')
                                    exit(0)

                                # print(torch.where(new_direct_abs==0, 0.0, new_direct/new_direct_abs).to(self.device).size())
                                print(noise_func.delta_x.size())
                                print((torch.zeros(new_direct_abs.size()).to(self.device)).dtype,(new_direct/new_direct_abs).float().to(self.device).dtype )
                                print(type(torch.where(new_direct_abs==0, torch.zeros(new_direct_abs.size()).to(self.device), (new_direct/new_direct_abs).float().to(self.device)).to(self.device)))
                                print(type(noise_func.delta_x.grad))
                                print(updated_grad.dtype)
                                noise_func.delta_x.grad=torch.sign(updated_grad.float())
                                old_grad=noise_func.delta_x.grad.detach().cpu().numpy()
                                old_variance=variance
                            if self.opt.testAttackType=='ensemble_loss':
                                # loss+=self.output_pixel_loss(output,y)+0.5*self.feature_match_loss(output,y,img_d.clone(),tmpX.clone(),img_att.clone())
                                loss+=5000000*self.feature_match_loss(output,y,img_d.clone(),tmpX.clone(),img_att.clone())
                                # loss+=-2000*self.ensemble_face_recognition_loss(output,y)
                                # loss+=-2000*self.face_classification_loss_ver2(output,y)
                                # loss+=-2000000*self.face_classification_loss_ver3(output,y)
                                # loss+=-2000*self.face_classification_loss_ver5(output,y)
                            if self.opt.testAttackType=='fr_multi_source':
                                loss+=-2000*self.ensemble_face_recognition_loss_mutli_sources(output,y,image1_path)
                            # if self.opt.dynamic_budget:
                            #     loss+=-20000*self.l2loss_fn(self.epsilon,noise_func.delta_x)                      
                    else:
                        print('there is no matched lossType')
                        exit()

                    if not self.opt.relighting and not self.opt.adjust_contrast:
                        # loss+=1000000*self.cosloss_fn(noise_func.delta_x.clone().view(noise_func.delta_x.size()[0],-1),self.get_img_gradient(original_X.clone().detach()).view(original_X.size()[0],-1))
                        gradient_of_input=self.get_img_gradient(original_X.clone().detach())
                    else:
                        print('independent attack loss',loss/len(tmpX_list) )
                        gradient_of_input=self.get_img_gradient(relighted_X.clone().detach())
                    
                    if self.opt.gradient_encourage:
                        # loss+=-2000000*self.l2loss_fn(noise_func.delta_x,self.get_img_gradient(original_X.clone().detach()))
                        
                        # loss+=-2000*torch.sum(self.get_img_gradient(original_X.clone().detach()))
                        # print(noise_func.delta_x.size(),self.get_img_gradient(relighted_X.clone().detach()).size())
                        if not self.opt.relighting and not self.opt.adjust_contrast:
                            # loss+=1000000*self.cosloss_fn(noise_func.delta_x.clone().view(noise_func.delta_x.size()[0],-1),self.get_img_gradient(original_X.clone().detach()).view(original_X.size()[0],-1))
                            # gradient_of_input=self.get_img_gradient(original_X.clone().detach())
                            loss+=-1500000*self.l2loss_fn(noise_func.delta_x*attack_mask,gradient_of_input)
                            # loss+=-1500000*self.l2loss_fn(noise_func.delta_x,self.get_img_gradient(original_X.clone().detach()))
                        else:
                            # print('independent attack loss',loss/len(tmpX_list) )

                            # gradient_of_input=self.get_img_gradient(relighted_X.clone().detach())
                            loss+=-1500000*self.l2loss_fn(noise_func.delta_x*attack_mask,gradient_of_input)
                            # loss+=50000*self.cosloss_fn(noise_func.delta_x.clone().view(noise_func.delta_x.size()[0],-1),self.get_img_gradient(relighted_X.clone().detach()).view(relighted_X.size()[0],-1))
                            # print('gradient loss:',50000*self.cosloss_fn(noise_func.delta_x.clone().view(noise_func.delta_x.size()[0],-1),self.get_img_gradient(relighted_X.clone().detach()).view(relighted_X.size()[0],-1)))
                            print('gradient loss',-1500000*self.l2loss_fn(noise_func.delta_x,self.get_img_gradient(relighted_X.clone().detach())))
                    if self.opt.relighting and self.opt.relighting_encourage :
                        def print_grad(name):
                            def grad_func(grad):
                                sh_inter_grad[name]=grad
                                print('biggest grad for '+name+'  is ',torch.max(torch.abs(grad)))
                            return grad_func

                        ##########################################################################################
                        if self.opt.relighting_dist_constraint:
                            cloned_sh=relighting_net.get_sh_input(original_sh=original_sh).view(-1,9).clone()
                            cloned_sh.register_hook(print_grad('natural'))

                            # loss+=0*sh_distribution(relighting_net.get_sh_input().view(-1,9))
                            min_thrd=torch.clamp(sh_distribution.mean-sh_distribution.var*2,min=-1,max=1)
                            max_thrd=torch.clamp(sh_distribution.mean+sh_distribution.var*2,min=-1,max=1)
                            relighting_contrain_loss=200*sh_distribution(cloned_sh)*((relighting_net.sh.data<min_thrd)+(relighting_net.sh.data > max_thrd))*1.0
                            relighting_contrain_loss=torch.sum(relighting_contrain_loss)
                            
                            loss+=relighting_contrain_loss
                            print('relighting_contrain_loss',relighting_contrain_loss)
                        ###############################################################
                        
                        # print(relighting_net.get_sh_input().view(-1,9))

                        ################################################
                        # print('relighting loss',20000*self.l2loss_fn(original_sh.view(-1,9),relighting_net.get_sh_input(original_sh=original_sh).view(-1,9)))
                        
                        
                        cloned_sh=relighting_net.get_sh_input(original_sh=original_sh).view(-1,9)
                        # cloned_sh.register_hook(print_grad('relighting model'))
                        
                        if self.opt.adjust_relighting :
                            print('max loss',max(loss_list).item())
                            
                            if max(loss_list).item()>0.09:
                                # print('yes')
                                relighting_loss_weight/=100
                                if relighting_loss_weight>1e-10:
                                    relighting_loss_weight=0
                                # exit()
                            loss+=-relighting_loss_weight*self.l2loss_fn(original_sh.view(-1,9),cloned_sh)
                        ################################################

                        


                        # loss+= -20000*self.l2loss_fn(original_sh.view(-1,9),relighting_net.get_sh_input().view(-1,9))
                        # loss+=20000*self.l2loss_fn(original_sh.view(-1,9),torch.Tensor([1,0,0.2,-0.05,-0.05,0,-0.2,0,0]).cuda())

                if self.opt.relighting:
                    print('relighting_loss_weight',relighting_loss_weight)

                loss=-loss/len(tmpX_list)

                
                print('attack loss: ',loss)
                # print('mask',torch.sum(attack_mask))
                print('noise max',torch.max(noise_func.delta_x))
                if self.hard_constrain:
                    if (self.opt.testAttackType!='integrated_gradient' or self.opt.gradient_encourage) and self.opt.testAttackType!='VT' and self.opt.testType!='id':
                    # if (self.opt.testAttackType!='integrated_gradient'  ) and self.opt.testAttackType!='VT' and self.opt.testType!='id':
                        loss.backward()
                        # print('relighting sh grad',relighting_net.sh.grad)
                        # if True:
                        #     print(noise_func.delta_x.grad)
                        #     print(torch.min(noise_func.delta_x.grad),torch.max(noise_func.delta_x.grad))

                        #     integrated_grad=torch.abs(noise_func.delta_x.grad)
                        #     integrated_grad=integrated_grad-torch.min(integrated_grad)
                        #     integrated_grad=integrated_grad/torch.max(integrated_grad)
                        #     integrated_grad=integrated_grad.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
                        #     print(image1_path,image2_path)
                        #     assert cv2.imwrite('advDF/ensemble_test/Gradient_graph/'+image1_path+'_'+image2_path+'integrated_grad.jpg',integrated_grad)
                        #     print('save gradient image')
                        #     return
                    # grad = X.grad
                    ######################################
                    grad=noise_func.delta_x.grad
                    # print('before process',noise_func.delta_x.grad)
                    # exit()
                    # if self.opt.testAttackType=='ensemble_loss':
                    #     noise_func.delta_x.grad=torch.sign(noise_func.delta_x.grad)
                    # print('before preprocess',noise_func.delta_x.grad)
                    if self.opt.lossType=='ensemble_wb_test' and (not self.opt.testAttackType=='ensemble_loss') or not self.opt.testAttackType=='integrated_gradient' :
                        # noise_func.delta_x.grad=self.grad_of_preprocess(noise_func.delta_x.grad)
                        # pass
                        noise_func.delta_x.grad=self.grad_of_dynamic_preprocess(noise_func.delta_x.grad,gradient_of_input)
                        # print('delta_x grad after preprocess', noise_func.delta_x.grad )
                        # exit()
                        #############################
                    if self.opt.sign:
                        ########### transfer attack baseline
                        if self.opt.testType=='transfer_attack' and transfer_iter==0:
                            delta_x_size=noise_func.delta_x.size()
                            def top_k_gradient(x_grad):
                                x_grad_tmp=x_grad.clone().view(1,-1)
                                # values,indexes=torch.topk(torch.abs(x_grad_tmp),int(x_grad.size()[-1]*x_grad.size()[-2]/2))
                                values,indexes=torch.topk(torch.abs(x_grad_tmp),5000)
                                index_list=[]
                                # print(indexes)
                                for num_dim in range(x_grad.dim()-1):
                                    x_grad_size=1
                                    for grad_size in x_grad.size()[num_dim+1:]:
                                        x_grad_size*=grad_size
                                    tmp_index=indexes//x_grad_size
                                    print(x_grad_size)
                                    index_list.append(tmp_index)
                                    indexes-=(tmp_index*x_grad_size).long()
                                index_list.append(indexes)
                                return index_list
                            
                            topk_mask=torch.zeros(delta_x_size)
                            topk_index=top_k_gradient(noise_func.delta_x.grad)
                            print(torch.sum(torch.abs(noise_func.delta_x.grad)))
                            print(topk_index)
                            print(noise_func.delta_x.grad[topk_index])
                            topk_mask[topk_index[0],topk_index[1],topk_index[2],topk_index[3]]=1
                            topk_mask=topk_mask.to(self.device)
                            noise_func.delta_x.grad=noise_func.delta_x.grad*topk_mask
                            transfer_iter+=1
                        elif self.opt.testType=='transfer_attack' and transfer_iter>0:
                            noise_func.delta_x.grad=torch.sign(noise_func.delta_x.grad*topk_mask).to(self.device)
                            print(torch.sum(torch.abs(noise_func.delta_x.grad)))
                        ########### transfer attack bsaeline


                        ################################
                        if self.opt.testType!='transfer_attack':
                            noise_func.delta_x.grad=torch.sign(noise_func.delta_x.grad).to(self.device)
                            print(torch.sum(torch.abs(noise_func.delta_x.grad)))
                        ################################
                    # grad=noise_func.delta_x.grad
                    # print('before process',noise_func.delta_x.grad)
                    # # exit()
                    # # if self.opt.testAttackType=='ensemble_loss':
                    # #     noise_func.delta_x.grad=torch.sign(noise_func.delta_x.grad)
                    # # print('before preprocess',noise_func.delta_x.grad)
                    # if self.opt.lossType=='ensemble_wb_test' and (not self.opt.testAttackType=='ensemble_loss') :
                    #     noise_func.delta_x.grad=self.grad_of_preprocess(noise_func.delta_x.grad)
                    #     print('delta_x grad',torch.max(torch.abs(noise_func.delta_x.grad)))
                    ##########################################################################
                    # print('noise_func.delta_x.grad',noise_func.delta_x.grad*0.004)
                        # exit(0)
                    if self.opt.relighting:
                        if sh_inter_grad is not {} and self.opt.relighting_dist_constraint:
                            # sh_combine_grad=0
                            # for sh_grad in sh_inter_grad:
                            #     sh_combine_grad+=sh_grad/torch.linalg.norm(sh_grad)
                            print(torch.linalg.norm(sh_inter_grad['attack']),sh_inter_grad['attack'].size(),sh_inter_grad['natural'].size())
                            print(((sh_inter_grad['attack'].view(relighting_net.sh.size())/torch.linalg.norm(sh_inter_grad['attack'])+0.5*sh_inter_grad['natural'].view(relighting_net.sh.size())/torch.linalg.norm(sh_inter_grad['natural']))*torch.linalg.norm(sh_inter_grad['attack'])).size())
                            relighting_net.sh.grad=(sh_inter_grad['attack'].view(relighting_net.sh.size())/torch.linalg.norm(sh_inter_grad['attack'])+0.5*sh_inter_grad['natural'].view(relighting_net.sh.size())/torch.linalg.norm(sh_inter_grad['natural']))*torch.linalg.norm(sh_inter_grad['attack'])


                        relighting_net.sh.grad=torch.clamp(relighting_net.sh.grad,min=-0.75,max=0.75)
                        # print(relighting_net.sh.grad)
                        # relighting_net.sh.grad=relighting_net.sh.grad/torch.max(torch.abs(relighting_net.sh.grad))

                    if self.opt.adjust_contrast:
                        contrast_func.contrast_parameter_a.grad=torch.clamp(contrast_func.contrast_parameter_a.grad,min=-1,max=1)
                        contrast_func.contrast_parameter_b.grad=torch.clamp(contrast_func.contrast_parameter_b.grad,min=-1,max=1)
                    # print('grad',noise_func.delta_x.grad)
                    # print(attack_optimizer.state_dict())
                    # print(parameters_to_learn[1]['params'][0] is relighting_net.sh)
                    # exit(0)
                    attack_optimizer.step()
                    if self.opt.relighting:
                        relighting_optimizer.step()
                    scheduler_for_noise_fn.step()

                    if self.opt.relighting:
                        scheduler_for_relighting.step()
                    # print('attack optimizer',attack_optimizer)
                    # exit()
                    
                    

                    if self.opt.relighting_encourage and self.opt.hard_constrain_sh and sh_distribution is not None:
                        min_thrd=torch.clamp(sh_distribution.mean-sh_distribution.var*2,min=-1,max=1)
                        max_thrd=torch.clamp(sh_distribution.mean+sh_distribution.var*2,min=-1,max=1)
                        relighting_net.sh.data[:,2:,...] =torch.clamp(relighting_net.sh.data ,min=torch.atanh(min_thrd).view(relighting_net.sh.data.size()),max=torch.atanh(max_thrd).view(relighting_net.sh.data.size()))[:,2:,...]
                        # relighting_net.sh.data[:,2:,...] =torch.clamp(relighting_net.sh.data ,min=torch.atanh(-1) ,max=torch.atanh(1))[:,2:,...]
                    if self.opt.relighting:
                        for index_i in range(9):
                            print('sh real ',relighting_net.sh[:,index_i,...],original_sh[:,index_i,...],relighting_net.sh.grad[:,index_i,...])
                    # print('noise', noise_func.delta_x)
                    # exit(0)
                    # noise_func.update()
                    if self.total_mask==True:
                        # eta=torch.clamp(noise_func.delta_x,min=-self.epsilon,max=self.epsilon).detach()
                        if not self.opt.dynamic_budget:
                            eta=torch.min(noise_func.delta_x,torch.ones((noise_func.delta_x.size())).to(self.device)*self.epsilon).detach()
                            eta=torch.max(eta,-torch.ones((noise_func.delta_x.size())).to(self.device)*self.epsilon).detach()
                            print('eta',torch.max(eta),torch.min(eta))
                            print('epsilon',self.epsilon)
                            # exit(0)
                        else:
                            print(noise_func.delta_x.size(),self.epsilon.size())
                            print('epsilon is ',self.epsilon)
                            eta=torch.min(noise_func.delta_x,self.epsilon).detach()
                            eta=torch.max(eta,-self.epsilon).detach()
                            
                        # noise_func.a.data=min(max(noise_func.a.data,-1),1)
                        # noise_func.b.data=min(max(noise_func.b.data,-1),1)
                        # noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                        # noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                        noise_func.update_func()
                        if self.opt.adjust_contrast:
                            contrast_func.update_func()
                    if self.total_mask==False:
                        eta = torch.clamp(noise_func.delta_x*eta_mask, min=-self.epsilon, max=self.epsilon)
                        eta = torch.div(eta,eta_mask)
                    # print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    # if torch.norm(eta,p=1)<0.0005:
                    #     print('eta is too small')
                    #     exit(0)
                    #     break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    # print('before',noise_func.delta_x.data)
                    if not self.opt.relighting and not self.opt.adjust_contrast:
                        noise_func.delta_x.data= (torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                    else:
                        noise_func.delta_x.data= (torch.clamp(noise_func(relighted_X,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(relighted_X,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                    # print('after',noise_func.delta_x.data)
                    # exit()
                    # noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                    # print('noise func delta', torch.sum(torch.isnan(noise_func.delta_x.data)))
                    # print('noise func delta',noise_func.delta_x.data,'sh',relighting_net.sh.data)
                    # exit(0)
                    if self.opt.left_one_out and not self.opt.adapt_iter:
                        if max(last_fr_loss).item()<0.001:
                            break
                    
                    if self.opt.adapt_iter and not self.opt.left_one_out:
                        print('index of iteration:',i ,max(loss_list).item()<0.09)
                        if max(loss_list).item()<0.09:
                            break


                    if self.opt.left_one_out and self.opt.adapt_iter :
                        if max(last_fr_loss).item()<0.001 and max(loss_list).item()<0.09:
                            break
                else: #soft constraint
                    # loss+=100000*torch.sum(torch.square(torch.log(self.epsilon*eta_mask-delta_x+1))+torch.square(torch.log(self.epsilon*eta_mask+delta_x+1)))
                    
                    # loss+=torch.sum(torch.exp((torch.square(noise_func.delta_x)-self.epsilon*self.epsilon)))+10000*self.TVLoss(noise_func.delta_x)
                    
                    # loss+= torch.sum(1000*torch.nn.LeakyReLU(negative_slope=0.001)((torch.sort(torch.square(torch.flatten(noise_func.delta_x)))[0][:-100]-self.epsilon*self.epsilon)))+10000*self.TVLoss(noise_func.delta_x)
                    
                    # loss=torch.sum(1000000*torch.nn.LeakyReLU(negative_slope=0.00001)((torch.square(delta_x)-self.epsilon*self.epsilon)))+100*self.TVLoss(delta_x)
                    
                    if True:
                        keypoints=torch.where((torch.square(noise_func.delta_x)-self.epsilon*self.epsilon)>0,torch.ones(noise_func.delta_x.size()).to(noise_func.delta_x.device),torch.zeros(noise_func.delta_x.size()).to(noise_func.delta_x.device))
                        print('num of keypoints',torch.sum(keypoints))
                        
                        if torch.sum(keypoints)>=1000:
                            topValues,topIndex=torch.sort(torch.square(torch.flatten(noise_func.delta_x)))
                            topIndex=topIndex[-800:]
                            remap_index=self.flatten2matrix(topIndex,noise_func.delta_x.size())
                            # print((remap_index))
                            keypoints_to_attack=torch.zeros(keypoints.size()).to(keypoints.device)
                            keypoints_to_attack[remap_index[0],remap_index[1],remap_index[2],remap_index[3]]=1

                            # noise_func.delta_x.data= (torch.clamp(noise_func(img_id,attack_mask)-noise_func.delta_x.data*attack_mask + keypoints_to_attack, min=-1, max=1)-(noise_func(img_id,attack_mask)-noise_func.delta_x.data*attack_mask) ).detach()
                            noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + keypoints_to_attack.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                            if self.total_mask:
                                # noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                                # noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                                noise_func.update_func()
                                X_with_noise=noise_func(img_id,attack_mask)-noise_func.delta_x.data+noise_func.delta_x*attack_mask
                            else:
                                X_with_noise=noise_func(img_id,attack_mask)-noise_func.delta_x.data+noise_func.delta_x
                                break

                    # print(torch.sum(torch.exp(100*(torch.square(delta_x)-self.epsilon*self.epsilon))))
                    # exit(0)
                    loss.backward()
                    # grad = X.grad
                    
                    # delta_x.grad=delta_x.grad/(max(torch.norm(delta_x.grad,p=2),1)/2)
                    grad=noise_func.delta_x.grad

                    attack_optimizer.step()
                    scheduler.step()
                    eta=noise_func.delta_x
                    
                    # eta = torch.clamp(delta_x*eta_mask, min=-self.epsilon, max=self.epsilon)
                    # eta=torch.div(eta,eta_mask)
                    print('loss: ',loss, 'torch.norm',torch.max(grad),' eta ',torch.mean(eta))
                    if loss>1e+9:
                        print('loss explods')
                        break
                    if torch.norm(eta,p=1)<0.0005:
                        print('eta is too small')
                        break
                    # X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()
                    if self.total_mask:
                        noise_func.a.data=torch.clamp(noise_func.a,min=-1,max=1).detach()
                        noise_func.b.data=torch.clamp(noise_func.b,min=-1,max=1).detach()
                    noise_func.delta_x.data=(torch.clamp(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask + eta.detach(), min=-1, max=1)-(noise_func(img_id,attack_mask).detach()-noise_func.delta_x.detach()*attack_mask) ).detach()
                if record:
                    output_list=torch.cat(output_list,dim=1).detach()
                    output_list=output_list.permute(1,2,0)
                    output_list=output_list.numpy()[:,:,::-1]*255
                    cv2.imwrite(os.path.join(self.opt.output_path,image1_path+' '+image2_path+' output_iter_'+str(i)+'.jpg'),output_list)
            # print('relighted sh',relighting_net.sh)
            # print('noise: ',noise_func.sigmoid(noise_func.a).data,noise_func.sigmoid(noise_func.b).data)
            # exit()
            try:
                print('noise',noise_func.a.data,noise_func.b.data)
            except:
                pass
            if isinstance(self.model,dict):
                for name in self.model.keys():
                    self.model[name].zero_grad()
            else:
                self.model.zero_grad()

            # exit()
            if not self.opt.relighting and not self.opt.adjust_contrast:
                X_with_noise=noise_func(img_id,attack_mask)-noise_func.delta_x.data+noise_func.delta_x*attack_mask# add the final noise to the origianl input
                print('final diff ',torch.max(noise_func(img_id,attack_mask)-X_with_noise))
                # X_with_noise=noise_func(img_id,attack_mask) # add the final noise to the origianl input
            else:
                relighted_X=relighting_net(img_id,original_sh=original_sh)
                final_relighted_X=relighted_X.clone().detach()
                X_with_noise=noise_func(relighted_X,attack_mask)-noise_func.delta_x.data+noise_func.delta_x*attack_mask# add the final noise to the origianl input
                # X_with_noise=noise_func(relighted_X,attack_mask) # add the final noise to the origianl input
                
                print('max noise:',torch.max(torch.abs(X_with_noise-relighted_X)))
            # exit()
            
            #############################################################################################################

            test_output=self.get_attacked_output_from_attacked_source(X_with_noise,img_att,self.model)
            ########################################
            # if isinstance(test_output,dict) and self.opt.self.testAttackType=='ensemble_loss':
            #     result_index=0
            #     for name,single_test_output in test_output.items():
            #         result_index+=self.internal_id_loss(img_id,single_test_output,default=False)
            #     result_index/=len(self.model)
            # else:
            #     result_index=self.internal_id_loss(img_id,test_output)
            ######################################
            result_index=None
        
        attack_source_image=X_with_noise
        
        # print('loss type',self.opt.lossType)
        if self.opt.lossType=='ensemble_wb_test':
            if self.opt.testType=='id':
                test_result=self.test_id_extractors(attack_source_image.detach(),img_id.detach())
                return None, None, None, None, test_result
                # return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),test_output.detach().cpu(), result_index.detach().cpu().item(),test_result
            elif self.opt.testType=='df':
                assert isinstance(test_output,dict)
                for i, single_test_output in test_output.items():
                    test_output[i]=test_output[i].detach().cpu()
                # print(noise_func.delta_x)
                # print(self.epsilon)
                # print(attack_mask)
                # exit()
                if self.opt.relighting:
                    if self.opt.reshape_to_min:
                        attack_source_image=transforms.Resize((1024,1024))(attack_source_image)
                        img_id=transforms.Resize((1024,1024))(img_id)
                        noise_func.delta_x.data=transforms.Resize((1024,1024))(noise_func.delta_x.data)
                        relighted_noise=transforms.Resize((1024,1024))(relighting_net(original_X.clone()).detach().cpu()-original_X.clone().detach().cpu())
                        relighted_output=relighted_noise+img_s_with_original_size.detach().cpu()

                        # cv2.imwrite('advDF/ensemble_test/attacked_noise.jpg',10*torch.abs(noise_func.delta_x.data)[0].detach().cpu().numpy().transpose(1,2,0)*255)
                        # exit()
                    else:
                        relighted_output=relighting_net(original_X.clone(),original_sh=original_sh).detach().cpu()
                        print('max final noise ',  torch.max(noise_func.delta_x))
                    # assert cv2.imwrite('advDF/ensemble_test/relighting_example.jpg',attack_source_image[0].detach().cpu().numpy().transpose(1,2,0)[...,::-1]*255)
                    # print('finisih saving')
                    # print('max noise',torch.max(torch.abs(attack_source_image.detach().cpu() - final_relighted_X.detach().cpu())))
                    # exit()
                    return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),test_output,None,\
                         noise_func.delta_x.detach().cpu(), final_relighted_X.detach().cpu()
                else:
                    if self.opt.reshape_to_min:
                        attack_source_image=transforms.Resize((1024,1024))(attack_source_image)
                        img_id=transforms.Resize((1024,1024))(img_id)
                        noise_func.delta_x.data=transforms.Resize((1024,1024))(noise_func.delta_x.data)
                    return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),test_output,None,\
                         noise_func.delta_x.detach().cpu(), img_id.detach().cpu()
        
        if isinstance(test_output,dict):
            for i, single_test_output in test_output.items():
                test_output[i]=test_output[i].detach().cpu()
            
            if self.opt.reshape_to_min:
                        attack_source_image=transforms.Resize((1024,1024))(attack_source_image)
                        img_id=transforms.Resize((1024,1024))(img_id)
                        noise_func.delta_x.data=transforms.Resize((1024,1024))(noise_func.delta_x.data)
            return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),test_output,None,\
                noise_func.delta_x.detach().cpu(), img_id.detach().cpu()
        
        
        
        return attack_source_image.detach().cpu(), attack_source_image.detach().cpu() - img_id.detach().cpu(),output.detach().cpu(),None 
    
    def test_id_extractors(self,attacked_input,objected_input): 
        attacked_input1=attacked_input.clone()
        objected_input1=objected_input.clone()

        attacked_input1= F.interpolate(attacked_input1, (112,112))
        objected_input1= F.interpolate(objected_input1, (112,112))

        test_result={}
        if self.face_recogntion_models!=None:
            for i,(model_name,fr_model) in enumerate(zip(self.opt.name_of_arcface_insight,self.face_recogntion_models)):
                attacked_feat=fr_model(attacked_input1)
                objected_feat=fr_model(objected_input1)
                objected_feat2=fr_model(objected_input1)
                assert attacked_input1.size()==objected_input1.size()
                test_result[str(i)+model_name]=self.cosloss_fn(attacked_feat,objected_feat).detach().cpu().numpy()[0]
                test_result[str(i)+model_name+'origin']=self.cosloss_fn(objected_feat2,objected_feat).detach().cpu().numpy()[0]
        
        attacked_input2=attacked_input.clone()
        objected_input2=objected_input.clone()
        origin_img_id_downsample = F.interpolate(objected_input2, (112,112))
        img_id_downsample = F.interpolate(attacked_input2, (112,112))
        origin_id=self.model['simswap'].model.netArc(origin_img_id_downsample)
        latend_id = self.model['simswap'].model.netArc(img_id_downsample)
        origin_id2 = self.model['simswap'].model.netArc(origin_img_id_downsample)
        latend_id = latend_id.to('cuda')
        # origin_id=origin_id/torch.norm(origin_id,p=2,dim=1,keepdim=True)
        test_result['default_arcface']=self.cosloss_fn(latend_id, origin_id ).detach().cpu().numpy()[0]
        test_result['default_arcface_original']=self.cosloss_fn(origin_id2, origin_id ).detach().cpu().numpy()[0]

        attacked_input3=attacked_input.clone()
        objected_input3=objected_input.clone()
        origin_img_id_downsample=preprocess_for_attack_arcface(objected_input3)
        img_id_downsample=preprocess_for_attack_arcface(attacked_input3)
        latend_id=self.top_model(img_id_downsample)
        origin_id=self.top_model(origin_img_id_downsample)
        origin_id2=self.top_model(origin_img_id_downsample)
        test_result['arcface_3p']=self.loss_fn(latend_id,origin_id).detach().cpu().numpy()[0]
        test_result['arcface_3p_original']=self.loss_fn(origin_id2,origin_id).detach().cpu().numpy()[0]
        return test_result
    
    
    
    def admin_perturb(self,img_s,img_d,y,transformer=None):
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        attack_optimizer=optim.SGD([{'params': [X], 'lr': self.a}])
        scheduler = StepLR(attack_optimizer, step_size=2333, gamma=0.1)
        # X.requires_grad=True
        with torch.enable_grad():
            for i in range(self.k):
                y=y.detach()
                X = X.cuda()
                X.requires_grad=True

                img_att = img_att.cuda()
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                latend_id = latend_id
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                # latend_id=0.1*torch.ones(latend_id.size()).to(self.device) # to test 0 id
                output = self.model(X, img_att, latend_id, latend_id, True)

            
                X.requires_grad = True

                
                self.model.netArc.zero_grad()
                self.top_model.zero_grad()
                self.model.zero_grad()
                # Minus in the loss means "towards" and plus means "away from"
                
                # loss_set=[]
                loss=0
                # loss+=50*self.output_pixel_loss(output,y)
                # loss+=200*self.output_id_loss(output,y,False)
                # loss+=200*self.internal_id_loss(X,y)
                # loss+=200*self.mapulate_id_loss(X)
                # # loss+=1000*self.discriminator_loss(output)
                # loss+=10*self.face_recognition_loss(output)
                loss+=100*self.mapulate_output_loss(output,latend_id=latend_id,img_att=img_att)
                loss=-loss
                loss.backward()
                grad = X.grad
                print('loss: ',loss, 'torch.norm',torch.max(grad))
                # X_adv = X + self.a * grad/torch.norm(grad)
                # X_adv = X + self.a * grad.sign()
                attack_optimizer.step()
                scheduler.step()
                # X_adv=X
                eta = torch.clamp(X - img_id, min=-self.epsilon, max=self.epsilon)
                print('eta',torch.mean(eta))
                if torch.norm(eta,p=1)<0.0005:
                    break
                X.data= torch.clamp(img_id + eta, min=-1, max=1).detach_()

            self.model.zero_grad()

        return X.detach().cpu(), X - img_id,output.detach().cpu()
    def perturb(self, img_s, img_d, y):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                # convert numpy to tensor
                X = X.cuda()
                X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                latend_id = latend_id
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                output = self.model(X, img_att, latend_id, latend_id, True)

            
                X.requires_grad = True

                
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                # Minus in the loss means "towards" and plus means "away from"
                
                loss = self.loss_fn(output, y)
                
                loss.backward()
                grad = X.grad
                print('loss: ',loss, 'torch.norm',torch.max(grad))
                # X_adv = X + self.a * grad/torch.norm(grad)
                X_adv = X + self.a * grad.sign()
                
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                print('eta',torch.mean(eta))
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()

            self.model.zero_grad()

        return X.detach().cpu(), X - img_id,output.detach().cpu()
    def third_party_internal_id_perturb(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        y=y.detach()
        y=preprocess_for_attack_arcface(y)
        origin_source=X.clone().detach()
        origin_source=preprocess_for_attack_arcface(origin_source)
        # y=transformer_Arcface(y.squeeze(0))
        # y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)
        # origin_id=self.model.netArc(y).detach_()
        
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad = True
                # X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                trans_X=preprocess_for_attack_arcface(X)
                print('x size',trans_X.size())
                latend_id=self.top_model(trans_X)
                # origin_id=self.top_model(y)
                origin_source_id=self.top_model(origin_source)
                # img_id_downsample = F.interpolate(X, scale_factor=0.5)
                # latend_id = self.model.netArc(img_id_downsample)
                # latend_id = latend_id
                # print('latent id',latend_id.size())
                # latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                # latend_id = latend_id.to('cuda')
                
            
                

                self.top_model.zero_grad()
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)
                # print(output.size())
                # output=transformer_Arcface(output.squeeze(0))
                # output=F.interpolate(output.unsqueeze(0),scale_factor=0.5)
                # output_id=self.model.netArc(output)
                
                # loss = self.loss_fn(latend_id, origin_id )
                loss=self.loss_fn(latend_id,origin_source_id)
                # average_latend_id=torch.mean(latend_id.clone().detach(),1).unsqueeze(1).repeat(1,latend_id.size()[1]).to(self.device)
                # zero_latend_id=torch.zeros(latend_id.size()).to(self.device)
                # print(zero_latend_id.size(),latend_id.size)
                # loss=self.loss_fn(latend_id,)
                # loss-=self.loss_fn(latend_id,zero_latend_id)
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                X_adv = X + self.a * grad.sign()
                # X_adv=X-self.a*grad/torch.norm(grad)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                # print('xgrad',X.grad)
            self.model.zero_grad()
        output = self.model(X, img_att, latend_id, latend_id, True)
        attack_output=output.clone()
        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
    
    def ensemble_attack(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        original_output=y.clone()
        y=y.detach()
        y=preprocess_for_attack_arcface(y)
        
        # y=transformer_Arcface(y.squeeze(0))
        # y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)
        # origin_id=self.model.netArc(y).detach_()
        
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad = True
                # X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                trans_X=preprocess_for_attack_arcface(X)
                print('x size',trans_X.size())

                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                internal_latend_id = self.model.netArc(img_id_downsample)
                internal_latend_id = internal_latend_id/torch.norm(internal_latend_id,p=2,dim=1,keepdim=True)
                latend_id=self.top_model(trans_X)
                origin_output_id=self.top_model(y)
                output = self.model(X, img_att, internal_latend_id, internal_latend_id, True)
                # print('outputsize',output.size(),X.size(),img_att.size(),internal_latend_id.size())
                # exit()
                processed_output=preprocess_for_attack_arcface(output.clone())
                output_id=self.top_model(processed_output)
                # img_id_downsample = F.interpolate(X, scale_factor=0.5)
                # latend_id = self.model.netArc(img_id_downsample)
                # latend_id = latend_id
                # print('latent id',latend_id.size())
                # latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                # latend_id = latend_id.to('cuda')
                
            
                

                self.top_model.zero_grad()
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)
                # print(output.size())
                # output=transformer_Arcface(output.squeeze(0))
                # output=F.interpolate(output.unsqueeze(0),scale_factor=0.5)
                # output_id=self.model.netArc(output)
                # latend_id=self.top_model(trans_X)
                # loss = 0.01*self.loss_fn(latend_id, origin_id )
                loss=200*self.loss_fn(output_id,origin_output_id)
                print('output id',output_id)
                # loss=0.2*torch.sum(1-self.cosloss_fn(output_id,origin_output_id))
                # average_latend_id=torch.mean(latend_id.clone().detach(),1).unsqueeze(1).repeat(1,latend_id.size()[1]).to(self.device)
                # zero_latend_id=torch.zeros(latend_id.size()).to(self.device)
                # print(zero_latend_id.size(),latend_id.size)
                # loss=self.loss_fn(latend_id,-latend_id.detach())
                loss+=0.5*self.loss_fn(output,original_output)
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                X_adv = X + self.a * grad.sign()
                # X_adv=X-self.a*grad/torch.norm(grad)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                print('xgrad',X.grad)
            self.model.zero_grad()
        output = self.model(X, img_att, internal_latend_id, internal_latend_id, True)
        attack_output=output.clone()
        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
    def third_party_id_perturb(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        y=y.detach()
        y=preprocess_for_attack_arcface(y)
        # y=transformer_Arcface(y.squeeze(0))
        # y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)

        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                if (i+1)%100==0:
                    self.a=self.a/2
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                # latend_id = latend_id
                print('latent id',latend_id.size())
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                output = self.model(X, img_att, latend_id, latend_id, True)
                attack_output=output.clone()

                output=preprocess_for_attack_arcface(output)
                X.requires_grad = True

                self.top_model.zero_grad()
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)
                
                # preprocess_for_attack_arcface(output)
               
                print('y ',y.size())
                output_id=self.top_model(output)
                origin_id=self.top_model(y)
                loss = self.loss_fn(output_id, origin_id )
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                X_adv = X + self.a * grad.sign()
                # X_adv=X+self.a*grad/torch.norm(grad,p=2)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                print('xgrad',X.grad)
            self.model.zero_grad()

        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
    def discriminator_attack(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        y=y.detach()
        y=preprocess_for_attack_arcface(y)
        # y=transformer_Arcface(y.squeeze(0))
        # y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)

        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                if (i+1)%100==0:
                    self.a=self.a/2
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                # latend_id = latend_id
                print('latent id',latend_id.size())
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                output = self.model(X, img_att, latend_id, latend_id, True)
                attack_output=output.clone()

                X.requires_grad = True

                self.top_model.zero_grad()
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)

               
                print('y ',y.size())

                loss=self.model.get_GANLoss(output)
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                X_adv = X + self.a * grad.sign()
                # X_adv=X+self.a*grad/torch.norm(grad,p=2)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                print('xgrad',X.grad)
            self.model.zero_grad()   
        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
    def id_perturb(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        y=y.detach()
        y=transformer_Arcface(y.squeeze(0))
        y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                if (i+1)%100==0:
                    self.a=self.a/2
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                latend_id = latend_id
                print('latent id',latend_id.size())
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                output = self.model(X, img_att, latend_id, latend_id, True)
                attack_output=output.clone()
            
                # X.requires_grad = True

                
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)
                print(output.size())
                # if transformer!=None:
                output=transformer_Arcface(output.squeeze(0))
                
                output=F.interpolate(output.unsqueeze(0),scale_factor=0.5)
                
                print(output.size())
                print('y ',y.size())
                output_id=self.model.netArc(output)
                origin_id=self.model.netArc(y)
                loss = self.loss_fn(output_id, origin_id )
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                # X_adv = X + self.a * grad.sign()
                X_adv=X+self.a*grad/torch.norm(grad,p=2)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                print('xgrad',X.grad)
            self.model.zero_grad()

        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
    
    def perturb_internal_id(self, img_s, img_d, y,transformer=None):
        """
        Vanilla Attack.
        """
        
        
            
        img_id = img_s.view(-1, img_s.shape[0], img_s.shape[1], img_s.shape[2]).cuda()

        # img_s = transformer_Arcface(img_s)
        if self.rand:
            X = img_id.clone().detach_().to(self.device) + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, img_id.shape).astype('float32')).to(self.device)
        else:
            X = img_id.clone().detach_().to(self.device)
            # use the following if FGSM or I-FGSM and random seeds are fixed
            # X = img_s.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, img_s.shape).astype('float32')).cuda()  

        # img_d = transformer(img_d)
        y=y.detach()
        y=transformer_Arcface(y.squeeze(0))
        y=F.interpolate(y.unsqueeze(0),scale_factor=0.5)
        origin_id=self.model.netArc(y).detach_()
        
        img_att = img_d.view(-1, img_d.shape[0], img_d.shape[1], img_d.shape[2])
        with torch.enable_grad():
            for i in range(self.k):
                # convert numpy to tensor
                y=y.detach()
                X = X.cuda()
                X.requires_grad=True
                img_att = img_att.cuda()

                #create latent id
                img_id_downsample = F.interpolate(X, scale_factor=0.5)
                latend_id = self.model.netArc(img_id_downsample)
                latend_id = latend_id
                print('latent id',latend_id.size())
                latend_id = latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)
                latend_id = latend_id.to('cuda')
                
            
                X.requires_grad = True

                
                self.model.netArc.zero_grad()
                self.model.zero_grad()
                
                # Minus in the loss means "towards" and plus means "away from"
                # print('\n'*3)
                # print(output.size())
                # output=transformer_Arcface(output.squeeze(0))
                # output=F.interpolate(output.unsqueeze(0),scale_factor=0.5)
                # output_id=self.model.netArc(output)
                loss = self.loss_fn(latend_id, origin_id )
                print('loss: ',loss)
                loss.backward()
                grad = X.grad
                # print('grad', grad)
                # X_adv = X + self.a * grad.sign()
                X_adv=X+self.a*grad/torch.norm(grad)
                eta = torch.clamp(X_adv - img_id, min=-self.epsilon, max=self.epsilon)
                X = torch.clamp(img_id + eta, min=-1, max=1).detach_()
                print('xgrad',X.grad)
            self.model.zero_grad()
        output = self.model(X, img_att, latend_id, latend_id, True)
        attack_output=output.clone()
        return X.detach().cpu(), X - img_id,attack_output.detach().cpu()
   