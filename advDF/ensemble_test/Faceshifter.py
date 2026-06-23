import sys
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
# from advDF.Faceshifter.face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import advDF.Faceshifter.face_modules.model as model
from advDF.Faceshifter.network.AEI_Net import *
from advDF.Faceshifter.face_modules.mtcnn import *
import torch.nn.functional as F
import cv2
import PIL.Image as Image
import numpy as np
# import advDF.Faceshifter.configparser
import argparse
class Faceshifter(torch.nn.Module):
    def __init__(self,device='cuda'):
        super(Faceshifter,self).__init__()
        self.device=device
        self.G = AEI_Net(c_id=512)
        self.G.eval()
        self.G.load_state_dict(torch.load('./advDF/Faceshifter/saved_models/G_latest.pth', map_location=torch.device('cpu')))
        self.G = self.G.cuda()

        self.arcface = model.Backbone(50, 0.6, 'ir_se').to(device)
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load('./advDF/Faceshifter/face_modules/model_ir_se50.pth', map_location=device), strict=False)

        self.test_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.de_test_transform=transforms.Compose([transforms.Normalize((-1,-1,-1),(2,2,2))])
    def forward(self,img_src,img_tgt):
        img_src=img_src.clone()
        img_tgt=img_tgt.clone()
        assert torch.is_tensor(img_src) and img_src.dim()==4 and img_src.size()[1]==3
        assert torch.is_tensor(img_tgt) and img_tgt.dim()==4 and img_tgt.size()[1]==3
        Xs=self.test_transform(img_src).float()
        Xs=F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True)
        embeds = self.arcface(Xs)
        Xt=self.test_transform(img_tgt).float()

        Yt, _ = self.G(Xt, embeds)
        return Yt
    def depreprocess(self,output):
        return self.de_test_transform(output)
    
    def get_inter_feats(self,img_src,img_tgt):
        img_src=img_src.clone()
        img_tgt=img_tgt.clone()
        assert torch.is_tensor(img_src) and img_src.dim()==4 and img_src.size()[1]==3
        assert torch.is_tensor(img_tgt) and img_tgt.dim()==4 and img_tgt.size()[1]==3
        Xs=self.test_transform(img_src).float()
        Xs=F.interpolate(Xs, (112, 112), mode='bilinear', align_corners=True)
        embeds = self.arcface(Xs)
        Xt=self.test_transform(img_tgt).float()

        Yt, inter_feats = self.G(Xt, embeds)
        return inter_feats