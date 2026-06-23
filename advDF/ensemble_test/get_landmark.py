from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
import sys
from os import path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# print(path.dirname(path.dirname(path.abspath(__file__))))
from advDF.ensemble_test.pytorch_face_landmark.common.utils import BBox,drawLandmark,drawLandmark_multiple
from advDF.ensemble_test.pytorch_face_landmark.models.basenet import MobileNet_GDConv
from advDF.ensemble_test.pytorch_face_landmark.models.pfld_compressed import PFLDInference
from advDF.ensemble_test.pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
# from pytorch_face_landmark.FaceBoxes import FaceBoxes
# from pytorch_face_landmark.Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
# from pytorch_face_landmark.MTCNN import detect_faces
import glob
import time
from advDF.ensemble_test.pytorch_face_landmark.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from torch import double, dtype, nn
from torchvision import transforms
class LankMark(nn.Module):
    def __init__(self,backbone='MobileFaceNet') -> None:
        super(LankMark,self).__init__()
        self.backbone=backbone
        self.model=self.load_model()
        self.MobileNet_preprocess=transforms.Compose([transforms.Normalize([ 0.485, 0.456, 0.406 ],[ 0.229, 0.224, 0.225 ])])
    def load_model(self):
        if self.backbone=='MobileNet':
            model = MobileNet_GDConv(136)
            model = torch.nn.DataParallel(model)
            # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
            checkpoint = torch.load('advDF/ensemble_test/pytorch_face_landmark/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar')
            print('Use MobileNet as backbone')
        elif self.backbone=='PFLD':
            model = PFLDInference() 
            # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
            checkpoint = torch.load('advDF/ensemble_test/pytorch_face_landmark/checkpoint/pfld_model_best.pth.tar')
            print('Use PFLD as backbone') 
            # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
        elif self.backbone=='MobileFaceNet':
            model = MobileFaceNet([112, 112],136)   
            checkpoint = torch.load('advDF/ensemble_test/pytorch_face_landmark/checkpoint/mobilefacenet_model_best.pth.tar')      
            print('Use MobileFaceNet as backbone')         
        else:
            print('Error: not suppored backbone')    
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def forward(self,x):
        if self.backbone=='MobileNet':
            x=transforms.Resize(224)(x)
            x = self.MobileNet_preprocess(x)
        else:
            x=transforms.Resize(112)(x)
        # print(x.size())
        if self.backbone=='MobileFaceNet':
            landmark = self.model(x)[0]
        else:
            landmark = self.model(x)
        landmark = landmark.view(-1,2)
        return landmark

if __name__=='__main__':
    lankMark_model=LankMark()
    lankMark_model.eval()
    input=torch.Tensor(cv2.imread('crop_224/1.jpg').transpose(2,0,1)/225).unsqueeze(0)
    # print(input.size())
    print(lankMark_model(input))