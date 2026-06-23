import torchvision.transforms as transforms
import torch
import cv2
from torch import nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image

class Gender_classifier(nn.Module):
    def __init__(self,input_dim,output_dim,opt=None) -> None:
        super(Gender_classifier,self).__init__()
        hidden_dim1=512
        hidden_dim2=1024
        hidden_dim3=512

        

        self.linear_layer1=nn.Linear(input_dim,hidden_dim1)
        self.act1=nn.ReLU()
        self.linear_layer2=nn.Linear(hidden_dim1,hidden_dim2)
        self.act2=nn.ReLU()
        self.linear_layer3=nn.Linear(hidden_dim2+input_dim,hidden_dim3)
        self.act3=nn.ReLU()
        self.linear_layer4=nn.Linear(hidden_dim3,output_dim)

    
    def forward(self,x):
        
        residual=x
        mid_output=self.linear_layer1(x)
        mid_output=self.act1(mid_output)
        mid_output=self.linear_layer2(mid_output)
        mid_output=self.act2(mid_output)
        mid_output=torch.cat((mid_output,residual),1)
        mid_output=self.linear_layer3(mid_output)
        mid_output=self.act3(mid_output)
        output=self.linear_layer4(mid_output)
        return output, F.softmax(output,dim=1)


# def resnet18(num_classes):
#     """Constructs a ResNet-18 model."""
#     model = ResNet(block=BasicBlock, 
#                    layers=[2, 2, 2, 2],
#                    num_classes=NUM_CLASSES,
#                    grayscale=GRAYSCALE)
#     return model