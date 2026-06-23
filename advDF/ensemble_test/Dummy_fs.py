from matplotlib.pyplot import cla
import torch.nn as nn
import torchvision.transforms as transforms
class dummy_fs(nn.Module):
    def __init__(self) -> None:
        super(dummy_fs,self).__init__()
    def forward(self,x,y):
        assert x.dim()==4
        return x
    def depreprocess(self,x,default_size=1024):
        return x
        return transforms.Resize((default_size,default_size))(x)