# from advDF import SimSwap
# from advDF.SimSwap.models.models import create_model
import advDF.SimSwap.models.models as Simswap_models

import sys
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
# sys.path.insert(0,'./advDF/Simswap/models')


class Simswap(torch.nn.Module):
    def __init__(self,opt) -> None:
        super(Simswap,self).__init__()
        self.model=self.get_pretrained_Simswap(opt).cuda()
        self.transformer_Arcface=transforms.Resize(224)
    def get_pretrained_Simswap(self,opt):
        torch.nn.Module.dump_patches = True
        Simswap_model = Simswap_models.create_model(opt)
        Simswap_model.eval()
        return Simswap_model
    def forward(self,img_s,img_t):
        
        assert torch.is_tensor(img_s) and torch.is_tensor(img_t)
        assert img_s.dim()==4 and img_t.dim()==4
        
        img_id=self.transformer_Arcface(img_s)
        img_att=self.transformer_Arcface(img_t)
        img_id_downsample=F.interpolate(img_id,scale_factor=0.5)
        latend_id=self.model.netArc(img_id_downsample)

        
        latend_id=latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)

        img_fake=self.model(img_id,img_att,latend_id,latend_id,True )
        # print('latent_id{:.20f}'.format(torch.sum(img_fake)))
        # exit()
        return img_fake
    def depreprocess(self,img):
        return img
    def get_inter_feats(self,img_s,img_t):
        assert torch.is_tensor(img_s) and torch.is_tensor(img_t)
        assert img_s.dim()==4 and img_t.dim()==4
        
        img_id=self.transformer_Arcface(img_s)
        img_att=self.transformer_Arcface(img_t)
        img_id_downsample=F.interpolate(img_id,scale_factor=0.5)
        latend_id=self.model.netArc(img_id_downsample)
        latend_id=latend_id/torch.norm(latend_id,p=2,dim=1,keepdim=True)

        img_fake, inter_feats=self.model(img_id,img_att,latend_id,latend_id,True, save_inter_feats=True )
        return inter_feats