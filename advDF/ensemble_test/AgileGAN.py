import torch
from torchvision import utils
from advDF.AgileGAN.lib.encoder.Encoder import EncoderModel
from argparse import Namespace
import cv2
import torchvision.transforms as transforms
from advDF.AgileGAN.lib.normal_image import  Normal_Image
from advDF.AgileGAN.lib.generator.model_dual import DualGenerator
from advDF.AgileGAN.lib.genderdetect import GenderDetection
class AgileGAN(torch.nn.Module):
    def __init__(self):
        super(AgileGAN,self).__init__()
        self.load_pretrain()
        ckpt = torch.load('./advDF/AgileGAN/pretrain/encoder.pt', map_location='cuda')
        opts = ckpt['opts']
        opts['checkpoint_path'] = './advDF/AgileGAN/pretrain/encoder.pt'
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        opts = Namespace(**opts)
        pspnet = EncoderModel(opts)
        pspnet.eval()
        pspnet.to('cuda')
        self.pspnet = pspnet

        # self.normal = Normal_Image()

        self.transforms = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(256),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.transforms_1024 = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.detransforms=transforms.Normalize([-1,-1,-1],[2,2,2])
        # # self.gender_detect = GenderDetection()
        # print(pspnet)
        print('agliegan init done')
        # exit(0)
    def load_pretrain(self):
        # styles=['pretrain/cartoon.pt','pretrain/comic.pt', 'pretrain/jackie.pt', 'pretrain/scarlett.pt']
        styles=['./advDF/AgileGAN/pretrain/jackie.pt']
        out=[]
        for s in styles:
            g = DualGenerator(1024, 512, 8)
            checkpoint = torch.load(s)
            g.load_state_dict(checkpoint["g_ema"], strict=False)
            g.eval()
            g.to('cuda')
            out.append(g)
        self.g_list = out
    def forward(self,img,img2=None):
        torch.manual_seed(0)
        # is_female = self.gender_detect.detect(img)
        # img = self.normal.run(img)
        # img = img.convert("RGB")
        
        # img=img.transpose(1,2,0).copy()/255
        assert torch.is_tensor(img) and img.dim()==4 and img.size()[1]==3
        transformed_image=self.transforms(img.clone())
        # exit()
        # print(transformed_image)
 
        # transformed_image_1024 = self.transforms_1024(img).unsqueeze(0).to("cuda").float()
        transformed_image_1024 = self.transforms_1024(img).to("cuda").float()
        output=[transformed_image_1024]
        # sampled, _, mu = self.pspnet.encoder(transformed_image.unsqueeze(0).to("cuda").float())
        sampled, _, mu = self.pspnet.encoder(transformed_image.to("cuda").float())

        latent = [self.pspnet.decoder.style(s) for s in mu]
        latent = [torch.stack(latent, dim=0)]

        for g in self.g_list:
            fake_img_A, fake_img_B, _ = g(latent, input_is_latent=True, noise=None)
            output.append(fake_img_B)
            
        return output[1]
    def depreprocess(self,img):
        if img.dim()==3: img=img.unsqueeze(0)
        return self.detransforms(img)