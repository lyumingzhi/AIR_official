# from base64 import encode
# from numpy.core.fromnumeric import transpose
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import numpy as np
import torch.nn.functional as F
# from advDF.Megagan.megafs import resnet50, HieRFE, Generator, FaceTransferModule
import  advDF.Megagan.megafs as megafs
class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask

class Megagan(nn.Module):
    def __init__(self,opt):
        super(Megagan,self).__init__()
        # self.encoder=encoder
        # self.swapper=swapper
        # self.generator=generator
        
        self.opt=opt
        self.size=1024
        self.swap_type='ftm'
        num_blocks=3 if self.swap_type=='ftm' else 1
        latent_split=[4,6,8]
        num_latents=18
        swap_indice=4
        self.encoder=megafs.HieRFE(megafs.resnet50(False),num_latents=latent_split,depth=50).cuda()
        self.swapper=megafs.FaceTransferModule(num_blocks=num_blocks,swap_indice=swap_indice,num_latents=num_latents,typ=self.swap_type).cuda()
        ckpt_e="./advDF/Megagan/checkpoint/{}_final.pth".format(self.swap_type)
        if ckpt_e is not None:
            print("load encoder & swapper:", ckpt_e)
            ckpts = torch.load(ckpt_e, map_location=torch.device("cpu"))
            self.encoder.load_state_dict(ckpts["e"])
            self.swapper.load_state_dict(ckpts["s"])
            del ckpts

        self.generator = megafs.Generator(self.size, 512, 8, channel_multiplier=2).cuda()
        ckpt_f = "./advDF/Megagan/checkpoint/stylegan2-ffhq-config-f.pth"
        if ckpt_f is not None:
            print("load generator:", ckpt_f)
            ckpts = torch.load(ckpt_f, map_location=torch.device("cpu"))
            self.generator.load_state_dict(ckpts["g_ema"], strict=False)
            del ckpts

        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        
        self.encoder.eval()
        self.swapper.eval()
        self.generator.eval()
    def forward(self,src,tgt_nat,origin_src=None,origin_tgt_nat=None,dummy=None):
        # import numpy as np
        # print('src {:.10f}'.format(torch.sum(src)))
        # print('tgt {:.10f}'.format(torch.sum(tgt_nat)))
        # np.save('advDF/ensemble_test/src2.npy',src.detach().cpu().numpy())
        # np.save('advDF/ensemble_test/tgt2.npy',tgt_nat.detach().cpu().numpy())
        # src=torch.from_numpy(np.load('advDF/ensemble_test/src1.npy')).to('cuda')
        # tgt_nat=torch.from_numpy(np.load('advDF/ensemble_test/tgt1.npy')).to('cuda')
        # exit()
        
        assert torch.is_tensor(src) and torch.is_tensor(tgt_nat)
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3
        tgt_nat=transforms.Compose([transforms.Resize((256,256))])(tgt_nat)
        if dummy!=None:
            src=transforms.Compose([transforms.Resize((256,256))])(src) + transforms.Compose([transforms.Resize((256,256))])(dummy)
        else:
            src=transforms.Compose([transforms.Resize((256,256))])(src)
        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt_nat = tF.normalize(tgt_nat, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)

        

        # print(torch.sum(origin_src.cuda()-src.cuda()))
        # print(torch.sum(origin_tgt_nat.cuda()-tgt_nat.cuda()))
        ts = torch.cat([tgt_nat, src], dim=0).cuda()
        
        # print('ts {:.10f}'.format(torch.sum(ts)))
        lats, struct = self.encoder(ts)
        # for param in self.encoder.parameters():
        #     print('param{:.10f}'.format(torch.sum(param.data)))
        
        # print(' lats {:.10f}'.format(torch.sum(lats)))
        # np.save('advDF/ensemble_test/result1.npy',lats.detach().cpu().numpy())
        # exit()
        idd_lats = lats[1:].clone()
        att_lats = lats[0].clone().unsqueeze_(0)
        att_struct = struct[0].clone().unsqueeze_(0)

        swapped_lats = self.swapper(idd_lats, att_lats)
        fake_swap, _ = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)

        # print('src {:.10f}'.format(torch.sum(src)))
        # print('swapped lats {:.10f}'.format(torch.sum(swapped_lats)))

        # print('fake swap',torch.sum(fake_swap))
        fake_swap_max = torch.max(fake_swap)
        fake_swap_min = torch.min(fake_swap)
        # denormed_fake_swap = (fake_swap[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0
        denormed_fake_swap = (fake_swap[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) 

        # print(denormed_fake_swap.size())
        # exit()
        # import cv2
        # cv2.imwrite('advDF/reverse_engineering/result.jpg',denormed_fake_swap.detach().cpu().numpy().transpose(1,2,0)*255)
        # exit()
        return denormed_fake_swap
    def depreprocess(self,img):
        if img.dim()==3: img=img.unsqueeze(0)
        return img

    def get_inter_feats(self,src,tgt_nat,dummy=None):
        assert torch.is_tensor(src) and torch.is_tensor(tgt_nat)
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3
        tgt_nat=transforms.Compose([transforms.Resize(256)])(tgt_nat)
        if dummy!=None:
            src=transforms.Compose([transforms.Resize(256)])(src) + transforms.Compose([transforms.Resize(256)])(dummy)
        else:
            src=transforms.Compose([transforms.Resize(256)])(src)
        src = tF.normalize(src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        tgt_nat = tF.normalize(tgt_nat, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), False)
        # print(torch.sum(origin_src.cuda()-src.cuda()))
        # print(torch.sum(origin_tgt_nat.cuda()-tgt_nat.cuda()))
        ts = torch.cat([tgt_nat, src], dim=0).cuda()

        lats, struct = self.encoder(ts)

        idd_lats = lats[1:].clone()
        att_lats = lats[0].clone().unsqueeze_(0)
        att_struct = struct[0].clone().unsqueeze_(0)

        swapped_lats = self.swapper(idd_lats, att_lats)
        fake_swap, inter_feats = self.generator(att_struct, [swapped_lats, None], randomize_noise=False,return_latents=True)
        return inter_feats
if __name__=='__main__':
    
    img1=torch.from_numpy(np.load('advDF/ensemble_test/src1.npy')).to('cuda')
    img2=torch.from_numpy(np.load('advDF/ensemble_test/tgt1.npy')).to('cuda')

    megagan=Megagan(None)
    with torch.no_grad():
        megagan(img1,img2)

    print('end')
    
