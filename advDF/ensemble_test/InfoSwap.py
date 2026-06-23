from genericpath import exists
import os
import cv2
import time
import argparse
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime, timedelta

from advDF.InfoSwap_master.utils import laplacian_blending
from advDF.InfoSwap_master.modules.encoder128 import Backbone128
from advDF.InfoSwap_master.modules.iib import IIB
from advDF.InfoSwap_master.modules.aii_generator import AII512
from advDF.InfoSwap_master.modules.decoder512 import UnetDecoder512
from advDF.InfoSwap_master.preprocess.mtcnn import MTCNN
# mtcnn = MTCNN()
# TRANSFORMS = transforms.Compose([
#             transforms.Resize((512, 512), interpolation=2),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
        
class InfoSwap(torch.nn.Module):
    def __init__(self,opt):
        super(InfoSwap,self).__init__()
        ROOT = {
            'smooth': {'root': './advDF/InfoSwap_master/modules/checkpoints_512/w_kernel_smooth', 'path': 'ckpt_ks_*.pth'},
            'no_smooth': {'root': './advDF/InfoSwap_master/modules/checkpoints_512/wo_kernel_smooth', 'path': 'ckpt_*.pth'}
        }


        """ Prepare Models: """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root = ROOT[opt.ib_mode]['root']
        path = ROOT[opt.ib_mode]['path']

        pathG = path.replace('*', 'G')
        pathE = path.replace('*', 'E')
        pathI = path.replace('*', 'I')

        self.encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
        state_dict = torch.load('./advDF/InfoSwap_master/modules/model_128_ir_se50.pth', map_location=device)
        self.encoder.load_state_dict(state_dict, strict=True)

        self.G = AII512().eval().to(device)
        self.decoder = UnetDecoder512().eval().to(device)

        # Define Information Bottlenecks:
        N = 10
        _ = self.encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
        _readout_feats = self.encoder.features[:(N + 1)]  # one layer deeper than the z_attrs needed
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]

        self.iib = IIB(in_c, out_c_list, device, smooth=args.ib_mode=='smooth', kernel_size=1)
        self.iib = self.iib.eval()

        self.G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=device), strict=True)
        print("Successfully load G!")
        self.decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=device), strict=True)
        print("Successfully load Decoder!")
        # 3) load IIB:
        self.iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=device),
                            strict=args.ib_mode=='smooth')
        print("Successfully load IIB!")

        self.TRANSFORMS = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        """ load pre-calculated mean and std: """
        self.param_dict = []
        for i in range(N + 1):
            state = torch.load(f'./advDF/InfoSwap_master/modules/weights128/readout_layer{i}.pth', map_location=device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            self.param_dict.append([state['m'].to(device), std, active_neurons])
    def forward(self,src,tgt_nat):
        assert torch.is_tensor(src) and torch.is_tensor(tgt_nat)
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3
        assert torch.is_tensor(src) and src.dim()==4 and src.size()[1]==3


        tgt_nat=self.TRANSFORMS(tgt_nat)
    
        src=self.TRANSFORMS(src)
        
        B=1
        
        X_id = self.encoder(
            F.interpolate(torch.cat((src, tgt_nat), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                            mode='bilinear', align_corners=True),
            cache_feats=True
        )
        # 01 Get Inter-features After One Feed-Forward:
        # batch size is 2 * B, [:B] for Xs and [B:] for Xt
        min_std = torch.tensor(0.01, device=device)
        readout_feats = [(self.encoder.features[i] - self.param_dict[i][0]) / torch.max(self.param_dict[i][1], min_std)
                            for i in range(N + 1)]

        # 02 information restriction:
        X_id_restrict = torch.zeros_like(X_id).to(device)  # [2*B, 512]
        Xt_feats, X_lambda = [], []
        Xt_lambda = []
        Rs_params, Rt_params = [], []
        for i in range(N):
            R = self.encoder.features[i]  # [2*B, Cr, Hr, Wr]
            Z, lambda_, _ = getattr(iib, f'iba_{i}')(
                R, readout_feats,
                m_r=self.param_dict[i][0], std_r=self.param_dict[i][1],
                active_neurons=self.param_dict[i][2],
            )
            X_id_restrict += self.encoder.restrict_forward(Z, i)

            Rs, Rt = R[:B], R[B:]
            lambda_s, lambda_t = lambda_[:B], lambda_[B:]

            m_s = torch.mean(Rs, dim=0)  # [C, H, W]
            std_s = torch.mean(Rs, dim=0)
            Rs_params.append([m_s, std_s])

            eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
            feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

            Xt_feats.append(feat_t)  # only related with lambda
            Xt_lambda.append(lambda_t)

        X_id_restrict /= float(N)
        Xs_id = X_id_restrict[:B]
        Xt_feats[0] = tgt_nat
        Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

        Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
        self.encoder.features = []

        # log identity similarities:
        Y_id_gt = self.encoder(
            F.interpolate(Y[:, :, 37:475, 37:475], size=[128, 128], mode='bilinear', align_corners=True),
            cache_feats=False
        )
        Xs_id_gt, Xt_id_gt = X_id[:B], X_id[B:]
        # msg = ''
        # msg += "cos<Xs, Xt>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Xt_id_gt,
        #                                                         dim=1).mean().detach().cpu().numpy()
        # msg += "cos<Y, Xt>=%.3f | " % torch.cosine_similarity(Xt_id_gt, Y_id_gt,
        #                                                         dim=1).mean().detach().cpu().numpy()
        # msg += "cos<Y, Xs>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Y_id_gt,
        #                                                         dim=1).mean().detach().cpu().numpy()
        # logger.info(msg)

        # '''(3) save Y: '''
        # img_Y = (Y[0].cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5) * 255
        # img_Y = img_Y.astype(np.uint8)

        return Y
    def depreprocess(self,img):
        if img.dim()==3: img=img.unsqueeze(0)
        return img
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    ROOT = {
        'smooth': {'root': './advDF/InfoSwap_master/modules/checkpoints_512/w_kernel_smooth', 'path': 'ckpt_ks_*.pth'},
        'no_smooth': {'root': './advDF/InfoSwap_master/modules/checkpoints_512/wo_kernel_smooth', 'path': 'ckpt_*.pth'}
    }

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('-ib', '--ib_mode', type=str, choices=list(ROOT.keys()))
    p.add_argument('-src', '--src_path', type=str)
    p.add_argument('-tar', '--tar_dir', type=str)
    args = p.parse_args()

    """ Prepare Models: """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT[args.ib_mode]['root']
    path = ROOT[args.ib_mode]['path']

    pathG = path.replace('*', 'G')
    pathE = path.replace('*', 'E')
    pathI = path.replace('*', 'I')

    encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    state_dict = torch.load('./advDF/InfoSwap_master/modules/model_128_ir_se50.pth', map_location=device)
    encoder.load_state_dict(state_dict, strict=True)

    G = AII512().eval().to(device)
    decoder = UnetDecoder512().eval().to(device)

    # Define Information Bottlenecks:
    N = 10
    _ = encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
    _readout_feats = encoder.features[:(N + 1)]  # one layer deeper than the z_attrs needed
    in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
    out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]

    iib = IIB(in_c, out_c_list, device, smooth=args.ib_mode=='smooth', kernel_size=1)
    iib = iib.eval()

    G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=device), strict=True)
    print("Successfully load G!")
    decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=device), strict=True)
    print("Successfully load Decoder!")
    # 3) load IIB:
    iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=device),
                        strict=args.ib_mode=='smooth')
    print("Successfully load IIB!")

    # with torch.no_grad():
    #     inference(args.src_path, args.tar_dir)

    Xs=Image.fromarray( cv2.imread(args.src_path))
    xt = cv2.imread(args.tar_dir)
    infoswap_model=InfoSwap(args)

    Xs=transforms.ToTensor()(Xs).unsqueeze(0).cuda()
    xt=transforms.ToTensor()(xt).unsqueeze(0).cuda()
    print(Xs.size())
    Y=infoswap_model(Xs,xt)
    img_Y = (Y[0].detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5) * 255
    img_Y = img_Y.astype(np.uint8)

    assert cv2.imwrite('advDF/ensemble_test/imgy.jpg',img_Y)
