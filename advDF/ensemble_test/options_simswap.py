import argparse
class options_simswap():
    def __init__(self) -> None:
        # self.parser=argparse.ArgumentParser()
        self.initialized=False
    def initialize(self,opt):
        
        opt.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        opt.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        opt.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        opt.add_argument('--latent_size', type=int, default=512, help='latent size of Adain layer')
        opt.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        opt.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG')
        opt.add_argument('--n_blocks_global', type=int, default=6, help='number of residual blocks in the global generator network')
        opt.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        opt.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        opt.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')   
        opt.add_argument('--isTrain', type=bool, default=True, help='local rank for distributed training')
        opt.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        opt.add_argument('--name', type=str, default='people', help='name of the experiment. It decides where to store samples and models')
        opt.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        opt.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
