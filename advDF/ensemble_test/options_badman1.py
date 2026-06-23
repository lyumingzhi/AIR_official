import argparse

# from  advDF.ensemble_test.options_simswap import options_simswap
class BaseOptions():
    def __init__(self) -> None:
        self.parser=argparse.ArgumentParser()
        self.initialized=False
    def initialize(self):
        self.parser = argparse.ArgumentParser()
        # self.parser.add_argument("--swap_type", type=str, default="ftm")
        # self.parser.add_argument("--img_root", type=str, default="/data/yuhao.zhu/CelebA-HQ")
        # self.parser.add_argument("--mask_root", type=str, default="/data/yuhao.zhu/CelebAMaskHQ-mask")
        # self.parser.add_argument("--srcID", type=int, default=2332)
        # self.parser.add_argument("--tgtID", type=int, default=2107)
        # self.parser.add_argument('--src_path',type=str,default='')
        # self.parser.add_argument('--tgt_path',type=str,default='')
        # self.parser.add_argument('--src_name',type=str,default='')
        # self.parser.add_argument('--tgt_name',type=str,default='')

        self.parser.add_argument("--pic_a_path", type=str, default='./crop_224/gdg.jpg', help="Person who provides identity information")
        self.parser.add_argument("--pic_b_path", type=str, default='./crop_224/zrf.jpg', help="Person who provides information other than their identity")
        self.parser.add_argument('--dir',type=str,default='./crop_224',help='diretory of the input files')
        self.parser.add_argument('--lossType',type=str,default='',help='type of loss to attack')
        self.parser.add_argument('--checkpoint_of_arcface_insight',action='append',help='weight for arcface model from insight',default=None)
        self.parser.add_argument('--name_of_arcface_insight',action='append',help='name of arcface model from insight',default=None)
        self.parser.add_argument('--total_mask',action='store_true',help='whether totally mask the face region')
        self.parser.add_argument('--facial_cls_type',action='append',help='facial classification type',default=None)
        self.parser.add_argument('--hard_constraint',action='store_true',help='whether to use infinit norm loss')
        self.parser.add_argument('--testType',type=str,default='',help='type of test')
        self.parser.add_argument('--output_path',type=str,default='outputs/',help='path to output the result')

        self.parser.add_argument('--inference_model',action='append',help='the name of models for inference')
        self.parser.add_argument('--testAttackType',type=str,default='fr',help='the attack type for ensemble white box attack')
        self.parser.add_argument('--dynamic_budget',action='store_true',help='whether to use dynamic attack budget')
        self.parser.add_argument('--dry_run', action='store_true', help='validate inputs and configuration without loading models')
        self.parser.add_argument('--max_pairs', type=int, default=None, help='optional maximum number of image pairs to process after pair range slicing')
        self.parser.add_argument('--pair_start', type=int, default=0, help='zero-based inclusive start index in input_pair_index.xlsx')
        self.parser.add_argument('--pair_end', type=int, default=None, help='zero-based exclusive end index in input_pair_index.xlsx')
        self.parser.add_argument('--save_every', type=int, default=50, help='number of processed pairs per PNG result sheet')
        self.parser.add_argument('--fail_on_missing_pairs', action='store_true', help='fail if any selected pair references missing input images')

        self.parser.add_argument('--ib_mode',type=str,default='smooth',help='the mode for models (Infoswap)')
        self.parser.add_argument('--sign',action='store_true',help='whether use the sign of gradient to attack')
        self.parser.add_argument('--relighting',action='store_true',help='whether to preprocess with relighting')
        self.parser.add_argument('--gradient_encourage', action='store_true',help='whether to encorage to change ')
        self.parser.add_argument('--adjust_contrast',action='store_true',help='whether to adjust contrast to attack')
        self.parser.add_argument('--source_model',action='append',help='which models to generate the noise')
        self.parser.add_argument('--relighting_encourage',action='store_true',help='let sh follow natural distribution')

        self.parser.add_argument('--preprocess_type',type=str,default='',help='type of preprocessing defense')
        self.parser.add_argument('--reshape_to_min',action='store_true',default='',help='reshape the attacked image to min')

        self.parser.add_argument('--universal_attack',action='store_true',help='whether use universal_attack')
        #setting for simswap
        self.initialize_simswap(self.parser)

        self.initialized=True
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

    def initialize_simswap(self,opt):
        
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
        opt.add_argument('--isTrain', type=bool, default=False, help='local rank for distributed training')
        opt.add_argument('--checkpoints_dir', type=str, default='./advDF/SimSwap/checkpoints', help='models are saved here')
        opt.add_argument('--name', type=str, default='people', help='name of the experiment. It decides where to store samples and models')
        opt.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        opt.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        opt.add_argument("--Arc_path", type=str, default='./advDF/SimSwap/arcface_model/arcface_checkpoint2.tar', help="run ONNX model via TRT")
        opt.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        opt.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

