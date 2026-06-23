from advDF.SimSwap.models.models import create_model
import sys
import torch
sys.path.insert(0,'./advDF/Simswap/models')
def get_Simswap(opt):
    torch.nn.Module.dump_patches = True
    Simswap_model = create_model(opt)
    Simswap_model.eval()
    return Simswap_model