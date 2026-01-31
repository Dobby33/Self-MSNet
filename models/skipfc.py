import torch
import torch.nn as nn
from .common import *

def skipfc(num_input_channels=2, num_output_channels=3, 
           num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
           filter_size_down=3, filter_size_up=1, filter_skip_size=1,
           need_sigmoid=True, need_bias=True, 
           pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
           need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """

    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_channels_down[0],bias=True))
    model.add(nn.ReLU6())
    model.add(nn.Linear(num_channels_down[0], num_output_channels))
    model.add(nn.Softmax())
    return model











