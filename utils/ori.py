import numpy as np
from models.common import Concat
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def orth_dist(mat, stride=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).to(device))


def deconv_orth_dist(kernel, stride=2, padding=1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).to(device)
    ct = int(np.floor(output.shape[-1] / 2))
    target[:, :, ct, ct] = torch.eye(o_c).to(device)
    return torch.norm(output - target)


def get_ori_loss(lam, cur_model):
    diff = 0
    import torch.nn as nn
    if isinstance(cur_model, nn.Sequential):
        for i in range(len(cur_model)):
            if isinstance(cur_model[i], nn.Sequential):
                diff += get_ori_loss(lam, cur_model[i])
            elif isinstance(cur_model[i], nn.Conv2d):
                if i != 0:
                    diff += deconv_orth_dist(cur_model[i].weight, stride=1)
    elif isinstance(cur_model, Concat):
        for module in cur_model._modules.values():
            if isinstance(module, nn.Sequential):
                diff += get_ori_loss(lam, module)
    return diff
