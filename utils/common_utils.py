import math
import torch

import torchvision
import cv2
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.fftpack import fft2, ifft2, ifftshift, ifftn
import kornia
from skimage.measure import label, regionprops
import scipy.io
import torch.fft as fft



def exe_time(func):
    # 测试函数运行时间
    def new_func(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back

    return new_func


# kernel_size set (n,n) default
def gaussian_2d_kernel(kernel_size=3, sigma=0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val


# 高斯核生成函数
def create_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    gauss[gauss < 0] = 0
    return gauss / np.sum(gauss)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, img.size[1] - img.size[1] % d)
    bbox = [
        int((img.size[0] - new_size[0])/2),
        int((img.size[1] - new_size[1])/2),
        int((img.size[0] + new_size[0])/2),
        int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def load(path):
    """
    Load PIL image
    对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是"RGB"
    而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为"L"
    PIL中有九种不同模式，分别为1，L，P，RGB，RGBA，CMYK，YCbCr，I，F
    具体请参见：https://blog.csdn.net/icamera0/article/details/50843172
    """
    img = Image.open(path)
    return img


# 0.006s taken for {get_image}
def get_image(path, imsize=-1):
    """
    Load an image and resize to a specific size.
    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)  # <PIL.PngImagePlugin.PngImageFile image mode=L size=255x255 at 0x1E8D36C94A8>
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    # imsize是想要reshape后的尺寸信息，不等于-1说明想要reshape原始的输入
    if imsize[0] != -1 and img.size != imsize:
        print("and")
        if imsize[0] > img.size[0]:  # 如果reshape后的尺寸大于原来的
            img = img.resize(imsize, Image.BICUBIC)  # 采用双立方插值增大图像尺寸
        else:  # 如果想要缩小图像尺寸
            img = img.resize(imsize, Image.ANTIALIAS)  # 采用抗锯齿方法

    # Converts image in PIL format to np.array
    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type):
    """
    Fills tensor `x` with noise of type `noise_type`
    """
    # 相当于直接在原tensors上改变，无需返回
    if noise_type == 'u':
        x.uniform_()  # 从连续均匀分布中采样的数字，均匀分布 -1, 1
    elif noise_type == 'n':
        x.normal_()  # 正态分布
    else:
        assert False


# 0.000s taken for {get_noise}
def get_noise(input_depth=1, method="noise", spatial_size=(1, 1), noise_type='u', var=1. / 10):
    """
    Returns a pyTorch Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    meshgrid函数就是用两个坐标轴上的点在平面上画网格(传入的参数是两个的时候)
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 均匀分布 'n' for normal 正态分布
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        # 如果spatial_size是一个整数的话(没有给出维度)，就进行转换
        spatial_size = (spatial_size, spatial_size)

    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        # 制作张量
        net_input = torch.zeros(shape)
        # 填充噪声(均匀分布或正态分布)
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''
        Converts image in PIL format to np.array.
        From W x H x C [0...255] to C x W x H [0..1]
        当使用PIL.Image.open()打开图片后，如果要使用img.shape函数，需要先将image形式转换成array数组
        img = numpy.array(im)
    '''
    ar = np.array(img_PIL)  # ar.shape (255, 255)
    if len(ar.shape) == 3:
        # transpose在不指定参数是默认是矩阵转置
        '''
        详情请参见：
        https://blog.csdn.net/u012762410/article/details/78912667?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
        '''
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]  # ar.shape (1, 255, 255)

    # PIL图像对象，其type都是np.array，array里的元素类型(dtype)均为np.uint8
    # numpy.astype()方法改变元素类型(可以变成np.float32)
    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''
        Converts image in np.array format to PIL image.
        From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, object):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.1)  # learning rates
        for j in range(num_iter):
            # scheduler.step(j)
            optimizer.zero_grad()
            closure(j, object)
            optimizer.step()
    else:
        assert False


def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def readimg(path_to_image):
    # opencv imread(img_path, type)
    """type:
        cv2.IMREAD_COLOR: load the color image, it's a default parameter, you can direct to write it with 1
        cv2.IMREAD_GRAYSCALE: 0
        cv2.IMREAD_UNCHANGED: -1
    """
    img = cv2.imread(path_to_image)  # default color space is BGR
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # BGR 2 YCrCb
    y, cr, cb = cv2.split(x)

    return img, y, cb, cr



def PSNR(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.float32) / info.max


def remove_transparency(im, bg_colour=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im
