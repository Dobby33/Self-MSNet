# dpc: data processing center
import glob
import os
import torch
from utils.common_utils import *
from configuration.params import *
import numpy as np
import cv2 as cv
from PIL import Image
from configuration.object_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def datasets(who):
    """ return the datasets path for "levin" or "lai", a list of all image name """
    return get_datasets_path(who)


def get_multi_channels_origin(input_depth, origin):
    """
    return origin image which have been resized,
    and with given the number of channels do some processing.
    """
    if input_depth == 1:  # 直接使用原始模糊图像初始化网络输入
        # TODO: 通过插值的方式扩大原始的输入图像
        return origin
    if input_depth > 1:
        # raise Exception("the number of input channels not be support for the moment.")
        # the number of channels is greater than 1, not be support for the moment.
        input_ = torch.Tensor().to(device)
        # print(origin.shape)
        # # origin as one channel, other channels using the noise
        # input_ = torch.cat([origin, get_noise(input_depth - 1, (origin.shape[2], origin.shape[3]))])

        basic_value = torch.zeros(origin.shape).type_as(origin.data)
        for _ in range(input_depth):
            # TODO: 8通道输入，添加噪声扰动
            input_t = origin + 0.01 * basic_value.normal_()
            input_ = torch.cat([input_, input_t], dim=1)
        return input_


def init_net_input(in_type, input_depth, size=(1, 1), origin=None):
    """
    initialize the input of networks that include net_x and net_k,
    the value of in_type that include "noise" and "origin".
    if the 'origin' is not 'None', then it is the raw image that have been resized and it's a tensor type.
    """
    input_ = None
    if in_type == "noise":
        input_ = get_noise(input_depth, spatial_size=size)
    if in_type == "origin" and origin is not None:
        # the function only used to adjust the number of input_ channels.
        input_ = get_multi_channels_origin(input_depth, origin)
    assert input_ is not None
    return input_


def get_ks(image_name, who, scale):
    """ get kernel size, it's a preset value """
    ks = get_kernel_size(image_name, who)
    ks = [int(ks[0] * scale), int(ks[1] * scale)]
    assert ks is not None
    return ks


def origin_img_up_sample(who, origin, img_add_pad_size):
    """
    if you want to resize the raw image to meet certain conditions of network inputs,
    you have to do some processing for this. The return value is tensor type. the origin is a RawImage object.
    """
    image = None
    if not who.colour:
        # 图像数据（采用双立方插值，增大图像尺寸）
        image = origin.pil_y.resize(img_add_pad_size, Image.BICUBIC)  # 采用双立方插值增大图像尺寸
        image = pil_to_np(image)
    else:
        image = cv.resize(origin.cv2_img, (img_add_pad_size[1], img_add_pad_size[0]), interpolation=cv.INTER_CUBIC)
        image = np.float32(image / 255.0)
        image = np.expand_dims(image, 0)
    resized_img = np_to_torch(image).to(device)
    assert image is not None
    return resized_img


def init_net_x_input(who, x_type, x_input_depth, img_add_pad_size, origin=None, need_up=True):
    """
    initialize the input of image generate network, Gx
    x_type: the initial type for net_x input, include "noise" and "origin"
    x_input_depth: input channels
    img_add_pad_size: the raw image size add the padw and padh.
    the origin is a RawImage Object.
    """
    net_input_x = None
    if x_type == "noise":  # 使用噪声作为输入
        net_input_x = init_net_input(x_type, x_input_depth, (img_add_pad_size[0], img_add_pad_size[1]), origin=None)
    if x_type == "origin":  # 使用原始的模糊图像作为输入
        assert origin is not None
        if need_up:
            resized_origin_img = origin_img_up_sample(who, origin, img_add_pad_size)
        else:
            resized_origin_img = origin
        # the function only used to adjust the number of resized_origin_img channels.
        net_input_x = init_net_input(x_type, x_input_depth,
                                     (img_add_pad_size[0], img_add_pad_size[1]), origin=resized_origin_img)
    if x_type == "shock":
        origin.processed_y = pil_to_np(origin.processed_y)
        origin.processed_y = np_to_torch(origin.processed_y)
        net_input_x = origin.processed_y
        for i in range(x_input_depth - 1):
            net_input_x = torch.cat([net_input_x, origin.processed_y], dim=1)

    assert net_input_x is not None
    return net_input_x


def get_raw_image(image_path_, who):
    """
    from the given image path read the image (gray or color)
    return the origin image object, contains some necessary attributions.
    """
    if not who.colour:
        """ 
        we use the PIL.Image to read the raw image,
        the y's ranges between 0 to 1 (because origin image y / 255.0)
        the type of 'y' is "np array"
        """
        pil_y, y = get_image(image_path_, -1)
        raw = RawImage(who, y, pil_y_=pil_y)  # pil_y for resize the input image
    else:
        _, y, cb, cr = readimg(image_path_)  # use cv2 read the image
        img_3channels_pil, _ = get_image(image_path_, -1)  # pil read img, (3, 680, 1024) C x H x W, [0..1]
        img_3channels_pil = img_3channels_pil.convert('RGB')
        img_3channels_np = pil_to_np(img_3channels_pil)
        img_3channels = np_to_torch(img_3channels_np).to(device)

        cv2_img = y  # "cv2_img" is needed to initialize net_x input, resize the origin image size.
        y = np.float32(y / 255.0)  # make y's ranges between 0 to 1
        y = np.expand_dims(y, 0)  # y shape is (1, 680, 1024) (channel, height, width)
        raw = RawImage(who, y, cb, cr, cv2_img, img_3channels=img_3channels)  # hwc -> chw
    return raw


def obj_generator(who, path, super_params=Parameters()):  # object generator
    """
    :param super_params
    :param path: datasets path
    :param who: levin or lai
    :return: a object generator
    """
    img_add_pad_size = [0, 0]
    sParams = super_params
    for image_path_ in path:

        processed_im_path = image_path_.split('/')[0] + "/" + image_path_.split('/')[1] + "/" + \
                            "bina_shock_filter_%s/" % who.name + image_path_.split('/')[2]
        # windows
        # processed_im_path = image_path_.split('/')[0] + "/" + image_path_.split('/')[1].split('\\')[0] + "/" + \
        #                     "bina_shock_filter_%s/" % who.name + image_path_.split('/')[1].split('\\')[1]
        processed_y = cv.imread(processed_im_path, 0)
        # ---- 2021-04-08 ----

        image_name = os.path.basename(image_path_)  # return the image name, xxx.png
        image_name = os.path.splitext(image_name)[0]

        ks = get_ks(image_name, who, sParams.scale)
        ks = who.manually_ks if len(ks) == 0 else ks  # for real image deblurring
        raw = get_raw_image(image_path_, who)  # raw image from given image path.
        y = raw.y
        raw.processed_y = processed_y  # 2021-04-08

        x_input_depth = sParams.x_input_depth
        x_input_type = sParams.init_x_type

        image_size = list(y.shape)
        origin_size = list(y.shape)
        image_data = np_to_torch(y).to(device)  # 已有的模糊图像，用作label
        img_3channels = raw.img_3channels

        if who.colour:
            image_data = [image_data, raw.cb, raw.cr]

        padh, padw = ks[0] - 1, ks[1] - 1  # H, W
        pad_size = [padh, padw]
        img_add_pad_size[0], img_add_pad_size[1] = int(image_size[1] * sParams.scales[0]) + pad_size[0], int(
            image_size[2] * sParams.scales[0]) + pad_size[1]
        # ----------------------------------------------------------------
        net_input_x = init_net_x_input(who, x_input_type, x_input_depth,
                                       (img_add_pad_size[0], img_add_pad_size[1]), raw).to(device)

        # initialization inputs
        input_x_basic = net_input_x.clone().detach()
        yield TrainingObject(image_name, image_data, image_size, ks, pad_size, input_x_basic, img_3channels,
                             origin_size)


def get(who, super_params):
    return obj_generator(who, datasets(who), super_params)


def get_dir_(step_, min_=0, max_=5000, directory="5000"):
    """
    given an integer determine which interval it is in
    e.g. the interval corresponding to 5000 is (0, 5000], return "5000"
    5005, (5000, 10000], so, return "10000"
    """
    while max_ <= 100000:
        if min_ < step_ <= max_:
            directory = str(max_)
            break
        min_ += 5000
        max_ += 5000
    return directory


def save_ycbcr(img_name, save_path, y, cb, cr):
    # save dip_uniform_ycbcr
    os.makedirs(save_path, exist_ok=True)
    cv.imwrite(save_path + img_name + ".png", y)
    cv.imwrite(save_path + img_name + "_cb.png", cb)
    cv.imwrite(save_path + img_name + "_cr.png", cr)
