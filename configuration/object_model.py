"""
this is a object model, store a series of the define of class
"""
import glob
import time
import os
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------
class RawImage:
    # origin image object, contains some attributions
    # gray and color
    def __init__(self, who_, y_, cb_=None, cr_=None, cv2_img_=None, pil_y_=None, processed_y_=None,
                 img_3channels=None):
        self.who = who_
        self.y = y_  # origin blurry image
        self.processed_y = processed_y_  # use the blurry image as the net input, need to process
        if who_.colour:
            self.cv2_img = cv2_img_
            self.cb = cb_
            self.cr = cr_
            self.img_3channels = img_3channels
        else:
            self.pil_y = pil_y_
            self.img_3channels = None


# ------------------------------------------------------------


# ------------------------------------------------------------
class TrainingObject:
    # blurring image object include the information show in below
    def __init__(self, image_name_, blur_image_, image_size_, kernel_size_, pad_size_, x_basic, img_3channels,
                 origin_size):
        self.image_name = image_name_
        self.blur_image = blur_image_
        self.image_size = image_size_
        self.kernel_size = kernel_size_
        self.pad_size = pad_size_
        self.x_input_basic = x_basic
        self.img_3channels = img_3channels
        self.origin_image_size = origin_size
        self.origin_input = x_basic

    def resize_obj_by_scale(self, who, params, origin=None):
        from idata.dpc import get_ks, init_net_x_input
        img_add_pad_size = [0, 0]
        self.image_size[1], self.image_size[2] = int(self.origin_image_size[1] * params.scale), int(
            self.origin_image_size[2] * params.scale)
        ks = get_ks(self.image_name, who, params.scale)
        self.kernel_size = ks
        self.pad_size[0], self.pad_size[1] = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        img_add_pad_size[0], img_add_pad_size[1] = self.image_size[1] + self.pad_size[0], self.image_size[2] + \
                                                   self.pad_size[1]
        # ----------------------------------------------------------------
        if origin is not None:
            net_input_x = init_net_x_input(who, params.init_x_type,
                                           params.x_input_depth,
                                           (img_add_pad_size[0], img_add_pad_size[1]), origin, need_up=False).to(device)
        else:
            net_input_x = F.interpolate(self.origin_input, [img_add_pad_size[0], img_add_pad_size[1]], mode="bilinear")

        self.x_input_basic = net_input_x.clone().detach()

    def get_three_input(self, scales, params, who):
        from idata.dpc import get_ks, init_net_x_input
        image_sizes, kernel_sizes, image_add_pad_sizes, pad_sizes = [], [], [], []
        net_inputs = []
        noises = []
        for i, scale in enumerate(scales):
            h, w = int(self.origin_image_size[1] * scale), int(self.origin_image_size[2] * scale)
            ks = get_ks(self.image_name, who, scale)
            pad_h, pad_w = ks[0] - 1, ks[1] - 1
            h_pad, w_pad = h + pad_h, w + pad_w

            image_sizes.append([h, w])
            kernel_sizes.append(ks)
            image_add_pad_sizes.append([h_pad, w_pad])
            pad_sizes.append([pad_h, pad_w])
            net_input_x = F.interpolate(self.origin_input, [h_pad, w_pad])
            noise = init_net_x_input(who, "noise", params.x_input_depth,
                             (h_pad, w_pad), need_up=False).to(device)
            noises.append(noise.clone().detach())
            net_inputs.append(net_input_x.clone().detach())

        return net_inputs, image_sizes, pad_sizes, kernel_sizes, noises
    # ------------------------------------------------------------


class Who:
    def __init__(self, HParams, manually_ks=None):
        """
        :param colour: True or False
        :param im_type: png or jpg
        :param data_path: datasets path
        """
        self.name = HParams.dataset_name
        self.colour = HParams.is_color
        self.im_type = HParams.im_type
        self.data_path = "datasets/%s" % HParams.dataset_name
        self.manually_ks = manually_ks  # manually define the kernel size


# ------------------------------------------------------------
class Parameters:

    def __init__(self, pad_="reflection", scale=1, scales=[1/8, 1/4, 1/2, 1], save_frequency_=100,
                 dataset="DatasetIsNone", im_type="png", is_color=True,
                 RESUME=False,
                 x_type="noise", x_input_depth_=8, x_output_depth_=1, flag="SoItIsIllegal",
                 lr_=0.01, lr_interval_list=None, lr_decay_gama=0.5,
                 p=90, beta=0.99, p_interval=00,
                 lambda_h=2e-5, gamma_x=2e-5, gamma_y=2e-5, rho=1.2, h_interval=100,
                 imdilate_nums_kernel=1, imdilate_nums_M=1, m_sign=1,
                 sparse_coeff_i=2.5, thres=0.01,
                 net_layer_nums=5, net_down_channels=128, net_up_channels=128, net_skip_channels=16,
                 num_iter_=2500, use_ssim_step=500, reg_noise_std_=0.001, tv_weight=0.0, sparse_coeff_is=[2,3,7,10],
                 model_name="Self_MSNet",
                 ):
        self.model_name = model_name
        self.sparse_coeff_is = sparse_coeff_is
        self.pad = pad_
        self.scale = scale
        self.scales = scales
        self.save_frequency = save_frequency_

        self.current_time = time.asctime(time.localtime(time.time()))

        self.dataset_name = dataset

        self.im_type = "jpg" if dataset == "real" else im_type
        self.is_color = False if dataset == "levin" else is_color

        self.RESUME = RESUME

        self.init_x_type = x_type
        self.x_input_depth = x_input_depth_
        self.x_output_depth = x_output_depth_
        self.flag = flag
        self.save_path = "results/%s_%s" % (dataset, flag)
        os.makedirs(self.save_path, exist_ok=True)

        self.lr = lr_
        self.lr_interval_list = [] if lr_interval_list is None else lr_interval_list
        self.lr_decay_gama = lr_decay_gama

        self.p = p
        self.beta = beta
        self.p_interval = p_interval  # p decay interval

        self.lambda_h = lambda_h
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.rho = rho
        self.h_interval = h_interval

        self.sparse_coeff_i = sparse_coeff_i
        self.thres = thres

        self.imdilate_nums_kernel = imdilate_nums_kernel
        self.imdilate_nums_M = imdilate_nums_M
        self.m_sign = m_sign

        self.net_layer_nums = net_layer_nums
        self.net_down_channels = net_down_channels
        self.net_up_channels = net_up_channels
        self.net_skip_channels = net_skip_channels

        self.num_iter = num_iter_
        self.use_ssim_step = use_ssim_step
        self.reg_noise_std = reg_noise_std_
        self.tv_weight = tv_weight

    def get_model_path(self):
        model_path = "results/model/%s/" % self.dataset_name
        all_models_path = glob.glob(os.path.join(model_path, '*.pth'))
        all_models_path.sort()
        assert all_models_path is not None
        return all_models_path

    def key_param(self):
        return "init_x_type: {param1:<30} # {explain1:<20} \n " \
               "x_channels: {param2:<30} # {explain2:<20}  \n " \
               "learning rate: {param3:<30} # {explain3:<20} \n " \
               "lr_interval_list: {param4:<30} # {explain4:<20} \n " \
               "lr_decay_gama: {param5:<30} # {explain5:<20} \n " \
               "p: {param6:<30} # {explain6:<20} \n " \
               "beta: {param7:<30} # {explain7:<20} \n " \
               "p_interval: {param8:<30} # {explain8:<20} \n " \
               "sparse_coeff_i: {param9:<30} # {explain9:<20} \n " \
               "rho: {param10:<30} # {explain10:<20} \n " \
               "h_interval: {param11:<30} # {explain11:<20} \n " \
               "sparse_coeff_i: {param12:<30} # {explain12:<20} \n" \
               "prune_iso_thres: {param13:<30} # {explain13:<20} \n " \
               "imdilate_nums_K: {param14:<30} # {explain14:<20} \n " \
               "imdilate_nums_M: {param15:<30} # {explain15:<20} \n " \
               "m_sign: {param16:<30} # {explain16:<20} \n " \
               "iteration_nums: {param17:<30} # {explain17:<20} \n " \
               "use_ssim_step: {param18:<30} # {explain18:<20} \n " \
               "reg_noise_std: {param19:<30} # {explain19:<20} \n" \
               "gamma_x: {param20:<30} # {explain20:<20} \n" \
               "gamma_y: {param21:<30} # {explain21:<20} \n" \
            .format(param1=self.init_x_type, explain1="用于设定网络输入的类型，包括随机向量noise和滤波后的图像origin",
                    param2=self.x_input_depth, explain2="网络初始化输入数据通道数，noise一般设置为8，origin设置为1",
                    param3=self.lr, explain3="学习率",
                    param4=str(self.lr_interval_list), explain4="学习率衰减区间，list类型",
                    param5=self.lr_decay_gama, explain5="学习率衰减系数",
                    param6=self.p, explain6="用于控制边缘的多少，值越大，边缘越少",
                    param7=self.beta, explain7="边缘筛选p值的衰减系数，随着迭代次数的增加，p值应该逐渐减小，使用更多的边缘估计模糊核",
                    param8=self.p_interval, explain8="p值的衰减区间",
                    param9=self.lambda_h, explain9="模糊核的正则化参数",
                    param10=self.rho, explain10="模糊核正则化参数lambda_h的衰减系数",
                    param11=self.h_interval, explain11="lambda_h的衰减区间",
                    param12=self.sparse_coeff_i, explain12="模糊核系数因子，值越大，模糊核越稠密",
                    param13=self.thres, explain13="联通区域阈值，小于该阈值的区域被置为0",
                    param14=self.imdilate_nums_kernel, explain14="模糊核的膨胀次数",
                    param15=self.imdilate_nums_M, explain15="M矩阵的膨胀次数",
                    param16=self.m_sign, explain16="多少次迭代时引入M矩阵",
                    param17=self.num_iter, explain17="迭代次数",
                    param18=self.use_ssim_step, explain18="在多少次迭代时使用SSIM损失",
                    param19=self.reg_noise_std, explain19="添加随机扰动的权重或系数",
                    param20=self.gamma_x, explain20="模糊核",
                    param21=self.gamma_y, explain21="模糊核")

    def __str__(self):
        return "current_time: {}        # 当前时间\n " \
               "dataset_name: {}        # 数据集名称，levin、lai、kohler、real\n" \
               "RESUME: {}              # 是否使用断点续传\n " \
               "init_x_type: {}         # 用于设定网络输入的类型，包括随机向量noise和滤波后的图像origin\n " \
               "x_channels: {}          # 网络初始化输入数据通道数，noise一般设置为8，origin设置为1 \n " \
               "x_output_channels: {}   # 网络初始化output通道数，1 or 3 \n " \
               "flag: {} # 本次实验结果保存的文件名\n" \
               "learning rate: {}       # 学习率\n " \
               "lr_interval_list: {}    # 学习率衰减区间，list类型\n " \
               "lr_decay_gama: {}       # 学习率衰减系数\n" \
               "p: {}                   # 用于控制边缘的多少，值越大，边缘越少\n " \
               "beta: {}                # 边缘筛选p值的衰减系数，随着迭代次数的增加，p值应该逐渐减小，使用更多的边缘估计模糊核\n " \
               "p_interval: {}          # p值的衰减区间\n " \
               "lambda_h: {}            # 模糊核的正则化参数\n " \
               "gamma_x: {}            # 模糊核\n " \
               "gamma_y: {}            # 模糊核\n " \
               "rho: {}                 # 模糊核正则化参数lambda_h的衰减系数\n " \
               "h_interval: {}          # lambda_h的衰减区间\n " \
               "sparse_coeff_i: {}      # 模糊核系数因子，值越大，模糊核越稠密\n " \
               "prune_iso_thres: {}     # 联通区域阈值，小于该阈值的区域被置为0\n " \
               "imdilate_nums_K: {}     # 模糊核的膨胀次数\n " \
               "imdilate_nums_M: {}     # M矩阵的膨胀次数\n " \
               "m_sign: {}              # 多少次迭代时引入M矩阵\n " \
               "net_layer_nums: {}      # Unet网络层数\n " \
               "net_down_channels: {}   # 降采样过程中输出的特征通道数\n " \
               "net_up_channels: {}     # 上采样过程中输出的特征通道数\n " \
               "net_skip_channels: {}   # 跨层连接输出的特征通道数\n" \
               "iteration_nums: {}      # 迭代次数\n " \
               "use_ssim_step: {}        # 在多少次迭代时使用MSE损失\n" \
               "reg_noise_std: {}       # 添加随机扰动的权重或系数\n " \
               "tv weight: {}           # tv weight\n" \
            .format(self.current_time,
                    self.dataset_name,
                    self.RESUME,
                    self.init_x_type, self.x_input_depth, self.x_output_depth, self.flag,
                    self.lr, self.lr_interval_list, self.lr_decay_gama,
                    self.p, self.beta, self.p_interval,
                    self.lambda_h, self.gamma_x, self.gamma_y, self.rho, self.h_interval,
                    self.sparse_coeff_i, self.thres,
                    self.imdilate_nums_kernel, self.imdilate_nums_M, self.m_sign,
                    self.net_layer_nums, self.net_down_channels, self.net_up_channels, self.net_skip_channels,
                    self.num_iter, self.use_ssim_step, self.reg_noise_std, self.tv_weight)
# ------------------------------------------------------------
#
# hp = HyperParams()
# print(hp.key_param())
