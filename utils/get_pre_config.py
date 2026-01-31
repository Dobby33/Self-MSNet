from configuration.object_model import Parameters
from models.skip import skip
from models_try.mimo_unet import MIMOUNet, MIMOUNetSimple
from models_try.mimo_unet_simple import MIMOUNetSimple1, MIMOUNetSimple3, MIMOUNet_Up_mutiply
from models_try.mimo_dip import MIMO_DIP
from models_try.mimo_res_net import Self_MSNet, Self_MSNet_S
import torch


def get_dataset_params(data_type, model_name):
    if data_type not in ["real", "lai", "kohler"]:
        assert Exception("data set error not int [real, lai, kohler]")

    if model_name == "Self_MSNet":
        scales = [1/8, 1/4, 1/2, 1]
    elif model_name == "Self_MSNet_S":
        scales = [1]
    else:
        print("Please enter the exact model name!")

    if data_type == "real":
        params = Parameters(dataset="real",
                            RESUME=False, scales=scales,
                            x_type="noise", x_input_depth_=16, x_output_depth_=3, flag="deblurred",
                            lr_=0.01, lr_interval_list=[700, 900, 1300], lr_decay_gama=0.5,
                            p=90, beta=0.99, p_interval=200, gamma_x=10, gamma_y=10,
                            lambda_h=2e-5, rho=1.2, h_interval=100,
                            imdilate_nums_kernel=1, imdilate_nums_M=1, m_sign=1,
                            sparse_coeff_i=2, thres=0.01, sparse_coeff_is=[2, 2, 2, 2],
                            num_iter_=1500 + 1, use_ssim_step=0, reg_noise_std_=0.001, tv_weight=0.0,
                            save_frequency_=100, model_name=model_name)
        return params
    if data_type == "lai":
        params = Parameters(dataset="lai",
                            RESUME=False, scale=1, scales=scales,
                            x_type="noise", x_input_depth_=16, x_output_depth_=3, flag="deblurred",
                            lr_=0.001, lr_interval_list=[500, 1000, 1500, 2000, 2500, 3500], lr_decay_gama=0.5,
                            p=90, beta=0.99, p_interval=200, gamma_x=10, gamma_y=10,
                            lambda_h=10, rho=1.2, h_interval=100,
                            imdilate_nums_kernel=1, imdilate_nums_M=1, m_sign=300,
                            sparse_coeff_i=2, sparse_coeff_is=[2, 2, 2, 2], thres=0.01, net_layer_nums=5,
                            num_iter_=2001, use_ssim_step=3000, reg_noise_std_=0.001, tv_weight=0,
                            save_frequency_=100, model_name=model_name)
        return params
    if data_type == "kohler":
        params = Parameters(dataset="kohler",
                            # lr_interval_list=[700, 1500, 1800], use_ssim_step=5000
                            RESUME=False, scales=scales,
                            x_type="noise", x_input_depth_=16, x_output_depth_=3, flag="deblurred",
                            p=90, beta=0.99, p_interval=200, lr_interval_list=[], lr_decay_gama=0.5,  lr_=0.01,
                            lambda_h=120, gamma_x=10, gamma_y=10, rho=1.2, h_interval=100,
                            imdilate_nums_kernel=1, imdilate_nums_M=1, m_sign=0,
                            sparse_coeff_i=2, thres=0.01,
                            num_iter_=2001, use_ssim_step=5000, reg_noise_std_=0.005, save_frequency_=100, tv_weight=1.0,
                            sparse_coeff_is=[2, 4, 5, 6], model_name=model_name)
        return params
    return None


def get_model(load_model, model_type, params, device, obj):
    net_x = skip(
        num_input_channels=params.x_input_depth, num_output_channels=params.x_output_depth,
        num_channels_down=[params.net_down_channels] * params.net_layer_nums,
        num_channels_up=[params.net_up_channels] * params.net_layer_nums,
        num_channels_skip=[params.net_skip_channels] * params.net_layer_nums,
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad=params.pad, act_fun='LeakyReLU').to(device)
    if load_model:
        print("load model ...")
        net_x.load_state_dict(
            torch.load('results/model/kohler/%s_model.pth' % obj.image_name, map_location='cuda:0'),
            strict=False)
        print("successful ...")

    if model_type == "u-net":
        return net_x
    if model_type == "mimo-unet":
        net_x = MIMOUNet().to(device)
    if model_type == "Self_MSNet":
        net_x = Self_MSNet(params.x_input_depth, params.x_output_depth).to(device)
    if model_type == "Self_MSNet_S":
        net_x = Self_MSNet_S(params.x_input_depth, params.x_output_depth).to(device) #ablation study
    return net_x
