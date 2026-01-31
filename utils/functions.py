import logging

from SSIM import SSIM
from configuration.object_model import *
from idata import dpc
from models.skip import skip
from tv_loss import TVLoss
from utils.blurkernel_utils import *
from utils.blurkernel_utils import *
from skimage.io import imsave
from skimage import img_as_ubyte, img_as_uint
from utils.detection_utils import getM
from utils.ori import get_ori_loss
from tqdm import tqdm
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(net_x, optimizer, scheduler, params: Parameters, obj, net_input):
    cp = {
        "net_x": net_x.state_dict(),
        "params": params,
        "optimizer": optimizer.state_dict(),
        "basic_value": obj.x_input_basic,
        "net_input": net_input,
        "scheduler": scheduler.state_dict()
    }
    if not os.path.exists(params.save_path + "/checkpoint"):
        os.mkdir(params.save_path + "/checkpoint")
    torch.save(cp, params.save_path + "/checkpoint" + "/%s_ckpt.pth" % obj.image_name)


def save_model(save_path, **kwargs):  # ckpt, net as model
    torch.save(kwargs, save_path)


def params_write_to_file(params: Parameters):
    with open(params.save_path + "/Parameters.txt", 'w') as f:
        f.write(params.__str__())


def str_to_file(text: str, path='./str2file.txt'):
    with open(path, 'w') as f:
        f.write(text)


class Loss:
    def __init__(self, params: Parameters, device):
        # Loss
        self.mse = torch.nn.MSELoss().to(device)
        self.l1 = torch.nn.L1Loss().to(device)
        self.ssim = SSIM().to(device)
        self.tv = TVLoss(TVLoss_weight=params.tv_weight).to(device)


def get_net(model_path="", **kwargs):
    if model_path != "":
        net_x = torch.load(model_path, map_location='cuda:0')['net_x']
    else:
        net_x = skip(
            num_input_channels=kwargs['input_channels'], num_output_channels=3,
            num_channels_down=[kwargs['down_channels']] * kwargs['net_layers'],
            num_channels_up=[kwargs['up_channels']] * kwargs['net_layers'],
            num_channels_skip=[kwargs['skip_channels']] * kwargs['net_layers'],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=kwargs['pad_size'], act_fun='LeakyReLU').cuda()
    return net_x


def load_checkpoint(params, img_name):
    ckpt_path = params.save_path + "/checkpoint/%s_ckpt.pth" % img_name
    checkpoint = torch.load(ckpt_path)
    return checkpoint


def load_ckpt(path):
    return torch.load(path)


def tensor4d_to_np2d(tensor_data: torch.Tensor):
    out = torch_to_np(tensor_data)
    out = out.squeeze()
    return out


def save_img_and_kernel_as_image(net_x=None, params: Parameters = Parameters(), need_process: bool = False,
                                 img_torch=None, kernel_torch=None, obj=None, step=-1):
    dir_ = ""
    os.makedirs(os.path.join(params.save_path, dir_), exist_ok=True)

    if img_torch is not None:
        img_name = '%s_%d.png' % (obj.image_name, step) if need_process else '%s.png' % obj.image_name
        img_save_path = os.path.join(params.save_path, dir_, img_name)
        out_x_np = tensor4d_to_np2d(img_torch)
        if params.is_color:
            out_x_np = np.uint8(255 * out_x_np)
            if (step + 1) % 20 == 0:
                out_x_np = cv2.merge([out_x_np, obj.blur_image[2], obj.blur_image[1]])
                out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(img_save_path, out_x_np)

    if kernel_torch is not None:
        kernel_name = '%s_k_%d.png' % (obj.image_name, step) if need_process else '%s_k.png' % obj.image_name
        kernel_save_path = os.path.join(params.save_path, dir_, kernel_name)
        out_k_np = tensor4d_to_np2d(kernel_torch)
        out_k_np /= np.max(out_k_np)
        cv2.imwrite(kernel_save_path, np.uint8(255 * out_k_np))

    if net_x is not None:
        model_name = "%s_model.pth" % obj.image_name
        os.makedirs(os.path.join(params.save_path, dir_, "model"), exist_ok=True)
        path = os.path.join(params.save_path, dir_, "model", model_name)
        save_model(path, net_model=net_x)


def get_kernel(x, y, params: Parameters, obj, step):
    M = getM(step, x, obj.image_name, params)  # M matrix
    ksize_torch = torch.tensor((1, 1, obj.kernel_size[0], obj.kernel_size[1]))

    kernel_torch = update_kernel_torch(x, y, ksize_torch, lambda_=params.lambda_h, gamma_x=params.gamma_x,
                                       gamma_y=params.gamma_y, M=M, ii=params.sparse_coeff_i,
                                       imdilate_iter=params.imdilate_nums_kernel, iso_thres=params.thres)
    return kernel_torch


def join_path(*args):  # tuple (1, 2, 3, ...)
    base_path = ""
    for arg in args:
        base_path = os.path.join(base_path, arg)
    return base_path


def save_3channels_img(img):
    channels, h, w = img.shape
    for c in range(channels):
        Image.fromarray(np.uint8(img[c] * 255)).save("channel_%d.png" % c)

def closure(net_x, net_inputs, noises, params: Parameters, obj: TrainingObject, loss: Loss, step, need_process, who):
    loss_is_not_nan = True
    for i in range(len(params.scales)):
        noise = net_inputs[i].clone().detach()
        net_inputs[i] = net_inputs[i] + params.reg_noise_std * noise.normal_()
    y_1channels_label = obj.blur_image[0] if params.is_color else obj.blur_image
    y_3channels_label = obj.img_3channels
    outs = net_x(net_inputs)
    total_loss = torch.zeros(1).to(device)
    _, image_sizes, pad_sizes, kernel_sizes, _ = obj.get_three_input(scales=params.scales, params=params, who=who)
    out_x_nps, kernel_torchs = [], []
    for i in range(len(params.scales)):
        if len(params.scales) > 1:
            out_x = outs[i]  # ============================multi-scale============================#
        else:
            out_x = outs  # ============================single-scale============================#
        handler_out_x_nan(out_x)

        _, channels, h, w = out_x.shape
        y_label = y_3channels_label if channels == 3 else y_1channels_label
        y_label = F.interpolate(y_label, scale_factor=params.scales[i])
        y_1channels_label_scale = F.interpolate(y_1channels_label, scale_factor=params.scales[i])

        crop_w, crop_h = pad_sizes[i][0], pad_sizes[i][1]
        out_x_torch = out_x[:, :, crop_w // 2:crop_w // 2 + image_sizes[i][0],
                          crop_h // 2: crop_h // 2 + image_sizes[i][1]]
        out_x_np = torch_to_np(out_x_torch)
        out_x_pil = np_to_pil(out_x_np)
        out_x_pil_gray = out_x_pil.convert('L')
        out_x_np_gray = pil_to_np(out_x_pil_gray)
        out_x_torch_gray = torch.from_numpy(out_x_np_gray).resize(1, 1, out_x_np_gray.shape[1],
                                                                  out_x_np_gray.shape[2]).to(device)
        out_x_nps.append(out_x_np)

        obj.kernel_size = kernel_sizes[i]
        params.sparse_coeff_i = params.sparse_coeff_is[i]
        kernel_torch = get_kernel(out_x_torch_gray, y_1channels_label_scale, params, obj, step)
        t = 0
        while torch.any(torch.isnan(kernel_torch)) and t < 10:
            t = t + 1
            kernel_torch = get_kernel(out_x_torch_gray, y_1channels_label_scale, params, obj, step)
        kernel_torchs.append(kernel_torch)

        if channels == 3:
            out_y_1 = F.conv2d(out_x[0, 0, :, :].resize(1, 1, h, w), kernel_torch, padding=0, bias=None)
            out_y_2 = F.conv2d(out_x[0, 1, :, :].resize(1, 1, h, w), kernel_torch, padding=0, bias=None)
            out_y_3 = F.conv2d(out_x[0, 2, :, :].resize(1, 1, h, w), kernel_torch, padding=0, bias=None)
            out_y = torch.cat((out_y_1, out_y_2, out_y_3), dim=1)
        else:
            out_y = F.conv2d(out_x, kernel_torch, padding=0, bias=None)

        tv_loss = loss.tv(out_x_torch)
        total_loss += ((loss.mse(out_y, y_label) if step > params.use_ssim_step else 1 - loss.ssim(out_y, y_label)) + tv_loss)

    # --------------------------------------------------------------------------------------------------------------
        if torch.isnan(total_loss):
            print(step, obj.image_name, kernel_torch, out_x_torch_gray, out_y)
            torch.save(out_x, "./outxTensor_1.pt")
            torch.save(out_x_torch_gray, "./out_x_gray_Tensor_1.pt")
            torch.save(y_1channels_label_scale, "./y_1channels_myTensor_1.pt")
            return False


    total_loss.backward()

    save_point(step, total_loss, 0, params, obj, need_process, out_x_nps, kernel_torchs, net_x, tv_loss)

    return loss_is_not_nan


def optimize_ours(net_x, net_input, noise, params: Parameters, optimizer, scheduler, obj, loss, index, save_the_process,
                  who):
    for step in tqdm(range(params.num_iter),
                     desc='Training with the %s.%s, %d / %d ' % (
                             obj.image_name, params.im_type, index + 1, len(dpc.datasets(who)))):
        scheduler.step(step)
        optimizer.zero_grad()
        closure(net_x, net_input, noise, params, obj, loss, step, save_the_process)
        optimizer.step()
        if (step + 1) % params.save_frequency == 0:
            os.makedirs(params.save_path + "/checkpoint", exist_ok=True)
            path = params.save_path + "/checkpoint" + "/%s_ckpt.pth" % obj.image_name
            save_model(path, net_x=net_x.state_dict(), params=params, optimizer=optimizer.state_dict(),
                       basic_value=obj.x_input_basic, net_input=net_input, scheduler=scheduler.state_dict())


def save_point(step, total_loss, diff, params, obj, need_process, out_x_nps, kernel_torchs, net_x, tv_loss=0):
    if step % 1 == 0:
        logging.basicConfig(filename=join_path(params.save_path, "loss.txt").format(params.save_path), level=logging.INFO)
        logging.info("%s: iter %d - total loss is %f, diff is %f, x input type is %s, x input depth is %d scale is %f"
                 % (obj.image_name, step, total_loss, diff, params.init_x_type, params.x_input_depth,
                    params.scale))

    if step % params.save_frequency == 0:
        print("total loss is %f, tv loss is %f, diff is %f, x input type is %s, x input depth is %d scale is %f"
                     % (total_loss, tv_loss, diff, params.init_x_type, params.x_input_depth, params.scale))
        for i in range(len(params.scales)):
            out_x_np = out_x_nps[i]
            kernel_torch = kernel_torchs[i]
            basic_path = join_path(params.save_path)
            basic_path_model = join_path(basic_path, "model")
            os.makedirs(basic_path, exist_ok=True)
            os.makedirs(basic_path_model, exist_ok=True)

            img_name = '%s_%d_%d.png' % (obj.image_name, i, step) if need_process else '%s.png' % obj.image_name
            kernel_name = '%s_k_%d_%d.png' % (obj.image_name, i, step) if need_process else '%s_k.png' % obj.image_name
            model_name = "%s_model.pth" % obj.image_name

            img_save_path = join_path(basic_path, img_name)
            kernel_save_path = join_path(basic_path, kernel_name)
            model_save_path = join_path(basic_path_model, model_name)
            np_to_pil(out_x_np).save(img_save_path)

            out_k_np = torch_to_np(kernel_torch)
            out_k_np /= np.max(out_k_np)
            np_to_pil(out_k_np).save(kernel_save_path)
            save_model(model_save_path, net_model=net_x)

            if params.model_name == "Self-MSNet":
                save_scale = 3
            else:
                save_scale = 0

            if step > 1 and step % (params.num_iter - 1) == 0 and i == save_scale:
                path_temp_3 = "results/%s_%s" % (params.dataset_name, params.flag)
                save_path_model_3 = join_path(path_temp_3, "final_out")
                os.makedirs(save_path_model_3, exist_ok=True)
                img_save_path_3 = join_path(save_path_model_3, img_name)
                kernel_save_path_3 = join_path(save_path_model_3, kernel_name)
                np_to_pil(out_x_np).save(img_save_path_3)
                np_to_pil(out_k_np).save(kernel_save_path_3)

def hanler_total_loss_is_nan(total_loss, params, net_input, out_x, kernel_torch, out_y):
    if torch.isnan(total_loss):
        with open(params.save_path + "/loss_nan_log.txt", 'w') as f:
            f.write(str(
                {
                    "net_input": net_input,
                    "out_x": out_x,

                    "kernel_torch": kernel_torch,
                    "out_y": out_y,
                    "total_loss": total_loss,
                    "params": params.__str__(),
                }
            ))
    assert torch.isnan(total_loss)


def handler_out_x_nan(out_x):
    if torch.any(torch.isnan(out_x)):
        import warnings
        warnings.warn("<<<------------------out_x is nan-------------->>>>>>>")
