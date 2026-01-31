from torch.optim.lr_scheduler import MultiStepLR
import warnings
from utils.send_mail import auto_send_emails
from utils.functions import *
from utils.ori import *
from utils.get_pre_config import get_dataset_params, get_model
import re
import argparse

def init_torch_setting():
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return device

def handler_special_case(flag, params, obj):
    if not flag:
        return
    if params.dataset_name == "lai":
        if int(obj.image_name.split('_')[-1]) == 2:
            params.sparse_coeff_is = [5,5,5,5]
        elif int(obj.image_name.split('_')[-1]) == 3:
            params.sparse_coeff_is = [5,5,5,5]
        elif int(obj.image_name.split('_')[-1]) == 4:
           params.sparse_coeff_is = [2,3,5,7]
        else:
            params.sparse_coeff_is = [2, 2, 2, 2]
    elif params.dataset_name == "kohler":
        if int(obj.image_name.split('_')[-1]) == 8:
            params.sparse_coeff_is = [2, 2, 2, 5]
        elif int(obj.image_name.split('_')[-1]) == 9:
            params.sparse_coeff_is = [2, 2, 4, 5]
        elif int(obj.image_name.split('_')[-1]) == 10:
            params.sparse_coeff_is = [2, 2, 4, 5]
        else:
            params.sparse_coeff_is=[2, 2, 2, 2]

def prepare_params_loss(device, data_set, model_name):
    params = get_dataset_params(data_set, model_name) #get the parameter settings
    who = Who(params)  # manually_ks set for---------------------------- real image
    params_write_to_file(params)

    loss = Loss(params, device)  # loss, mes, ssim, l1, tv
    generator = dpc.get(who, params)
    return who, loss, generator, params


def handler_image_nan():

    img_is_nan_list = [7, 8, 9, 10, 19, 20, 21, 22, 31, 32, 33, 34, 43, 44, 45, 46]
    img_is_nan_list = []
    for i in range(0, 100):
        img_is_nan_list.append(i)
    return img_is_nan_list


def save_params_to_file(params):
    os.makedirs(params.save_path, exist_ok=True)
    str_to_file(params.__str__(), path=params.save_path + "/Parameters.txt")


def check_point(params, obj, net_x, optimizer, scheduler):
    # TODO Checkpoint
    if params.RESUME:
        print("load checkpoint ...")
        ckpt_path = params.save_path + "/checkpoint/%s_ckpt.pth" % obj.image_name
        print("ckpt_path", ckpt_path)
        checkpoint = load_ckpt(ckpt_path)
        net_x.load_state_dict(checkpoint['net_x'])
        obj.x_input_basic = checkpoint['basic_value']
        net_input = checkpoint['net_input']
        noise = obj.x_input_basic.clone().detach()
        hava_step = checkpoint['step']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("successful!!!")


def train_one_iter(params, scheduler, optimizer, net_x, net_inputs, obj, index, who, noises, loss, hava_step,
                   need_to_save_the_process, img_is_nan_list):
    for step in tqdm(range(params.num_iter),
                     desc='Training with the %s.%s, %d / %d ' % (
                             obj.image_name, params.im_type, index + 1, len(dpc.datasets(who)))):
        scheduler.step(step)
        optimizer.zero_grad()
        loss_is_not_nan = closure(net_x, net_inputs, noises, params, obj, loss, step + hava_step,
                                                need_to_save_the_process, who)
        if not loss_is_not_nan:  # if loss is nan
            print("loss is nan, process break, current image index is ", index)
            auto_send_emails("*********@qq.com", "Nan Happen", "Nan Happen {}".format(index))  # 1016676856
            img_is_nan_list.remove(index)
            assert loss_is_not_nan

        optimizer.step()
        if step % params.save_frequency == 0:
            os.makedirs(params.save_path + "/checkpoint", exist_ok=True)
            path = params.save_path + "/checkpoint" + "/%s_ckpt.pth" % obj.image_name
            save_model(path, net_x=net_x.state_dict(), params=params, optimizer=optimizer.state_dict(),
                       basic_value=obj.x_input_basic, net_input=net_inputs, scheduler=scheduler.state_dict(),
                       step=step)


"""
kohler: Blurry: 1-4, _: 1-12. There are 48 images.
Lai: manmade_01_kernel_01, manmade:01-05, kernel: 01-04; natural:01-05, kernel: 01-04;
people:01-05, kernel: 01-04; saturated:01-05, kernel: 01-04; text:01-05, kernel: 01-04. There are 100 images.
Levin: im1_kernel1_img, im: 1-4, kernel: 1-8, There are 32 images.
real: bird.jpg et al. There are 100 images.
"""


def begin_train(load_model, need_to_save_the_process, data_set, model_name):
    device = init_torch_setting()
    img_is_nan_list = handler_image_nan()
    who, loss, generator, params = prepare_params_loss(device, data_set, model_name)
    path_temp = params.save_path
    start = time.time()

    for index, obj in enumerate(generator):  # datasets
        params = get_dataset_params(data_type=params.dataset_name, model_name=params.model_name)
        if index not in img_is_nan_list:
            continue
        print("index is", index)
        handler_special_case(True, params, obj)

        params.save_path = path_temp + "/{}_{}".format(index + 1, obj.image_name)
        save_params_to_file(params)
        net_x = get_model(load_model, params.model_name, params, device, obj)

        net_input = obj.x_input_basic.clone().detach()
        noise = obj.x_input_basic.clone().detach()
        optimizer = torch.optim.Adam([{'params': net_x.parameters()}], lr=params.lr)
        scheduler = MultiStepLR(optimizer, milestones=params.lr_interval_list, gamma=params.lr_decay_gama)

        # checkpoint ===================================================================
        hava_step = 0  # in last iteration, some steps have been completed.
        check_point(params, obj, net_x, optimizer, scheduler)
        # ===================================================================
        start_one = time.time()
        net_inputs, _, _, _, noises = obj.get_three_input(scales=params.scales, params=params, who=who)

        train_one_iter(params, scheduler, optimizer, net_x, net_inputs, obj, index, who, noises, loss,
                       hava_step, need_to_save_the_process, img_is_nan_list)

        torch.cuda.empty_cache()

        params.save_path = path_temp
        end_one = time.time()
        print("=============cost============: {}".format(end_one - start_one))
        logging.basicConfig(filename=join_path(params.save_path, "loss.txt").format(params.save_path),
                            level=logging.INFO)
        logging.info("=============cost============: {}".format(end_one - start_one))
    end = time.time()
    print("================total cost ===============:{}".format(end - start))
    logging.basicConfig(filename=join_path(params.save_path, "loss.txt").format(params.save_path), level=logging.INFO)
    logging.info("================total cost ===============:%f" % (end - start)) # save total time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    data_set = args.data_set
    model_name = args.model_name  # Self_MSNet or Self_MSNet_S
    begin_train(load_model=False, need_to_save_the_process=True, data_set=data_set, model_name=model_name)
