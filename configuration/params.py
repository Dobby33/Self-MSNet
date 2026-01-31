# initial the super-parameters
import functools
import glob
import os
from configuration.object_model import Parameters


# ------------------------------------------------------------
def get_super_params(who):  # get super-parameters for specific datasets
    sParams = None
    if who == "lai":
        sParams = Parameters(pad_='reflection', LR_=0.01, num_iter_=2000, reg_noise_std_=0.001,
                             x_input_depth_=1, k_input_depth_=200, save_frequency_=100)
    assert sParams is not None
    return sParams


# ------------------------------------------------------------


def set_super_params():
    pass


# ------------------------------------------------------------
def get_kernel_size(kernel_name, who):
    d_kernel = None
    who = who.name
    if who == "lai":
        kernel_name = "kernel_%s" % kernel_name.split('_')[-1]
        d_kernel = {
            'kernel_01': [31, 31],
            'kernel_02': [51, 51],
            'kernel_03': [55, 55],
            'kernel_04': [75, 75]
        }
    if who == "real":
        kernel_name = "real"
        d_kernel = {
            'real': []
        }
    if who == "kohler":
        kernel_name = "kernel_%s" % kernel_name.split('_')[-1]
        d_kernel = {
            'kernel_1': [47, 35],
            'kernel_2': [41, 41],
            'kernel_3': [41, 41],
            'kernel_4': [41, 41],
            'kernel_5': [41, 41],
            'kernel_6': [41, 41],
            'kernel_7': [41, 41],
            'kernel_8': [151, 47],
            'kernel_9': [141, 141],
            'kernel_10': [139, 139],
            'kernel_11': [139, 59],
            'kernel_12': [41, 41],
        }
    assert d_kernel is not None
    return d_kernel.get(kernel_name)

# ------------------------------------------------------------
def get_datasets_path(who):
    # return the list of image name for specific datasets
    datasets_path = who.data_path
    # if you have any other datasets, please write down the da
    #     return d_kernel.get(kernel_name)
    #
    #
    # # ------------------------------------------------------------tasets_path here.
    assert datasets_path != ""

    imagePath = glob.glob(os.path.join(datasets_path, '*.%s' % who.im_type))
    if who.name == "kohler":
        imagePath.sort(key=functools.cmp_to_key(cmp))
    else:
        imagePath.sort()
    assert len(imagePath) != 0
    return imagePath


# ------------------------------------------------------------

# ------------------------------------------------------------
def cmp(x, y):
    x1 = x.split("y")[1].split(".")[0].split("_")
    y1 = y.split("y")[1].split(".")[0].split("_")
    if int(x1[0]) < int(y1[0]):
        return -1
    elif int(x1[0]) > int(y1[0]):
        return 1
    return -1 if int(x1[1]) < int(y1[1]) else 1
