from utils.blurkernel_utils import *
from configuration.object_model import *
import numpy as np
import kornia
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def percentile(t: torch.tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def prctile(x, p):
    return np.percentile(x, p, interpolation='midpoint')  # the result same as matlab


def prctile_torch(x, p):
    return torch.quantile(x, .01 * float(p))


def guassian(img, size=(3, 3)):
    return cv2.GaussianBlur(img, size, 0)


def guassian_torch(img, size=(3, 3)):
    sigma = 0.3 * ((size[0] - 1) * 0.5 - 1) + 0.8
    return kornia.filters.gaussian_blur2d(img, kernel_size=size, sigma=(sigma, sigma))


def getM2(img, bd_type=cv2.BORDER_REFLECT, p=90):
    dir_n = 4
    mask1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # x
    mask2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])  # 45
    mask3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # y
    mask4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  # 135
    mask5 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    mask6 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    mask7 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    mask8 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    mask = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]

    h, w = img.shape
    omasks = np.zeros((h, w, dir_n))
    for i in range(dir_n):
        tmp = cv2.filter2D(img.astype('float32'), -1, mask[i], borderType=bd_type)
        tmp = tmp ** 2
        currth = prctile(tmp, p)  # threshold value
        currm = (tmp > currth).astype(int)
        omasks[:, :, i] = currm
    M1 = np.sum(omasks, 2)
    M2 = (M1 > 0).astype(int)
    return M2


def getM2_torch(img, p=90):
    dir_n = 4
    mask1 = torch.from_numpy(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])).to(device)  # x
    mask2 = torch.from_numpy(np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])).to(device)  # 45
    mask3 = torch.from_numpy(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])).to(device)  # y
    mask4 = torch.from_numpy(np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])).to(device)  # 135
    mask5 = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).to(device)
    mask6 = torch.from_numpy(np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])).to(device)
    mask7 = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])).to(device)
    mask8 = torch.from_numpy(np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])).to(device)
    mask = [mask1.resize(1, 3, 3), mask2.resize(1, 3, 3), mask3.resize(1, 3, 3), mask4.resize(1, 3, 3),
            mask5.resize(1, 3, 3), mask6.resize(1, 3, 3), mask7.resize(1, 3, 3), mask8.resize(1, 3, 3)]
    _, _, h, w = img.shape
    omasks = torch.zeros((dir_n, h, w)).to(device)
    for i in range(dir_n):
        tmp = kornia.filters.filter2D(img, mask[i]).to(device)  # default: reflect
        tmp = tmp ** 2
        currth = prctile_torch(tmp, p)  # threshold value
        currm = (tmp > currth)
        omasks[i, :, :] = currm

    M1 = torch.sum(omasks, 0)
    M2 = (M1 > 0).int()
    return M2.resize(1, 1, M2.shape[0], M2.shape[1]).to(device)


def my_sobel(img, bd_type=cv2.BORDER_REFLECT):
    mask1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # x
    mask2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])  # 45
    mask3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # y
    mask4 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  # 135

    diff_gray1 = cv2.filter2D(img.astype('float32'), -1, mask1, borderType=bd_type)
    diff_gray2 = cv2.filter2D(img.astype('float32'), -1, mask2, borderType=bd_type)
    diff_gray3 = cv2.filter2D(img.astype('float32'), -1, mask3, borderType=bd_type)
    diff_gray4 = cv2.filter2D(img.astype('float32'), -1, mask4, borderType=bd_type)

    diff_gray_mat = np.maximum(abs(diff_gray1), abs(diff_gray2))
    diff_gray_mat = np.maximum(diff_gray_mat, abs(diff_gray3))
    diff_gray_mat = np.maximum(diff_gray_mat, abs(diff_gray4))

    return diff_gray_mat


def edge_detection(img, p=0.5):  # for our edge detection
    result = my_sobel(img)
    currth = prctile(result, p)
    return (result > currth).astype(int)


def binaryzation(im, thres=0.5):  # for sobel
    _, img_bin = cv2.threshold(im, 0, 1, cv2.THRESH_BINARY)  # _ return thresholds
    return img_bin


def getM(step, img, image_name, HParams: Parameters):
    M = None
    if step >= HParams.m_sign:
        clear_torch = guassian_torch(img, size=(5, 5))
        HParams.p = HParams.p * HParams.beta if (step + 1) % HParams.p_interval == 0 else HParams.p
        M = getM2_torch(clear_torch, p=HParams.p)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        M = imdilate_torch(M, kernel, HParams.imdilate_nums_M)
    return M