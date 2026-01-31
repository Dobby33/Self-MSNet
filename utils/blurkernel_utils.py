from torch import tensor

from utils.common_utils import *
from skimage.measure import label, regionprops


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def otf2psf_(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))
    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)

    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)

        n = max(np.size(outsize), np.size(insize))
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))

        outsize = np.pad(colvec_out, ((0, max(0, n - np.size(colvec_out))), (0, 0)), mode="constant")
        insize = np.pad(colvec_in, ((0, max(0, n - np.size(colvec_in))), (0, 0)), mode="constant")

        pad = (insize - outsize) / 2

        if np.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")

        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)

        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf


def otf2psf_torch(otf, outsize=None):
    insize = torch.tensor(otf.shape).to(device)
    psf = torch.fft.ifftn(otf, dim=(2, 3)).to(device)

    for axis, axis_size in enumerate(insize):
        psf = torch.roll(psf, np.floor(axis_size.cpu().numpy() / 2).astype(int), dims=axis).to(device)

    if outsize is not None:
        insize = torch.tensor(otf.shape).to(device)
        outsize = torch.tensor(outsize).to(device)

        n = max(torch.numel(outsize), torch.numel(insize))
        colvec_out = torch.flatten(outsize).reshape((1, 1, torch.numel(outsize), 1)).to(device)
        colvec_in = torch.flatten(insize).reshape((1, 1, torch.numel(insize), 1)).to(device)

        padding1 = (
            0, 0,
            0, 0,
            0, max(0, n - torch.numel(colvec_out)),
            0, 0
        )
        padding2 = (
            0, 0,
            0, 0,
            0, max(0, n - torch.numel(colvec_in)),
            0, 0
        )

        outsize = F.pad(colvec_out, padding1, mode="constant").to(device)
        insize = F.pad(colvec_in, padding2, mode="constant").to(device)

        pad = (insize - outsize) / 2

        if torch.any(pad < 0):
            print("otf2psf error: OUTSIZE must be smaller than or equal than OTF size")

        prepad = torch.floor(pad).to(device)
        postpad = torch.ceil(pad).to(device)

        dims_start = prepad.long()  # to int
        dims_end = (insize - postpad).long()

        psf = torch.real(psf).to(device)
        # TODO: this
        # for i in range(len(dims_start.shape)):
        #     start_n = dims_start[0][0][i][0].item() * i
        #     end_n = dims_end[0][0][i][0].item() * i
        #     print("start_n = ", start_n, "\n", "end_n = ", end_n)
        #     psf = torch.gather(psf, index=torch.arange(start_n, end_n), dim=i)

    return psf


def otf2psf(otf, outsize=None):
    k = ifft2(otf)
    k = ifftshift(k)
    w, h = k.shape
    i3_start = int(np.floor((w - outsize[0]) // 2))
    i3_stop = i3_start + outsize[0]

    i4_start = int(np.floor((h - outsize[1]) // 2))
    i4_stop = i4_start + outsize[1]
    k = k[i3_start: i3_stop, i4_start: i4_stop]
    return k


def otf2psf_mytorch(otf, outsize=None):
    k = fft.ifft2(otf)
    k = fft.ifftshift(k)
    if outsize is not None:
        _, _, w, h = k.shape
        i3_start = int(torch.floor((w - outsize[-2]) / 2).item())
        i3_stop = int(i3_start + outsize[-2].item())

        i4_start = int(torch.floor((h - outsize[-1]) / 2).item())
        i4_stop = int(i4_start + outsize[-1].item())
        k = k[:, :, i3_start: i3_stop, i4_start: i4_stop]
    return k


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops * 2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def otf2psf_as(otf, psf_size):  # is true
    # calculate psf from otf with size <= otf size
    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(psf.imag)) / np.max(abs(psf)) <= num_small:
            psf = psf.real
            # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0] / 2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1] / 2)), axis=1)
        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf


def otf2psf_as_torch(otf, psf_size=None):  # is true
    # calculate psf from otf with size <= otf size
    psf = fft.ifftn(otf).to(device)
    if psf_size is not None:
        # this condition depends on psf size
        num_small = np.log2(otf.shape[2]) * 4 * np.spacing(1)  # np.spacing(1) is 2.220446049250313e-16
        if torch.max(abs(psf.imag)) / torch.max(abs(psf)) <= num_small:
            psf = psf.real.to(device)

            # circularly shift psf
        psf = torch.roll(psf, int(np.floor(psf_size[2] / 2)), dims=2).to(device)
        psf = torch.roll(psf, int(np.floor(psf_size[3] / 2)), dims=3).to(device)

        # crop psf
        psf = psf[:, :, 0: psf_size[2], 0: psf_size[3]].to(device)

    return psf


def psf2otf_as_torch(psf, sizek, otf_size=None):  # is true
    if otf_size is not None:
        # padding psf to oft_size
        psf_size = sizek
        pad_rightsize = otf_size[3] - sizek[3]
        pad_downsize = otf_size[2] - sizek[2]
        psf = F.pad(psf, (0, pad_rightsize, 0, pad_downsize), 'constant')

        # circularly shift psf
        psf = torch.roll(psf,  int(-np.floor(psf_size[3] / 2)), dims=3)
        psf = torch.roll(psf,  int(-np.floor(psf_size[2] / 2)), dims=2)

        # calculate otf
        otf = fft.fftn(psf).to(device)
        # this condition depends on psf size
        num_small = np.log2(psf_size[2]) * 4 * np.spacing(1)  # np.spacing(1) is 2.220446049250313e-16
        if torch.max(abs(otf.imag)) / torch.max(abs(otf)) <= num_small:
            otf = otf.real.to(device)
    else:
        otf = fft.fftn(psf).to(device)

    return otf


def partial_support(k, i):
    k_f = k.flatten()
    k_sorted = np.sort(k_f)
    thres = k_sorted[-1] / (2 * i * math.sqrt(len(k_sorted)))

    diff_k = np.diff(k_sorted)
    j = np.argwhere(diff_k > thres)

    if len(j) == 0:
        epsilon_s = 0
    else:
        epsilon_s = k_sorted[j.min()]
    return np.where(k >= epsilon_s, 1, 0)


def partial_support_torch(k, i=2):  # is true
    k_f = k.flatten().to(device)
    k_sorted, indices = torch.sort(k_f)
    thres = k_sorted[-1] / (2 * i * math.sqrt(len(k_sorted)))
    diff_k = torch.diff(k_sorted).to(device)
    j = (diff_k > thres).nonzero()  # np.argwhere --> nonzero()
    if len(j) == 0:
        epsilon_s = 0
    else:
        epsilon_s = k_sorted[j.min()]
    return torch.where(k >= epsilon_s, 1, 0).to(device)


def imgradientxy(I):
    size = np.shape(I)
    Gx = np.zeros(size)
    Gx[:, 0: -1] = I[:, 1:] - I[:, 0: -1]

    Gy = np.zeros(size)
    Gy[0: -1, :] = I[1:, :] - I[0: -1, :]
    return Gx, Gy


def imgradientxy_torch(I):  # is true
    shape = I.shape
    Gx = torch.zeros(shape).to(device)
    Gy = torch.zeros(shape).to(device)
    Gx[-1, -1, :, 0: -1] = I[-1, -1, :, 1:] - I[-1, -1, :, 0: -1]
    Gy[-1, -1, 0: -1, :] = I[-1, -1, 1:, :] - I[-1, -1, 0: -1, :]
    return Gx, Gy


def imdilate(im, kernel, iter=1):
    im_dilate = cv2.dilate(im, kernel=kernel, iterations=iter)
    return im_dilate


def imdilate_torch(im, kernel, iter=1):  # is true, iter = 1
    im_dilate = None
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).to(device)
    for i in range(iter):
        im_dilate = torch.clamp(F.conv2d(im.to(device, dtype=torch.float), kernel_tensor, padding=(1, 1)), 0, 1).to(device)
        im = im_dilate

    assert im_dilate is not None
    return im_dilate


def prune_iso(bw_im, im, thres):  # is true
    con_labels, nums = label(bw_im, connectivity=2, return_num=True, background=0)
    if nums == 0 or nums == 1:
        return im
    else:
        for i in range(1, nums + 1):
            c_domain_b = (con_labels == i)  # bool matrix
            c_domain_v = c_domain_b * im  # float matrix
            if np.sum(c_domain_v) < thres:
                im = (c_domain_v == 0) * im
    return im / np.sum(im)


def prune_iso_torch(bw_im, im, thres):
    bw_im = torch_to_np(bw_im)
    im = torch_to_np(im)
    con_labels, nums = label(bw_im, connectivity=2, return_num=True, background=0)
    if nums == 0 or nums == 1:
        return np_to_torch(im).to(device)
    else:
        for i in range(1, nums + 1):
            # connected_domain
            c_domain_b = (con_labels == i)  # bool matrix
            c_domain_v = c_domain_b * im  # float matrix
            if np.sum(c_domain_v) < thres:
                im = (c_domain_v == 0) * im
    im = im / np.sum(im)
    return np_to_torch(im).to(device)


def get_const(x, y, lambda_):
    Gxx, Gxy = imgradientxy(x)
    FGxx = fft2(Gxx)
    FGxy = fft2(Gxy)
    Gyx, Gyy = imgradientxy(y)
    FGyx = fft2(Gyx)
    FGyy = fft2(Gyy)
    nomin = np.conj(FGxx) * FGyx + np.conj(FGxy) * FGyy
    deno = abs(FGxx) ** 2 + abs(FGxy) ** 2 + lambda_
    return nomin, deno


def get_const_torch(x, y, lambda_, M=None):  # M matrix
    if M is None:
        M = torch.ones(x.shape).type_as(x.data)

    Gxx, Gxy = imgradientxy_torch(x)
    FGxx = torch.fft.fft2(Gxx * M).to(device)
    FGxy = torch.fft.fft2(Gxy * M).to(device)

    Gyx, Gyy = imgradientxy_torch(y)
    FGyx = torch.fft.fft2(Gyx).to(device)
    FGyy = torch.fft.fft2(Gyy).to(device)

    nomin = torch.conj(FGxx) * FGyx + torch.conj(FGxy) * FGyy
    deno = abs(FGxx) ** 2 + abs(FGxy) ** 2 + lambda_

    import warnings
    warnings.warn("get_const_torch function, nomin is zero, please debug")
    return nomin, deno


def update_kernel_L2(x, y, sizek, lambda_):
    nomin, deno = get_const(x, y, lambda_)
    k = np.real(otf2psf_as((nomin / deno), sizek))
    return k


def update_kernel_L2_torch(x, y, sizek, lambda_, M=None):
    nomin, deno = get_const_torch(x, y, lambda_, M=M)
    k = otf2psf_as_torch((nomin / deno), sizek).to(device)
    k = torch.real(k) if torch.is_complex(k) else k
    return k


def update_kernel_L2_Zhang_torch(x, y, sizek, lambda_, gamma_x, gamma_y, M=None):
    nomin, deno = get_const_torch(x, y, lambda_, M=M)

    # matrix Z
    Z = otf2psf_as_torch((nomin / deno), sizek).to(device)

    # matrix U V
    m, n = sizek[2], sizek[3]
    x0, y0 = 1 / 2, 1 / 2
    U = ((torch.arange(1, m + 1)/m).reshape(m, 1).type(torch.FloatTensor) @ torch.ones((1, n)) - x0).to(device)
    V = (torch.ones((m, 1)) @ (torch.arange(1, n+1)/n).reshape(1, n).type(torch.FloatTensor) - y0).to(device)

    # matrix F K
    uns_U = U.reshape(1, 1, m, n)
    uns_V = V.reshape(1, 1, m, n)

    otfsize = nomin.shape[2:]

    F = torch.fft.ifft2((torch.fft.fft2(uns_U, otfsize) / deno)).to(device)
    K = torch.fft.ifft2((torch.fft.fft2(uns_V, otfsize) / deno)).to(device)
    F = F[:, :, :m, :n]
    K = K[:, :, :m, :n]

    # matrix A
    nn = m * n
    vF, vK = F.reshape(-1) * gamma_x, K.reshape(-1) * gamma_y
    vU, vV, vZ = U.reshape(-1).type_as(vF), V.reshape(-1).type_as(vK), Z.reshape(-1)
    A = vF.unsqueeze(dim=1) @ vU.unsqueeze(dim=0) + vK.unsqueeze(dim=1) @ vV.unsqueeze(dim=0)
    A = A + torch.eye(nn, dtype=torch.complex64).to(device)

    if vZ.dtype != torch.complex64:
        vZ = vZ.type(torch.complex64)

    k = torch.linalg.solve(A.t(), vZ).to(device)

    k = k.reshape(m, n)
    k = torch.real(k) if torch.is_complex(k) else k
    k = k.reshape(1, 1, m, n)
    return k


def update_kernel(x, y, sizek, lambda_, ii=2, thres=0.01):
    k = update_kernel_L2(x, y, sizek, lambda_)
    k[k < 0] = 0
    S = partial_support(k, ii)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    S = imdilate(np.uint8(S), kernel=kernel)
    k = S * k
    k /= np.sum(k)
    k = prune_iso(S, k, thres)
    k = np.rot90(k, k=2)
    return k


def update_kernel_torch(x, y, sizek, lambda_, gamma_x, gamma_y, ii=2, iso_thres=0.01, M=None, imdilate_iter=1):
    # k = update_kernel_L2_torch(x, y, sizek, lambda_, M=M)
    k = update_kernel_L2_Zhang_torch(x, y, sizek, lambda_, gamma_x, gamma_y, M=M)
    k[k < 0] = 0
    S = partial_support_torch(k, ii)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    S = imdilate_torch(S, kernel=kernel, iter=imdilate_iter)
    k = S * k
    if torch.sum(k) == 0:
        k[0, 0, 0, 0] = 1
    else:
        k /= torch.sum(k)
    k = prune_iso_torch(S, k, iso_thres)
    k = torch.rot90(k, k=2, dims=(2, 3))
    return k

def update_kernel_torch_li(x, y, sizek, lambda_, gamma_x, gamma_y, ii=2, iso_thres=0.01, M=None, imdilate_iter=1):
    # k = update_kernel_L2_torch(x, y, sizek, lambda_, M=M)
    k = update_kernel_L2_Zhang_torch(x, y, sizek, lambda_, gamma_x, gamma_y, M=M)
    k[k < 0] = 0
    S = partial_support_torch(k, ii)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    S = imdilate_torch(S, kernel=kernel, iter=imdilate_iter)
    k = S * k
    if torch.sum(k) == 0:
        k[0, 0, 0, 0] = 1
    else:
        k /= torch.sum(k)
    k = prune_iso_torch(S, k, iso_thres)
    k = torch.rot90(k, k=2, dims=(2, 3))
    return k