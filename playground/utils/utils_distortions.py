import math
from typing import Tuple, Union

import cv2
import numpy as np
import scipy
import torch
from skimage.morphology import binary_closing, binary_opening, disk
from torch.nn import functional as F
import torchvision.transforms.functional as TF


def sign(x: float) -> int:
    return 1 if x >= 0 else -1


def fspecial(filter_type: str, p2: Union[int, Tuple[int, int]], p3: Union[int, float] = None) -> np.ndarray:
    if filter_type == 'gaussian':
        m, n = [(ss - 1.) / 2. for ss in p2]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * p3 * p3))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        return h

    elif filter_type == 'disk':
        rad = p2
        crad = math.ceil(rad - 0.5)

        x, y = np.ogrid[-rad: rad + 1, -rad: rad + 1]
        y = np.tile(y.transpose(), y.shape[1])
        x = np.tile(x, x.shape[0]).transpose()

        y = np.abs(y)
        x = np.abs(x)

        maxxy = np.maximum(x, y)
        minxy = np.minimum(x, y)

        r1 = (rad ** 2 - (maxxy + 0.5) ** 2)
        r2 = (rad ** 2 - (minxy - 0.5) ** 2)

        if (r1 > 0).all():
            warn_m1 = r1 ** 0.5
        else:
            warn_m1 = 0
        if (r2 > 0).all():
            warn_m2 = r2 ** 0.5
        else:
            warn_m2 = 0

        m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + (
                rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * warn_m1
        m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + (
                rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * warn_m2

        sgrid = (rad ** 2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) +
                             0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - (
                         maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * np.logical_or(
            np.logical_and((rad ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2),
                           (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)),
            np.logical_and(np.logical_and(minxy == 0, maxxy - 0.5 < rad), maxxy + 0.5 >= rad))

        sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad ** 2)
        sgrid[crad, crad] = np.minimum(math.pi * rad ** 2, math.pi / 2)
        if (crad > 0) and (rad > crad - 0.5) and (rad ** 2 < (crad - 0.5) ** 2 + 0.25):
            m1 = np.sqrt(rad ** 2 - (crad - 0.5) ** 2)
            m1n = m1 / rad
            sg0 = 2 * (rad ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
            sgrid[2 * crad, crad] = sg0
            sgrid[crad, 2 * crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0
            sgrid[2 * crad, crad] = sgrid[2 * crad, crad] - sg0
            sgrid[crad, 2 * crad] = sgrid[crad, 2 * crad] - sg0
            sgrid[crad, 2] = sgrid[crad, 2] - sg0
            sgrid[2, crad] = sgrid[2, crad + 1] - sg0

        sgrid[crad, crad] = np.minimum(sgrid[crad, crad], 1)
        h = sgrid / np.sum(sgrid)
        return h
    elif filter_type == 'motion':

        eps = 2.2204e-16
        length = max(1, p2)
        half_len = (length - 1) / 2.
        phi = (p3 % 180) / 180 * math.pi

        cosphi = math.cos(phi)
        sinphi = math.sin(phi)
        xsign = sign(cosphi)
        linewdt = 1

        sx = int(half_len * cosphi + linewdt * xsign - length * eps)
        sy = int(half_len * sinphi + linewdt - length * eps)
        x, y = np.mgrid[0:sx + (1 * xsign):xsign, 0:sy + 1]
        x = x.transpose()
        y = y.transpose()

        dist2line = (y * cosphi - x * sinphi)
        rad = (x ** 2 + y ** 2) ** 0.5

        lastpix = np.where(np.logical_and((rad >= half_len), (abs(dist2line) <= linewdt)))
        x2lastpix = half_len - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi);

        dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
        dist2line = linewdt + eps - np.abs(dist2line)
        dist2line[dist2line < 0] = 0

        h = np.rot90(dist2line, 2)
        tmp_h = np.zeros((h.shape[0] * 2 - 1, h.shape[1] * 2 - 1))
        tmp_h[0:h.shape[0], 0:h.shape[1]] = h
        tmp_h[(h.shape[0]) - 1:, h.shape[1] - 1:] = dist2line
        h = tmp_h

        h /= np.sum(h) + eps * length * length

        if cosphi > 0:
            h = np.flipud(h)

        return h

    else:
        raise NotImplementedError(f"Filter type {filter_type} not implemented")


def filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    img = img.float()
    k1 = kernel.size(-2)
    k2 = kernel.size(-1)

    b, c, h, w = img.size()
    if k1 % 2 == 1 or k2 % 2 == 1:
        img = F.pad(img, (k2 // 2, k2 // 2, k1 // 2, k1 // 2), mode='replicate')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k1, k2)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k1, k2).repeat(1, c, 1, 1).view(b * c, 1, k1, k2)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def curves(xx: torch.Tensor, coef: float) -> torch.Tensor:
    if type(coef) == list:
        coef = [[0.3, 0.5, 0.7],
                [coef[0], 0.5, coef[1]]]
    else:
        coef = [[0.5], [coef]]

    x = np.array([0] + [p for p in coef[0]] + [1])
    y = np.array([0] + [p for p in coef[1]] + [1])

    cs = spline(x, y)

    yy = ppval(cs, xx)

    yy = torch.clamp(yy, 0, 1)

    return yy


def spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    dd = 1
    dx = np.diff(x)
    divdif = np.diff(y) / dx

    if n == 3:
        y[1:3] = divdif
        y[2] = np.diff(divdif.T).T / (x[2] - x[0])
        y[1] -= y[2] * dx[0]
        dlk = y[[2, 1, 0]].shape[0]
        l = x[[0, 2]].shape[0] - 1
        dl = np.prod(dd) * l
        k = np.fix(dlk / dl + 100 * 2.2204e-16)

        pp = (x[[0, 2]], y[[2, 1, 0]], l, int(k), dd)

    elif n > 3:
        b = np.zeros(n)
        b[1:n - 1] = 3 * (dx[1:n] * divdif[0:n - 2] + dx[0:n - 2] * divdif[1:n])

        x31 = x[2] - x[0]
        xn = x[n - 1] - x[n - 3]

        b[0] = ((dx[0] + 2 * x31) * dx[1] * divdif[0] + dx[0] ** 2 * divdif[1]) / x31
        b[n - 1] = (dx[n - 2] ** 2 * divdif[n - 3] + (2 * xn + dx[n - 2]) * dx[n - 3] * divdif[n - 2]) / xn;

        dxt = dx.T
        c = np.zeros((3, 5))
        c[0, :] = [x31] + list(dxt[0:n - 2]) + [0]
        c[1, :] = [dxt[1]] + list(2 * (dxt[1:n - 1] + dxt[0:n - 2])) + [dxt[n - 3]]
        c[2, :] = [0] + list(dxt[1:n - 1]) + [xn]

        c = scipy.sparse.dia_matrix((c, [-1, 0, 1]), shape=(5, 5))
        c = scipy.sparse.csc_matrix(c)
        ic = scipy.sparse.linalg.inv(c)
        s = b * ic

        n = x.shape[0]
        d = 1
        dxd = dx

        dzzdx = (divdif - s[0:n - 1]) / dxd
        dzdxdx = (s[1:n] - divdif) / dxd

        coefs = np.vstack(((dzdxdx - dzzdx) / dxd, 2 * dzzdx - dzdxdx, s[0:n - 1], y[0:n - 1])).T

        pp = (x, coefs, x.shape[0], x.shape[0], d)
    else:
        raise ValueError('x.shape[0] must be >= 3')

    return pp


def ppval(pp: np.ndarray, xx: torch.Tensor) -> torch.Tensor:
    lx = torch.numel(xx)
    xs = xx.reshape(1, lx)
    b, c, l, k, dd = pp
    b = torch.as_tensor(b, device=xx.device)
    ranges = b.clone()
    ranges[0] = -torch.inf
    ranges[-1] = torch.inf
    index = histc(xs, ranges)

    xs = xs - b[index]

    c = torch.as_tensor(c, device=xx.device)

    if len(c.shape) == 1:
        v = c[0]
        for i in range(1, k):
            v = xs * v + c[i]
    else:
        v = c[index, 0]

        for i in range(1, k - 1):
            v = xs * v + c[index, i]
    v = v.view(xx.shape)
    return v


def histc(x: torch.Tensor, binranges: torch.Tensor) -> torch.Tensor:
    indices = torch.bucketize(x, binranges)
    return torch.remainder(indices, len(binranges)) - 1


def bilinear_interpolate_torch(im: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dtype_long = torch.LongTensor

    x0 = torch.floor(x).type(dtype_long).to(im.device)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long).to(im.device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    R1 = Ia * (x1 - x) / (x1 - x0 + eps) + Ic * (x - x0) / (x1 - x0 + eps)
    R2 = Ib * (x1 - x) / (x1 - x0 + eps) + Id * (x - x0) / (x1 - x0 + eps)
    P = R1 * (y1 - y) / (y1 - y0 + eps) + R2 * (y - y0) / (y1 - y0 + eps)
    return P

def skin_segmentation(x: torch.Tensor) -> torch.Tensor:
    # Convert tensor to numpy array and adjust channel order for OpenCV
    x_np = x.permute(1, 2, 0).cpu().numpy()
    x_np = (x_np * 255).astype(np.uint8)  # Assuming x is in [0, 1]
    x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

    # Converting from BGR to HSV and YCrCb color spaces
    img_HSV = cv2.cvtColor(x_np, cv2.COLOR_BGR2HSV)
    img_YCrCb = cv2.cvtColor(x_np, cv2.COLOR_BGR2YCrCb)

    # Skin color range for HSV color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Skin color range for YCrCb color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Merge skin detection (YCrCb and HSV)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    # Convert mask back to tensor
    global_mask = torch.from_numpy(global_mask).to(torch.float32) / 255.0

    # Apply morphological operations using skimage which expects boolean arrays
    global_mask = global_mask.bool()
    global_mask = binary_opening(global_mask, disk(2))
    global_mask = binary_closing(global_mask, disk(5))

    return global_mask

def perspective_transform(x: torch.Tensor, src_points: torch.Tensor, dst_points: torch.Tensor) -> torch.Tensor:
    """Applies a perspective transformation to an image given source and destination points as tensors."""
    src_points_list = src_points.tolist()
    dst_points_list = dst_points.tolist()
    x_pil = TF.to_pil_image(x)
    x_transformed = TF.perspective(x_pil, src_points_list, dst_points_list)
    return TF.to_tensor(x_transformed)

def apply_perspective(x: torch.Tensor, amount: float, mode: str) -> torch.Tensor:
    """Generic function to apply perspective changes based on the mode."""
    if amount == 0.0:
        return x
    w, h = x.size(2), x.size(1)
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone()

    if mode == 'top':
        dst_points[2][0] -= w * amount  # Adjust left point
        dst_points[3][0] += w * amount  # Adjust right point
    elif mode == 'bottom':
        dst_points[0][0] -= w * amount  # Adjust left point
        dst_points[1][0] += w * amount  # Adjust right point
    elif mode == 'left':
        dst_points[1][1] -= h * amount  # Adjust top point
        dst_points[3][1] += h * amount  # Adjust bottom point
    elif mode == 'right':
        dst_points[0][1] -= h * amount  # Adjust top point
        dst_points[2][1] += h * amount  # Adjust bottom point

    return perspective_transform(x, src_points, dst_points)