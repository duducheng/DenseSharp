import collections
from itertools import repeat
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import find_contours


def plot_voxel(arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    _, axes = plt.subplots(length, 1, figsize=(4, 4 * length))
    for i, ax in enumerate(axes):
        ax.set_title("@%s" % i)
        ax.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            ax.imshow(aux[i], alpha=0.3)
    plt.show()


def plot_voxel_save(path, arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    for i in range(length):
        plt.clf()
        plt.title("@%s" % i)
        plt.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.2)
        plt.savefig(path + "%s.png" % i)


def plot_voxel_enhance(arr, arr_mask=None, figsize=10, alpha=0.1):  # zyx
    '''borrow from yuxiang.'''
    plt.figure(figsize=(figsize, figsize))
    rows = cols = int(round(np.sqrt(arr.shape[0])))
    img_height = arr.shape[1]
    img_width = arr.shape[2]
    assert img_width == img_height
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
    if arr_mask is not None:
        res_mask_img = np.zeros(
            (rows * img_height, cols * img_width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row * cols + col) >= arr.shape[0]:
                continue
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height,
            target_x:target_x + img_width] = arr[row * cols + col]
            if arr_mask is not None:
                res_mask_img[target_y:target_y + img_height,
                target_x:target_x + img_width] = arr_mask[row * cols + col]
    plt.imshow(res_img, plt.cm.gray)
    if arr_mask is not None:
        plt.imshow(res_mask_img, alpha=alpha)
    plt.show()


def find_edges(mask, level=0.5):
    edges = find_contours(mask, level)[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    xs, ys = find_edges(aux, level)
    ax.plot(xs, ys)


def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''
    shape = voxel.shape
    # z, y, x = zyx
    # d, h, w = dhw
    crop_pos = []
    padding = [[0, 0], [0, 0], [0, 0]]
    for i, (center, length) in enumerate(zip(zyx, dhw)):
        assert length % 2 == 0
        # assert center < shape[i] # it's not necessary for "moved center"
        low = round(center) - length // 2
        high = round(center) + length // 2
        if low < 0:
            padding[i][0] = int(0 - low)
            low = 0
        if high > shape[i]:
            padding[i][1] = int(high - shape[i])
            high = shape[i]
        crop_pos.append([int(low), int(high)])
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1]
                                                   [0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:
        cropped = np.lib.pad(cropped, padding, 'constant',
                             constant_values=fill_with)
    return cropped


def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''
    # assert v.min() <= window_low
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)


def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx


def get_uniform_assign(length, subset):
    assert subset > 0
    per_length, remain = divmod(length, subset)
    total_set = np.random.permutation(list(range(subset)) * per_length)
    remain_set = np.random.permutation(list(range(subset)))[:remain]
    return list(total_set) + list(remain_set)


def split_validation(df, subset, by):
    df = df.copy()
    for sset in df[by].unique():
        length = (df[by] == sset).sum()
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)
    df['subset'] = df['subset'].astype(int)
    return df


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
