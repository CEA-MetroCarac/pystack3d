"""
utilities functions
"""
import numpy as np
from tifffile import imread, imwrite


def outputs_saving(output_dirname, fname, img, img_res, stats):
    """ Append stats and Write img_res after reformatting """
    img_res2 = img_reformatting(img_res, img.dtype)

    stats.append([[img.min(), img.max(), img.mean()],
                  [img_res.min(), img_res.max(), img_res.mean()],
                  [img_res2.min(), img_res2.max(), img_res2.mean()]])

    imwrite(output_dirname / fname.name, img_res2, dtype=img.dtype)


def img_reformatting(arr, dtype):
    """ Reformat the array in the machine limits related to the data type """
    try:
        info = np.iinfo(dtype)
    except:
        info = np.finfo(dtype)
    return np.clip(arr, info.min, info.max).astype(dtype)


def imread_3d_skipping(fnames, skip_factors=(10, 10, 10)):
    """ Return a 3D-Array from .tif files wrt skip factors in all directions """
    arr0 = imread(fnames[0])
    shape = arr0.shape
    dtype = arr0.dtype

    arr_red = np.zeros((int(np.ceil(len(fnames) / skip_factors[2])),
                        int(np.ceil(shape[0] / skip_factors[1])),
                        int(np.ceil(shape[1] / skip_factors[0]))), dtype=dtype)

    for k, fname in enumerate(fnames[::skip_factors[2]]):
        arr_red[k, ...] = imread(fname)[::skip_factors[1], ::skip_factors[0]]

    return arr_red


def skipping(arr, skip_factors):
    """ Skip the array in each axis """
    arr_red = arr.copy()
    for k in range(arr_red.ndim):
        arr_red = np.swapaxes(arr_red, 0, k)
        arr_red = arr_red[::skip_factors[k], ...]
        arr_red = np.swapaxes(arr_red, 0, k)
    return arr_red


def division(num):
    """ Return the highest dividers pair. Ex: 6 -> (3, 2), 12 -> (4, 3) """
    for i in reversed(range(1, int(num ** 0.5) + 1)):
        if num % i == 0:
            return num // i, i


def cumdot(mat):
    """ Return the cumulative matrix product """
    mat = np.asarray(mat)
    cum_mat = np.empty(mat.shape)
    cum_mat[0] = mat[0]
    for i in range(1, mat.shape[0]):
        cum_mat[i] = cum_mat[i - 1] @ mat[i]
    return cum_mat
