"""
Functions related to the registration transformation processing
"""
import os
from itertools import count
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.transform import PiecewiseAffineTransform, AffineTransform, warp
from scipy.ndimage import uniform_filter1d
from tifffile import TiffFile

from pystack3d.registration_calculation import registration_plot
from pystack3d.utils import cumdot, outputs_saving
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def init_args(params, shape):
    """
    Initialize arguments related to the current processing
    ('registration_transformation')

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process
    """
    tmats = None
    for k in [0, 1]:
        dir_parent = params['output_dirname'].parents[k]
        if dir_parent.name == "process":
            fname = dir_parent / "registration_calculation" / "tmats.npy"
            if os.path.exists(fname):
                tmats = np.load(fname)

    if tmats is None:
        raise IOError("File 'tmats.npy' not found")

    params.update({'pre_calculated_tmats': tmats})


def registration_transformation(fnames=None,
                                inds_partition=None, queue_incr=None,
                                nb_blocks=None,
                                pre_calculated_tmats=None,
                                constant_drift=None,
                                box_size_averaging=None,
                                subpixel=True,
                                mode='constant',
                                cropping=False,
                                output_dirname=None):
    """
    Function dedicated to transformation matrices application for the image
    registration

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    nb_blocks: tuple of 2 ints (p, q), optional
        Number of patches (sub-images) to consider in each axis to perform
        the registrations. If None, consider the whole image
    pre_calculated_tmats: numpy.ndarray((nslices * nb_blocks, 3, 3)), optional
        Transformations matrices previously calculated
    constant_drift: tuple of 2 floats, optional
        Constant translation values (transl_x, transl_y) to remove at each
        slice in the transformation matrices
    box_size_averaging: int, optional
        Box size associated to the running-averaged transformation matrices
        to remove at each slice
    subpixel: bool, optional
        Activation keyword to enable subpixel translation
    mode: {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        For extrapolation, points outside the boundaries of the input are
        filled according to the given mode that is related to `numpy.pad`.
        If 'constant' (Default mode), cval value is automatically set to
        np.nan.
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if pre_calculated_tmats is None:
        raise IOError("'pre_calculated_tmats' should be given")
    tmats = pre_calculated_tmats

    # remove subpixel translation
    if not subpixel:
        for k, tmat in enumerate(tmats):
            tmat_transl = tmat[:, :, 2]
            tmat_transl[np.abs(tmat_transl) < 1] = 0

    # cumulative transformation matrices by dot products
    tmats_cumul = cumdot(tmats)

    # constant drift or running average removal
    if constant_drift is not None:
        tmats_cumul = constant_drift_removal(tmats_cumul, constant_drift)
    if box_size_averaging is not None:
        box_size = max(box_size_averaging, 0)
        tmats_cumul = running_avg_removal(tmats_cumul, box_size=box_size)

    if cropping:
        with TiffFile(fnames[0]) as tiff:
            shape = tiff.asarray().shape
        inds_crop, img_crop = inner_rectangle(shape, tmats_cumul, nb_blocks)
        imin, imax, jmin, jmax = inds_crop

    if pid_0:
        np.save(output_dirname / 'outputs' / 'tmats_cumul.npy', tmats_cumul)
        if cropping:
            np.savetxt(output_dirname / 'outputs' / 'inds_crop.txt', inds_crop)
            np.save(output_dirname / 'outputs' / 'img_crop.npy', img_crop)

    # transformation matrices application
    stats = []
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        tmat = tmats_cumul[inds_partition[k]]
        dtype_orig = img.dtype
        img_res = img_transformation(img.astype(float), tmat,
                                     nb_blocks=nb_blocks, mode=mode)
        img_res = img_res.astype(dtype_orig)

        if cropping:
            img_res = img_res[imin:imax, jmin:jmax]

        outputs_saving(output_dirname, fname, img, img_res, stats)

        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    if not os.path.exists(output_dirname / 'outputs' / 'tmats_cumul.npy'):
        return

    tmats_cumul = np.load(output_dirname / 'outputs' / 'tmats_cumul.npy')
    fig = registration_plot(tmats_cumul, title='Registration (cumul.)')
    plt.savefig(output_dirname / 'outputs' / 'tmats_cumul_evol.png')
    plt.close(fig)

    if os.path.exists(output_dirname / 'outputs' / 'inds_crop.txt'):
        inds_crop = np.loadtxt(output_dirname / 'outputs' / 'inds_crop.txt')
        img_crop = np.load(output_dirname / 'outputs' / 'img_crop.npy')

        imin, imax, jmin, jmax = inds_crop
        rect = patches.Rectangle((jmin - 0.5, imin - 0.5),
                                 jmax - jmin, imax - imin,
                                 lw=1, ec='r', fc='none')
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title('registration_transformation')
        ax.imshow(img_crop, cmap='gray', origin='lower')
        ax.add_patch(rect)
        plt.savefig(output_dirname / 'outputs' / 'registration_area.png')
        plt.close(fig)


def img_transformation(img, tmats, nb_blocks=None,
                       mode='constant', cval=np.nan, order=None):
    """
    Apply affine or piecewise affine transformation to operate the image
    registration according to 'nb_blocks'

    Parameters
    ----------
    img: numpy.ndarray((m,n))
        Image considered as 'moving' image
    tmats: numpy.ndarray((p*q, 3*3))
        Transformation matrix calculated in all patches
    nb_blocks: tuple of 2 ints (p, q), optional
        Number of patches (sub-images) to consider in each axis to perform
        the registrations. If None, consider the whole image
    mode: str, optional
        Padding mode among {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}.
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order: int, optional
        Order associated to the interpolation in the warp function

    Returns
    -------
    img_reg: numpy.ndarray((m,n))
        Resulting registered image
    tform: PiecewiseAffineTransform object
        Transformation applied for the image registration
    """
    if nb_blocks is None or nb_blocks == (1, 1):
        tform = AffineTransform(matrix=tmats[0])

    else:
        shape = (img.shape[0] // nb_blocks[0], img.shape[1] // nb_blocks[1])
        x0 = shape[1] * (0.5 + np.arange(nb_blocks[1]))
        y0 = shape[0] * (0.5 + np.arange(nb_blocks[0]))

        src = []
        dst = []
        k = count()
        for i, y in enumerate(y0):
            for j, x in enumerate(x0):
                tmat = tmats[next(k)]

                def src_dst(x_in, y_in):
                    src.append([x_in, y_in])
                    dst.append(list((tmat @ np.array([x_in, y_in, 1]).T)[:2]))

                src_dst(x, y)

                # shifts extension for Delaunay triangulation to all pixels
                if j == 0:
                    src_dst(x - shape[1], y)

                if j == nb_blocks[1] - 1:
                    src_dst(x + shape[1], y)

                if i == 0:
                    src_dst(x, y - shape[0])

                if i == nb_blocks[0] - 1:
                    src_dst(x, y + shape[0])

        src = np.asarray(src)
        dst = np.asarray(dst)

        # registration
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)

    img_reg = warp(img, tform,
                   mode=mode, cval=cval, preserve_range=True, order=order)

    return img_reg


def constant_drift_removal(tmats, drift, is_cumulative=True):
    """
    Remove a constant drift in the (cumulative) transformation matrices

    Parameters
    ----------
    tmats: numpy.ndarray((nslices * nb_blocks, 3, 3))
        Transformations matrices calculated during the registration processing
    drift: sequence of 2 floats
        Constant translation values (transl_x, transl_y) to remove at each slice
    is_cumulative: bool, optional
        Key to consider 'tmats' as cumulative transformation matrices

    Returns
    -------
    tmats_new: numpy.ndarray((nslices * nb_blocks, 3, 3))
        Transformations matrices with the constant drift removed
    """
    assert (len(drift) == 2)

    if drift == (0, 0):
        return tmats

    tmats_new = tmats.copy()
    if is_cumulative:
        for k, tmat in enumerate(tmats_new):
            tmat[:, 0, 2] -= k * drift[0]  # transl_x
            tmat[:, 1, 2] -= k * drift[1]  # transl_y
    else:
        for k, tmat in enumerate(tmats_new):
            tmat[:, 0, 2] -= drift[0]  # transl_x
            tmat[:, 1, 2] -= drift[1]  # transl_y

    return tmats_new


def running_avg_removal(tmats, box_size=20):
    """
    Remove a running average in the transformation matrices

    Parameters
    ----------
    tmats: numpy.ndarray((nslices * nb_blocks, 3, 3))
        Transformations matrices calculated during the registration processing
    box_size: int, optional
        Size of the window for the running average

    Returns
    -------
    tmats_new: numpy.ndarray((nslices * nb_blocks, 3, 3))
        Transformations matrices with the running average removed
    """
    if box_size in [0, 1]:
        return tmats

    identity = np.identity(3)
    tmats_new = tmats + identity - uniform_filter1d(tmats, box_size, axis=0)
    return tmats_new


def inner_rectangle(shape, tmats_cumul, nb_blocks=None):
    """
    Calculate indices related to the maximum inner rectangle resulting of the
    successive 'tmats_cumul' affine transformations.

    Parameters
    ----------
    shape: tuple of 2 ints (m, n)
        Shape of the images/slices
    tmats_cumul: numpy.ndarray((nslices * p * q, 3, 3))
        Cumulative tansformations matrices previously calculated during
        registration flow steps
    nb_blocks: tuple of 2 ints (p, q), optional
        Number of patches (sub-images) used during the registration processing

    Returns
    -------
    (imin, imax, jmin, jmax): tuple of 4 ints
        Indices associated to the largest rectangle resulting from tmats_cumul
    img_reg: numpy.ndarray((m, n))
        Image resulting from the tmats_cumul application.
        Area with 0 values correspond to the region affected by the registration
    """
    img = np.ones(shape)
    img_reg = np.ones(shape)
    for tmat in tmats_cumul:
        mask = img_transformation(img, tmat, nb_blocks=nb_blocks)
        img_reg[np.isnan(mask)] = 0
    imin, imax, jmin, jmax = find_max_inner_rectangle(img_reg, value=1)

    return (imin, imax, jmin, jmax), img_reg


def find_max_inner_rectangle(arr, value=0):
    """
    Returns coordinates of the largest rectangle containing the 'value'.
    From : https://stackoverflow.com/questions/2478447

    Parameters
    ----------
    arr: numpy.ndarray((m, n), dtype=int)
        2D array to work with
    value: int, optional
        Reference value associated to the area of the largest rectangle

    Returns
    -------
    imin, imax, jmin, jmax: ints
        indices associated to the largest rectangle
    """
    Info = namedtuple('Info', 'start height')

    def rect_max_size(histogram):
        stack = []
        top = lambda: stack[-1]
        max_size = (0, 0, 0)  # height, width and start position of the max rect
        pos = 0  # current position in the histogram
        for pos, height in enumerate(histogram):
            start = pos  # position where rectangle starts
            while True:
                if not stack or height > top().height:
                    stack.append(Info(start, height))  # push
                elif stack and height < top().height:
                    tmp = (top().height, pos - top().start, top().start)
                    max_size = max(max_size, tmp, key=area)
                    start, _ = stack.pop()
                    continue
                break  # height == top().height goes here

        pos += 1
        for start, height in stack:
            max_size = max(max_size, (height, (pos - start), start), key=area)

        return max_size

    def area(size):
        return size[0] * size[1]

    iterator = iter(arr)
    hist = [(el == value) for el in next(iterator, [])]
    max_rect = rect_max_size(hist) + (0,)
    for irow, row in enumerate(iterator):
        hist = [(1 + h) if el == value else 0 for h, el in zip(hist, row)]
        max_rect = max(max_rect, rect_max_size(hist) + (irow + 1,), key=area)

    imax = int(max_rect[3] + 1)
    imin = int(imax - max_rect[0])
    jmin = int(max_rect[2])
    jmax = int(jmin + max_rect[1])

    return imin, imax, jmin, jmax
