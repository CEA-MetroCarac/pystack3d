"""
Functions related to the registration calculation processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pystackreg import StackReg
from tifffile import TiffFile
from skimage.util.shape import view_as_blocks
from skimage.transform import AffineTransform

from pystack3d.cropping import inds_from_area
from pystack3d.utils import cumdot, division
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def init_args(params, shape):
    """
    Initialize arguments related to the current processing
    ('registration_calculation') and return a specific array to share (tmats)

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process

    Returns
    -------
    tmats: numpy.ndarray((shape[0], nblocks[0]*nblocks[1], 3, 3))
        Transformation matrices to be shared during the (multi)processing
    """
    global TRANSFORMATION, NB_BLOCKS  # pylint:disable=W0601

    TRANSFORMATION = params['transformation']
    NB_BLOCKS = params['nb_blocks']

    shape = (shape[0], NB_BLOCKS[0] * NB_BLOCKS[1], 3, 3)
    tmats = np.zeros(shape, dtype=float)
    return tmats


def registration_calculation(fnames=None, inds_partition=None, queue_incr=None,
                             area=None, threshold=None,
                             nb_blocks=None, transformation='TRANSLATION',
                             output_dirname=None):
    """
    Function dedicated to the transformation matrices calculation used in the
    image registration

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    area: iterable of 4 ints, optional
        Area defined from coordinates (xmin, xmax, ymin, ymax) to calculate
        the transformation matrices. If None, consider the whole images.
    threshold: float, optional
        Threshold value used to perform registration on binarized images
    nb_blocks: tuple of 2 ints (p, q), optional
        Number of patches (sub-images) to consider in each axis to perform
        the registrations. If None, consider the whole image
    transformation: str, optional
        Transformation matrix type to consider for the registration. To choose
        among :  ['TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE']
        (See pystackreg documentation for more details).
        If None, the default Transformation matrix type is 'TRANSLATION'
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if nb_blocks is None:
        nb_blocks = (1, 1)

    transf_list = ['TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE']
    msg = f"{transformation} not in {transf_list}"
    assert (transformation in transf_list), msg

    # indices related to area
    imin, imax, jmin, jmax = inds_from_area(area, fnames, pid_0, output_dirname)

    # transformation matrices calculation
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        img = img[imin: imax, jmin: jmax]
        if threshold is not None:
            img = img <= threshold
        if k == 0:
            tmats = [np.array([np.identity(3)] * nb_blocks[0] * nb_blocks[1])]
        else:
            tmats.append(tmat_calculation(img_ref, img,
                                          nb_blocks=nb_blocks,
                                          transformation=transformation))
        img_ref = img
        queue_incr.put(1)

    queue_incr.put('finished')

    # array sharing between multiproc
    k0 = 0 if pid_0 else 1  # to manage overlaid frames
    kmin, kmax = inds_partition[k0], inds_partition[-1]
    collect_shared_array_parts(tmats[k0:], kmin, kmax)
    tmats = get_complete_shared_array()
    if pid_0:
        np.save(output_dirname / 'tmats.npy', tmats)


def tmat_calculation(ref, img, nb_blocks=None, transformation='TRANSLATION'):
    """
    Transformation matrix calculation with StackReg from pystackreg package
    applied on patches (sub-images)

    Parameters
    ----------
    ref: numpy.ndarray((m,n))
        Image used as reference
    img: numpy.ndarray((m,n))
        Image considered as 'moving' image
    nb_blocks: tuple of 2 ints (p, q), optional
        Number of patches (sub-images) to consider in each axis to perform
        the registrations. If None, consider the whole image
    transformation: str, optional
        Transformation matrix type to consider for the registration. To choose
        among :  ['TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE']
        (See pystackreg documentation for more details).
        If None, the default Transformation matrix type is 'TRANSLATION'

    Returns
    -------
    tmats: numpy.ndarray((p*q, 3*3))
        Transformation matrix calculated in all patches
    """
    if nb_blocks is None:
        nb_blocks = (1, 1)

    streg = StackReg(eval(f"StackReg.{transformation}"))

    shape = (img.shape[0] // nb_blocks[0], img.shape[1] // nb_blocks[1])
    imax = nb_blocks[0] * shape[0]
    jmax = nb_blocks[1] * shape[1]

    img_blocks = view_as_blocks(img[:imax, :jmax], block_shape=shape)
    ref_blocks = view_as_blocks(ref[:imax, :jmax], block_shape=shape)

    tmats = []
    for i in range(nb_blocks[0]):
        for j in range(nb_blocks[1]):
            ref_bl = ref_blocks[i, j, ...]
            img_bl = img_blocks[i, j, ...]

            tmat = streg.register(ref_bl, img_bl)
            tmats.append(tmat)

    tmats = np.asarray(tmats)

    return tmats


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    if not os.path.exists(output_dirname / 'tmats.npy'):
        return

    tmats = np.load(output_dirname / 'tmats.npy')

    fig = registration_plot(tmats, nb_blocks=NB_BLOCKS, transformation=TRANSFORMATION)
    plt.savefig(output_dirname / 'outputs' / 'tmats_evol.png')
    plt.close(fig)

    fig = registration_plot(tmats, nb_blocks=NB_BLOCKS, transformation=TRANSFORMATION, cumul=True)
    plt.savefig(output_dirname / 'outputs' / 'tmats_cumul_evol.png')
    plt.close(fig)


def registration_plot(tmats, nb_blocks=None, transformation='RIGID_BODY',
                      cumul=False, title='Registration'):
    """
    Plot the evolution of the transformation matrix components issued from the
    registration processing

    Parameters
    ----------
    tmats: sequence of nslices * nb_blocks * numpy.ndarray((3, 3))
        Transformations matrices calculated during the registration processing
    nb_blocks: tuple of 2 ints (p, q)
        Number of patches (sub-images) used during the registration processing
    transformation: str, optional
        Transformation matrix type considered for the plotting.
        If None, consider a 'RIGID_BODY' transformation type.
    cumul: bool, optional
        Activation key to plot the cumulative transformations matrices evolution
    title: str, optional
        Title of the figure

    Returns
    -------
    fig: matplotlib.figure
    """
    transf_list = ['TRANSLATION', 'RIGID_BODY', 'SCALED_ROTATION', 'AFFINE']
    assert (transformation in transf_list)

    if nb_blocks is None:
        nb_blocks = division(len(tmats[0]))
    nbl = nb_blocks[0] * nb_blocks[1]

    if cumul:
        tmats = cumdot(tmats)
        title += " (cumul.)"

    translations = []
    rotations = []
    scales = []
    shears = []
    for tmats_slice in tmats:
        for tmat in tmats_slice:
            tr_x, tr_y, rot, scale_x, scale_y, shear = decomposition(tmat)
            translations.append([tr_x, tr_y])
            rotations.append([rot])
            scales.append([scale_x, scale_y])
            shears.append([shear])
    translations = np.vstack(translations)
    rotations = np.vstack(rotations)
    scales = np.vstack(scales)
    shears = np.vstack(shears)

    transf_index = transf_list.index(transformation)

    fig = plt.figure(figsize=(9, 6))
    fig.canvas.manager.set_window_title('registration_calculation')
    plt.suptitle(title)
    for i in range(nbl):
        transl_i = translations[i::nbl]
        rot_i = rotations[i::nbl]
        scale_i = scales[i::nbl]
        shear_i = shears[i::nbl]
        plt.subplot(nb_blocks[0], nb_blocks[1], i + 1)
        ax1 = plt.gca()
        ax1.plot(transl_i[:, 0], label="transl_x")
        ax1.plot(transl_i[:, 1], label="transl_y")
        if transf_index > 0:
            ax2 = ax1.twinx()
            ax2.plot(np.rad2deg(rot_i[:]), '--', label="rotation (°)")
        if transf_index > 1:
            ax2.plot(scale_i[:, 0] - 1., '--', label="scale_x - 1")
            ax2.plot(scale_i[:, 1] - 1., '--', label="scale_y - 1")
        if transf_index > 2:
            ax2.plot(shear_i[:], '--', label="shear")
        if i == 0:
            ax1.legend(loc=2)
            if transf_index > 0:
                ax2.legend(loc=1)
        ax1.set_xlabel('# Frames')

    return fig


def decomposition(tmat):
    """ Return 'transl_x', 'transl_y', 'rotation', 'scale_x', 'scale_y' and
    'shear' related to a 2D-Affine transformation matrix

    Notes
    -----
    Translation values are given according to "natural" (x, y) coordinates.
    Rotation angle are given according to the counter-clockwise convention.
    """
    aff_tr = AffineTransform(tmat)

    transl_x, transl_y = aff_tr.translation[0], -aff_tr.translation[1]
    rotation = -aff_tr.rotation
    scale_x, scale_y = aff_tr.scale[0], aff_tr.scale[1]
    shear = aff_tr.shear

    return transl_x, transl_y, rotation, scale_x, scale_y, shear
