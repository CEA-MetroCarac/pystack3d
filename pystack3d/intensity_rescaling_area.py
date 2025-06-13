"""
Functions related to the intensity rescaling processing from averaged values
contained in a area defined by the user
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile

from pystack3d.cropping import inds_from_area
from pystack3d.utils import mask_creation, outputs_saving
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def init_args(params, shape):
    """
    Initialize arguments related to the current processing
    ('intensity_rescaling_area') and return a specific array to share (histo)

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process

    Returns
    -------
    tmp: numpy.ndarray(shape[0])
        shared array (means or rescaling_factors) associated with the intensity
        rescaling processing
    """
    tmp = np.zeros(shape[0])
    return tmp


def intensity_rescaling_area(fnames=None, inds_partition=None, queue_incr=None,
                             area=None, threshold_min=None, threshold_max=None,
                             factors_range=None,
                             output_dirname=None):
    """
    Function for image intensity rescaling according to the averaged gray
    evolution in an area specified by the user.

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
        Reference area, defined from coordinates (xmin, xmax, ymin, ymax) in px,
         to estimate the rescaling factors
    threshold_min, threshold_max: floats, optional
        Relative thresholds used to exclude low and high values in the
        selected area
    factors_range: iterable of 2 floats, optional
        Authorized range for the rescaling factors. Default is (0.8, 1.2)
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if factors_range is None:
        factors_range = [0.8, 1.2]

    # conversion from area coordinates to indices
    imin, imax, jmin, jmax = inds_from_area(area, fnames, pid_0, output_dirname)

    # calculation of the averaged values in the selected area
    means = []
    for fname in fnames:
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        img_roi = img[imin:imax, jmin:jmax]
        mask = mask_creation(img_roi,
                             threshold_min=threshold_min,
                             threshold_max=threshold_max)
        means.append(img_roi[mask].mean())
        queue_incr.put(0.5)

    # arrays sharing and saving between multiproc
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(means, kmin, kmax)
    means = get_complete_shared_array()
    means_ref = means.median()

    with np.errstate(all='ignore'):
        factors = np.divide(means_ref, means)
    factors = np.clip(factors, factors_range[0], factors_range[1])

    if pid_0:
        np.save(output_dirname / 'outputs' / 'means.npy', means)
        np.save(output_dirname / 'outputs' / 'factors.npy', factors)

    stats = []
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        img_res = factors[k + inds_partition[0]] * img
        outputs_saving(output_dirname, fname, img, img_res, stats)
        queue_incr.put(0.5)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    if not os.path.exists(output_dirname / 'outputs' / 'means.npy'):
        return

    means = np.load(output_dirname / 'outputs' / 'means.npy')
    factors = np.load(output_dirname / 'outputs' / 'factors.npy')

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('intensity_rescaling_area (means)')
    ax.plot(means)
    ax.set_xlabel("# Frames")
    ax.set_ylabel("Means")
    plt.savefig(output_dirname / 'outputs' / 'means.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('intensity_rescaling_area (factors)')
    ax.plot(factors)
    ax.set_xlabel("# Frames")
    ax.set_ylabel("Rescaling factors")
    plt.savefig(output_dirname / 'outputs' / 'factors.png')
    plt.close(fig)
