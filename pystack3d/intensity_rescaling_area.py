"""
Functions related to the intensity rescaling processing from averaged values
contained in a area defined by the user
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

from pystack3d.utils import outputs_saving
from pystack3d.utils_mp import send_shared_array, receive_shared_array
from pystack3d.cropping import inds_from_area


def init_args(params, nslices):
    """
    Initialize arguments related to the current processing
    ('intensity_rescaling_area') and return a specific array to share (histo)

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    nslices: int
        Number of the total slices to process

    Returns
    -------
    tmp: numpy.ndarray(nslices)
        shared array (means or rescaling_factors) associated with the intensity
        rescaling processing
    """
    tmp = np.zeros(nslices)
    return tmp


def intensity_rescaling_area(fnames=None, inds_partition=None, queue_incr=None,
                             area=None,
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
         to estimate the rescaling factor
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    # conversion from area coordinates to indices
    imin, imax, jmin, jmax = inds_from_area(area, fnames, pid_0, output_dirname)

    # calculation of the averaged values in the selected area
    means = []
    for fname in fnames:
        img = imread(fname)[imin:imax, jmin:jmax]
        means.append(img.mean())
        queue_incr.put(0.5)

    # array sharing and saving between multiproc
    kmin, kmax = inds_partition[0], inds_partition[-1]
    send_shared_array(means, kmin, kmax)
    means = receive_shared_array()
    if pid_0:
        np.save(output_dirname / 'outputs' / 'means.npy', means)

    mean_ref = means.min()

    stats, rescaling_factors = [], []
    for k, fname in enumerate(fnames):
        img = imread(fname)
        mean = means[k + inds_partition[0]]
        rescaling_factor = mean_ref / mean
        img_res = rescaling_factor * img
        outputs_saving(output_dirname, fname, img, img_res, stats)
        rescaling_factors.append(rescaling_factor)
        queue_incr.put(0.5)

    queue_incr.put('finished')

    # rescaling_factors and stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    send_shared_array(rescaling_factors, kmin, kmax)
    rescaling_factors = receive_shared_array()
    send_shared_array(stats, kmin, kmax, is_stats=True)
    stats = receive_shared_array(is_stats=True)
    if pid_0:
        np.save(output_dirname / 'outputs' / 'factors.npy', rescaling_factors)
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

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('intensity_rescaling_area (factors)')
    ax.plot(factors)
    ax.set_xlabel("# Frames")
    ax.set_ylabel("Rescaling factors")
    plt.savefig(output_dirname / 'outputs' / 'factors.png')
