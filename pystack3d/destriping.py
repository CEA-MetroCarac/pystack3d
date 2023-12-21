"""
Functions related to the destriping processing
"""
import numpy as np
from tifffile import imread
from pyvsnr import vsnr2d

from pystack3d.utils import outputs_saving
from pystack3d.utils_mp import send_shared_array, receive_shared_array


def destriping(fnames=None, inds_partition=None, queue_incr=None,
               maxit=100, filters=None,
               output_dirname=None):
    """
    Function dedicated to destriping from vsnr 'cupy' or 'cuda' algorithm

    Parameters
    ----------
    fnames: list os pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    maxit: int, optional
        Number of maximum iterations used by the VSNR algorithm
    filters: list of dict
        Filters defined as a list of dictionaries that specifies for each filter
         the corresponding 'name', 'noise_level', 'sigma' and 'theta'.
        See the pyvsnr documentation for more details.
        Example:
            filters = [{'name': "Gabor", 'noise_level': 10, 'sigma': (1, 20),
                        'theta': -10},
                       {'name': "Gabor", 'noise_level': 40, 'sigma': (3, 40),
                       'theta': -10}]
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    stats = []
    for fname in fnames:
        img = imread(fname)

        img_res = vsnr2d(img, filters, maxit)

        outputs_saving(output_dirname, fname, img, img_res, stats)

        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    send_shared_array(stats, kmin, kmax, is_stats=True)
    stats = receive_shared_array(is_stats=True)
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)
