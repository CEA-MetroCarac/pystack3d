"""
Functions related to the cropping processing
"""
import numpy as np
from tifffile import imread

from pystack3d.utils import outputs_saving
from pystack3d.utils_mp import send_shared_array, receive_shared_array


def cropping(fnames=None, inds_partition=None, queue_incr=None,
             area=None,
             output_dirname=None):
    """
    Function dedicated to the cropping processing

    Parameters
    ----------
    fnames: list os pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    area: iterable of 4 ints, optional
        Cropping area defined from coordinates (xmin, xmax, ymin, ymax)
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    imin, imax, jmin, jmax = inds_from_area(area, fnames, pid_0, output_dirname)

    stats = []
    for fname in fnames:
        img = imread(fname)
        img_res = img[imin:imax, jmin:jmax]
        outputs_saving(output_dirname, fname, img, img_res, stats)
        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    send_shared_array(stats, kmin, kmax, is_stats=True)
    stats = receive_shared_array(is_stats=True)
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def inds_from_area(area, fnames, pid_0, output_dirname):
    """ Return imin, imax, jmin, jmax from 'area' """
    img0 = imread(fnames[0])

    if area is None:
        imin, imax, jmin, jmax = 0, img0.shape[0], 0, img0.shape[1]

    else:
        shape = (len(fnames), img0.shape[0], img0.shape[1])

        jmin, jmax = area[0], area[1]
        imin, imax = shape[1] - area[3], shape[1] - area[2]

        if pid_0:
            msg = '\nERROR!! your area exceed the image shape according to the '
            assert area[1] <= shape[2], msg + 'x-direction'
            assert area[3] <= shape[1], msg + 'y-direction'
            with open(output_dirname / 'outputs' / 'log.txt', 'w') as fid:
                fid.write(f"Original shape: {shape}")
                fid.write(f"New shape: {(shape[0], jmax - jmin, imax - imin)}")
                fid.write(f"Area: {area}")

    return imin, imax, jmin, jmax
