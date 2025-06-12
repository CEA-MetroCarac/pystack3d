"""
Functions related to the cropping processing
"""
import warnings
import numpy as np
from tifffile import TiffFile

from pystack3d.utils import outputs_saving
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def cropping(fnames=None, inds_partition=None, queue_incr=None,
             area=None,
             output_dirname=None):
    """
    Function dedicated to the cropping processing

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
        Cropping area defined from coordinates (xmin, xmax, ymin, ymax)
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    imin, imax, jmin, jmax = inds_from_area(area, fnames, pid_0, output_dirname)

    stats = []
    for fname in fnames:
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        img_res = img[imin:imax, jmin:jmax]
        outputs_saving(output_dirname, fname, img, img_res, stats)
        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def inds_from_area(area, fnames, pid_0, output_dirname):
    """ Return imin, imax, jmin, jmax from 'area' """
    with TiffFile(fnames[0]) as tiff:
        img0 = tiff.asarray()

    if area is None:
        imin, imax, jmin, jmax = 0, img0.shape[0], 0, img0.shape[1]

    else:
        shape = (len(fnames), img0.shape[0], img0.shape[1])

        jmin, jmax = area[0], area[1]
        imin, imax = shape[1] - area[3], shape[1] - area[2]

        if pid_0:
            msg = 'your area ({}) exceed the image shape ({}) according to the {}-direction'
            if area[1] > shape[2]:
                warnings.warn(msg.format(area[1], shape[2], 'x'), category=UserWarning)
            if area[3] > shape[1]:
                warnings.warn(msg.format(area[3], shape[1], 'y'), category=UserWarning)
            with open(output_dirname / 'outputs' / 'log.txt', 'w') as fid:
                fid.write(f"Original shape: {shape}")
                fid.write(f"New shape: {(shape[0], jmax - jmin, imax - imin)}")
                fid.write(f"Area: {area}")

    return imin, imax, jmin, jmax
