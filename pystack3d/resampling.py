"""
Functions related to the resampling processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from parse import Parser
from tifffile import TiffFile

from pystack3d.utils import img_reformatting, save_tif
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def init_args(params, shape):
    """
    Initialize arguments related to the current processing ('resampling')

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process

    Returns
    -------
    stats_out: numpy.ndarray((len(zpos_out), 2, 3))
        Statistics related to outputs (resampled frames)
    """
    fnames = params['fnames']
    policy = params['policy']
    zpos_in = extract_z_from_filenames(fnames, policy)

    if 'zpos_out' in params:
        zpos_out = params['zpos_out']
    elif 'dz' in params:
        zpos_out = np.arange(min(zpos_in), max(zpos_in), params['dz'])
    else:
        raise IOError("Neither 'zpos_out' nor 'dz' have been defined")

    mask = (zpos_out >= min(zpos_in)) * (zpos_out <= max(zpos_in))
    zpos_out = zpos_out[mask]

    # to ensure monotonic behavior (strictly increasing)
    zpos_out = np.unique(np.maximum.accumulate(zpos_out))

    # kwargs updating
    params.update({'zpos_in': zpos_in, 'zpos_out': zpos_out})
    if 'dz' in params:
        params.pop('dz')

    stats_out = np.zeros((len(zpos_out), 2, 3))
    return stats_out


def resampling(fnames=None, inds_partition=None, queue_incr=None,
               policy=None, zpos_in=None, zpos_out=None,
               output_dirname=None):
    """
    Function for image resampling in z (slices) direction

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    policy: str, optional
        Policy to consider for .tif images with {slice_nb} and {z_coord} format
        parameters. Example : policy = 'slice_{slice_nb}_z={z_coord}um.tif'
    zpos_in: np.array(n), optional
        Original slice positions in z-direction
    zpos_out: np.array(m), optional
        Slice positions of the final interpolated resampled image
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if pid_0:
        with open(output_dirname / 'outputs' / 'log.txt', 'w') as fid:
            fid.write(f"Original z-shape: {len(zpos_in)}")
            fid.write(f"New z-shape: {len(zpos_out)}")

    stats, stats_out, z_out = [], [], []
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img_k = tiff.asarray()
        stats.append([[img_k.min(), img_k.max(), img_k.mean()],
                      [None, None, None], [None, None, None]])

        z_k = zpos_in[inds_partition[k]]

        if k == 0:
            img_km1 = img_k
            z_km1 = z_k

        if z_k < z_km1:
            continue

        if k == 0 and pid_0:
            z_interp = zpos_out[(zpos_out >= z_km1) * (zpos_out <= z_k)]
        else:
            z_interp = zpos_out[(zpos_out > z_km1) * (zpos_out <= z_k)]

        if z_k - z_km1 == 0.:
            slope = 0.
        else:
            delta = img_k.astype(float) - img_km1.astype(float)
            slope = delta / (z_k - z_km1)

        for z_int in z_interp:
            img_int = img_km1 + (z_int - z_km1) * slope
            img_int2 = img_reformatting(img_int, img_k.dtype)
            stats_out.append([[img_int.min(), img_int.max(), img_int.mean()],
                              [img_int2.min(), img_int2.max(), img_int2.mean()]
                              ])
            z_out.append(z_int)

            # resampled image saving
            slice_nb = f'{list(zpos_out).index(z_int):04d}'
            z_coord = f'{z_int:.4f}'
            name = policy.format(slice_nb=slice_nb, z_coord=z_coord)
            save_tif(img_int2, fname, output_dirname / name)

        img_km1 = img_k
        z_km1 = z_k

        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')

    if len(z_out) > 0:
        kmin = list(zpos_out).index(z_out[0])
        kmax = list(zpos_out).index(z_out[-1])
        collect_shared_array_parts(stats_out, kmin, kmax)
    stats_out = get_complete_shared_array()

    # extend and fuse stats arrays
    stats_in = stats[:, 0, :]
    stats_out0 = stats_out[:, 0, :]
    stats_out1 = stats_out[:, 1, :]
    delta = len(stats_out) - len(stats_in)
    stats_ext = np.full((np.abs(delta), 3), np.nan)
    if delta > 0:
        stats_in = np.vstack((stats_in, stats_ext))
    else:
        stats_out0 = np.vstack((stats_out0, stats_ext))
        stats_out1 = np.vstack((stats_out1, stats_ext))
    stats = np.swapaxes(np.dstack((stats_in, stats_out0, stats_out1)), 1, 2)

    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)
        np.save(output_dirname / 'outputs' / 'zpos_in.npy', zpos_in)
        np.save(output_dirname / 'outputs' / 'zpos_out.npy', zpos_out)


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    if not os.path.exists(output_dirname / 'outputs' / 'zpos_in.npy'):
        return

    zpos_in = np.load(output_dirname / 'outputs' / 'zpos_in.npy')
    zpos_out = np.load(output_dirname / 'outputs' / 'zpos_out.npy')
    zpos_in_incr = np.diff(zpos_in)
    zpos_out_incr = np.diff(zpos_out)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('resampling (z-positions)')
    ax.set_title("Z-positions")
    ax.plot(zpos_in, 'o-', label='in')
    ax.plot(zpos_out, 'o-', label='out')
    ax.set_xlabel("# Frames")
    ax.set_ylabel("Z")
    ax.legend()
    plt.savefig(output_dirname / 'outputs' / 'z_positions.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('resampling (z-increments)')
    ax.set_title("Z-increments")
    ax.plot(zpos_in_incr, 'o-', label='in')
    ax.plot(zpos_out_incr, 'o-', label='out')
    ax.set_xlabel("# Frames")
    ax.set_ylabel("dz")
    ax.legend()
    plt.savefig(output_dirname / 'outputs' / 'z_increments.png')
    plt.close(fig)


def extract_z_from_filenames(fnames, policy, verbosity=False):
    """
    Return z coordinates from the filenames parsing

    Parameters
    ----------
    fnames: list of str
        List of images filenames to handle
    policy: str
        Policy to consider for .tif images with {slice_nb} and {z_coord} format
        parameters. Example : policy = 'slice_{slice_nb}_z={z_coord}um.tif'
        parameters
    verbosity: bool, optional
        Activation key for verbosity

    Returns
    -------
    z_coords: list of floats
        List of z coordinates
    """
    parser = Parser(policy)

    z_coords = []
    for k, fname in enumerate(fnames):
        _, name = os.path.split(fname)
        res = parser.parse(os.path.split(name)[1])
        slice_nb = res.named['slice_nb']
        z_coord = res.named['z_coord']
        z_coords.append(float(z_coord))
        if verbosity:
            print(f"{k} - slice={slice_nb} - z={z_coord}")

    return z_coords
