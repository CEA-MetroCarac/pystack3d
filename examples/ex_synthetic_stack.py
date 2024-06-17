"""
Examples with a synthetic stack
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from tifffile import imwrite, imread
from skimage.draw import line
from pyvsnr.utils import curtains_addition

import pystack3d
from pystack3d import Stack3d

from utils import init_dir, postpro, plot_results, plot_cube_faces
from utils import UserTempDirectory

POLICY = "slice_{slice_nb}_z={z_coord}um.tif"


def ex_synthetic_stack(process_steps=None, dirfunc=None, nproc=None,
                       verbosity=True, show_pbar=True, show_plots=True):
    """ Example with synthetic data """

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # project directory creation with input data
        dir_proj = init_dir(dirpath, case='synthetic')
        synthetic_images_generation(dir_proj, process_steps=process_steps)

        # processing
        stack = Stack3d(input_name=dir_proj)
        stack.eval(process_steps=process_steps, nproc=nproc,
                   show_pbar=show_pbar)

        # post-processing
        if process_steps is None:
            process_steps = stack.params['process_steps']
        channel = stack.params['channels'][0]
        dirnames, labels, stats = postpro(process_steps, dir_proj, channel,
                                          verbosity=verbosity)

        # add targeted stack
        dirnames.append(dir_proj / 'target')
        labels.append('target')

        # cropping indices related to the registration transformation
        dir_registration = dir_proj / 'process' / 'registration_transformation'
        fname = dir_registration / 'outputs' / 'inds_crop.txt'
        inds_crop = None
        if 'registration_transformation' in process_steps:
            inds_crop = np.loadtxt(fname).astype(int)

        for dirname, label in zip(dirnames, labels):
            print(dirname)
            fnames = sorted(dirname.glob('*.tif'))
            arr = imread(fnames)
            arr = np.swapaxes(arr, 0, 2).T
            arr = np.swapaxes(arr, 1, 2)
            if label == 'target' and inds_crop is not None:
                imin, imax, jmin, jmax = inds_crop
                arr = arr[:, jmin:jmax, imin:imax]

            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(label)
            plot_cube_faces(arr, ax, show_colorbar=True, vmin=0, vmax=255)

        if show_plots:
            plot_results(dirnames, labels,
                         vmin=0, vmax=255,
                         inds_cutplanes=(83, 150),
                         inds_crop=inds_crop)
            plt.savefig(f"{labels[-2]}.png")
            plt.show()

        return stats[3:]


def synthetic_images_generation(dirpath, process_steps=None):
    """ Generate stack with defects to be removed by the workflow """

    from skimage.data import binary_blobs
    from scipy.ndimage import gaussian_filter

    if process_steps is None:
        process_steps = pystack3d.PROCESS_STEPS

    length = 300
    sigma = 2
    extra_band = 20  # external blobs not calculated impacting the vol. fraction
    arr = binary_blobs(length=length + 2 * extra_band,
                       n_dim=3,
                       blob_size_fraction=0.07,
                       volume_fraction=0.5,
                       rng=3).astype(float)
    arr = gaussian_filter(arr, sigma=sigma, mode='mirror').clip(0, 1)
    sl_intra = slice(extra_band, length + extra_band)
    arr = arr[sl_intra, sl_intra, sl_intra]
    arr *= 0.5

    arr_target = arr.copy()
    dirname = Path(dirpath) / "target"
    os.makedirs(dirname, exist_ok=True)
    save_tiff(arr_target, length, dirname)

    # add shift
    if "registration_calculation" in process_steps:
        for k in range(20, 70):
            shift_x = 10 * (k % 20 > 10)
            arr[k] = shift(arr[k], shift=[shift_x, 0], order=1, cval=0)

    # change contrast
    if "intensity_rescaling" in process_steps:
        sl0 = slice(75, 78)
        sl1 = slice(80, 85)
        sl2 = slice(90, 93)
        sl3 = slice(95, 100)
        arr[sl0, :, :] = arr[sl0, :, :] * 0.7
        arr[sl1, :, :] = arr[sl1, :, :] * 0.4
        arr[sl2, :, :] = arr[sl2, :, :] * 0.6
        arr[sl3, :, :] = arr[sl3, :, :] * 0.5

    # delete slices
    inds_resampling = []
    if "resampling" in process_steps:
        inds = np.arange(100, arr.shape[0])
        inds_resampling = [ind for ind in inds if ind % 2]
        arr = np.delete(arr, inds_resampling, axis=0)

    # add a background
    if "bkg_removal" in process_steps:
        z0 = np.linspace(0, 1, arr.shape[0])
        x0 = np.linspace(0, 1, arr.shape[1])
        y0 = np.linspace(0, 1, arr.shape[2])[::-1]
        z, x, y = np.meshgrid(z0, x0, y0, indexing='ij')
        bkg = x * y * z
        bkg *= 0.5 / bkg.max()
        mask = arr > 0.05
        arr[mask] += bkg[mask]

    # add curtains
    if "destriping" in process_steps:
        for k in [0, 83]:
            arr[k] = curtains_addition(arr[k],
                                       seed=0,
                                       amplitude=0.5,
                                       sigma=(2, 80),
                                       angle=80,
                                       threshold=0.9999,
                                       norm=False)

    # add extra area
    if "cropping" in process_steps:
        pad = 40
        arr = np.pad(arr, ((0, 0), (pad, 0), (0, 0)), 'constant')

    save_tiff(arr, length, dirpath, inds_resampling)


def save_tiff(arr, length, dirpath, inds_resampling=None):
    """ Save slices as .tif images """

    if inds_resampling is None:
        inds_resampling = []

    arr *= 255
    arr = arr.astype(np.uint8)

    k_slice = 0
    for k in range(length):
        if k not in inds_resampling:
            img = arr[k_slice, ...].T.copy()
            z = 0.01 * k

            # save image
            slice_nb = f'{k_slice:04d}'
            z_coord = f'{z:.4f}'
            name = POLICY.format(slice_nb=slice_nb, z_coord=z_coord)
            imwrite(Path(dirpath) / name, img)
            k_slice += 1


if __name__ == '__main__':
    DIRFUNC = UserTempDirectory  # use the user temp location
    # DIRFUNC = TemporaryDirectory  # use a TemporaryDirectory
    NPROC = 1
    PROCESS_STEPS = [
        'cropping',
        'bkg_removal',
        'destriping',
        'registration_calculation', 'registration_transformation',
        'intensity_rescaling',
        'resampling',
    ]

    ex_synthetic_stack(PROCESS_STEPS, DIRFUNC, NPROC)
