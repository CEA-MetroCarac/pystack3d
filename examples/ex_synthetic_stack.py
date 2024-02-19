"""
Examples with a synthetic stack
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from tifffile import imwrite
from skimage.draw import line

from examples.utils import init_dir, postpro, plot_results, UserTempDirectory

from pystack3d import Stack3d

POLICY = "slice_{slice_nb}_z={z_coord}um.tif"


def ex_synthetic_stack(process_steps=None, dirfunc=None, nproc=None,
                       serial=True, verbosity=True, show_pbar=True,
                       show_plots=True):
    """ Example with synthetic data """

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # project directory creation with input data
        dirname = init_dir(dirpath, case='synthetic')
        synthetic_images_generation(dirname)

        # processing
        stack = Stack3d(input_name=dirname)
        stack.eval(process_steps=process_steps, nproc=nproc, serial=serial,
                   show_pbar=show_pbar)

        # post-processing
        if process_steps is None:
            process_steps = stack.params['process_steps']
        channel = stack.params['channels'][0]
        dirnames, labels, stats = postpro(process_steps, dirname, channel,
                                          verbosity=verbosity)
        if show_plots:
            plot_results(dirnames, labels, stats.min(), stats.max())
            plt.savefig(f"{labels[-1]}.png")
            plt.show()

        return stats[3:]


def synthetic_images_generation(dirpath, no_def=False):
    """ Generate stack with disturbances to be removed by the workflow """

    # central pattern
    x0 = y0 = z0 = np.linspace(0, 1, 100)
    x, y, z = np.meshgrid(x0, y0, z0)
    arr = np.zeros_like(x)
    mask = (np.abs(x - 0.5) < 0.1) * (np.abs(y - 0.5) < 0.1)
    arr[mask] = 0.5

    # background adapted to the cropped area made after the pattern shifting
    x1 = y1 = np.linspace(0, 1, 50)
    x, y, z = np.meshgrid(x1, y1, z0)
    bkg = 1 + 2 * z + y + x + x * y + x ** 2

    # images generation
    for k in range(arr.shape[2]):
        img = arr[..., k].copy()
        z = 0.01 * k ** 2

        # add stripes
        if 20 <= k <= 80:
            ampli = 0.8 * bkg.max()
            img[line(45, 45, 55, 55)] = ampli
            img[line(62, 45, 72, 55)] = ampli

        # shifting and cropping
        img = shift(img, shift=[0, int(0.1 * z)], order=1, cval=0)
        img = img[25:75, 25:75]

        # add background
        mask = img != 0.
        img[~mask] += bkg[~mask, k]

        # change intensity in some slices
        if (k + 1) % 11:
            img[~mask] += 0.5

        # save image
        slice_nb = f'{k:04d}'
        z_coord = f'{z:.4f}'
        name = POLICY.format(slice_nb=slice_nb, z_coord=z_coord)
        imwrite(Path(dirpath) / name, img)
        if no_def:
            imwrite(Path(dirpath) / 'no_def' / name, arr[25:75, 25:75, k])


def plot_input_synth(dirfunc):
    """ Plot the input 'no def'/'def' synthetic images """
    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        dirname = Path(dirpath) / f'pystack3d_synth'
        os.makedirs(dirname, exist_ok=True)
        os.makedirs(dirname / 'no_def', exist_ok=True)

        synthetic_images_generation(dirname, no_def=True)
        dirnames = [dirname / 'no_def', dirname]
        labels = ['no defect', 'with defects']
        vmin, vmax = 0, 6
        plot_results(dirnames, labels, vmin, vmax)
        plt.savefig("synth_use_case.png")


if __name__ == '__main__':
    # DIRFUNC = UserTempDirectory  # use the user temp location
    DIRFUNC = TemporaryDirectory  # use a TemporaryDirectory
    NPROC = 1
    REGISTRATION = ['registration_calculation', 'registration_transformation']

    # ex_synthetic_stack('cropping', DIRFUNC, NPROC)
    # ex_synthetic_stack('bkg_removal', DIRFUNC, NPROC)
    # ex_synthetic_stack('intensity_rescaling', DIRFUNC, NPROC)
    # ex_synthetic_stack('destriping', DIRFUNC, NPROC)
    # ex_synthetic_stack(REGISTRATION, DIRFUNC, NPROC)
    # ex_synthetic_stack('resampling', DIRFUNC, NPROC)

    # Launch all process steps (independently)
    ex_synthetic_stack(dirfunc=DIRFUNC, nproc=NPROC, serial=False)
