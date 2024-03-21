"""
Figures generation for the documentation or articles
"""

import os
from pathlib import Path
from tempfile import gettempdir
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

from examples.utils import postpro, plot_results, plot_cube_faces


def init_synthetic():
    """ Initialize 'dirnames, labels, inds_crop' from the synthetic example """

    dir_proj = Path(gettempdir()) / 'pystack3d_synthetic'
    channel = '.'
    process_steps = [
        'cropping',
        'bkg_removal',
        'destriping',
        'registration_calculation', 'registration_transformation',
        'intensity_rescaling',
        'resampling',
    ]

    dirnames, labels, _ = postpro(process_steps, dir_proj, channel)
    dirnames.append(dir_proj / 'target')
    labels.append('target')

    dir_registration = dir_proj / 'process' / 'registration_transformation'
    fname = dir_registration / 'outputs' / 'inds_crop.txt'
    inds_crop = None
    if os.path.exists(fname):
        inds_crop = np.loadtxt(fname).astype(int)

    return dirnames, labels, inds_crop


def gen_2d_synthetic():
    """ Generate 2D cut-planes visualizations """

    dirnames, labels, inds_crop = init_synthetic()

    if len(dirnames) > 4:
        dirnames_ = [dirnames[:4], dirnames[4:]]
        labels_ = [labels[:4], labels[4:]]
        titles = ['workflow_1', 'workflow_2']

        for dirnames, labels, title in zip(dirnames_, labels_, titles):
            plot_results(dirnames, labels, 0, 255,
                         inds_cutplanes=(83, 150),
                         inds_crop=inds_crop)
            plt.tight_layout()
            plt.savefig(f"{title}.png")

    else:
        plot_results(dirnames, labels, 0, 255,
                     inds_cutplanes=(83, 150),
                     inds_crop=inds_crop)
        plt.tight_layout()
        plt.savefig(f"{labels[-2]}.png")


def gen_3d_synthetic():
    """ Generate 3D visualizations """

    dirnames, labels, inds_crop = init_synthetic()

    inds = [0, -2, -1]
    dirnames_ = [dirnames[i] for i in inds]
    labels_ = [labels[i] for i in inds]

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(labels_)

    for dirname, label in zip(dirnames_, labels_):
        print(dirname)
        fnames = sorted(dirname.glob('*.tif'))
        arr = imread(fnames)
        arr = np.swapaxes(arr, 0, 2).T
        arr = np.swapaxes(arr, 1, 2)
        if label == 'target' and inds_crop is not None:
            imin, imax, jmin, jmax = inds_crop
            arr = arr[:, jmin:jmax, imin:imax]
        plot_cube_faces(arr, ax, show_colorbar=True, vmin=0, vmax=255)


def gen_3d_aligned_synthetic():
    """ Generate 3D visualizations """

    dirnames, labels, inds_crop = init_synthetic()

    fig = plt.figure(figsize=(7, 3))
    ax = [fig.add_subplot(131, projection='3d'),
          fig.add_subplot(132, projection='3d'),
          fig.add_subplot(133, projection='3d'),
          ]

    for i, (dirname, label) in enumerate(zip(dirnames, labels)):
        print(dirname)
        fnames = sorted(dirname.glob('*.tif'))
        arr = imread(fnames)
        arr = np.swapaxes(arr, 0, 2).T
        arr = np.swapaxes(arr, 1, 2)
        if label == 'target' and inds_crop is not None:
            imin, imax, jmin, jmax = inds_crop
            arr = arr[:, jmin:jmax, imin:imax]
        ax[i].set_title(label)
        plot_cube_faces(arr, ax[i], show_colorbar=False, vmin=0, vmax=255)

    fig.tight_layout()
    plt.savefig(f'{labels[-2]}_3D.png')


gen_2d_synthetic()
# gen_3d_synthetic()
# gen_3d_aligned_synthetic()
plt.show()
