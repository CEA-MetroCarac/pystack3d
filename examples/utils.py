"""
Utilities functions related to the examples execution
"""
import os
from pathlib import Path
import shutil
from tempfile import gettempdir
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tifffile import imread

ASSETS = Path(__file__).parents[1] / 'assets'


class UserTempDirectory:
    """ Class to call user temp via the 'with' statement """

    def __enter__(self):
        return gettempdir()

    def __exit__(self, exc, value, tb):
        pass


def init_dir(dirpath, case, copy_params_toml=True):
    """ Initialize the project directory """
    dirname = Path(dirpath) / f'pystack3d_{case}'
    os.makedirs(dirname, exist_ok=True)

    # delete .tif to avoid bad surprises when working with user temp
    for fname in dirname.glob("*.tif"):
        os.remove(fname)
    dirname_target = dirname / 'target'
    if os.path.exists(dirname_target):
        shutil.rmtree(dirname_target)

    if copy_params_toml:
        src = ASSETS / 'toml' / f'params_{case}_stack.toml'
        dst = dirname / 'params.toml'
        shutil.copy(src, dst)

    return dirname


def postpro(process_steps, input_dir, channel, verbosity=False):
    """ Return 'dirnames', 'labels' and 'stats' related to each process step """
    if not isinstance(process_steps, list):
        process_steps = [process_steps]

    input_dir = Path(input_dir)
    dirnames = [input_dir / channel]
    labels = ['input']
    for process_step in process_steps:
        if process_step != 'registration_calculation':
            dirnames.append(input_dir / 'process' / process_step / channel)
            if process_step == 'registration_transformation':
                labels.append('registration')
            else:
                labels.append(process_step)

    fname = Path(dirnames[-1]) / 'outputs' / 'stats.npy'
    stats = np.load(fname)
    stats_vals = []
    stats_labels = ['min', 'max', 'mean']
    for j, pfx in zip([0, 2], ['input', 'output']):
        for k in range(3):
            values = stats[:, j, k]
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            stats_vals.append([vmin, vmax])
            if verbosity:
                print(f"{pfx}-{stats_labels[k]}: {vmin} {vmax}")
    stats_vals = np.asarray(stats_vals)

    return dirnames, labels, stats_vals


def plot_cube_faces(arr, ax, levels=40,
                    show_colorbar=True, vmin=None, vmax=None):
    """
    External faces representation of a 3D array with matplotlib

    Parameters
    ----------
    arr: numpy.ndarray()
        3D array to handle
    ax: Axes3D object
        Axis to work with
    levels: int or array-like, optional
        Determines the number and positions of the contour lines / regions
    show_colorbar: bool, optional
        Activation key for colorbar displaying
    vmin, vmax: floats, optional
        Data range that the colormap covers
    """
    if vmin is None:
        vmin = arr.min()

    if vmax is None:
        vmax = arr.max()

    shape = arr.shape

    Y, X, Z = np.meshgrid(np.arange(shape[0]),
                          np.arange(shape[1]),
                          -np.arange(shape[2]), indexing='ij')

    kwargs = {'vmin': vmin, 'vmax': vmax,
              'levels': np.linspace(arr.min(), arr.max(), levels)}

    # Plot contour surfaces
    ax.contourf(X[:, :, 0], Y[:, :, 0], arr[:, :, 0],
                zdir='z', offset=0, **kwargs)
    ax.contourf(X[0, :, :], arr[0, :, :], Z[0, :, :],
                zdir='y', offset=0, **kwargs)
    ax.contourf(arr[:, -1, :], Y[:, -1, :], Z[:, -1, :],
                zdir='x', offset=X.max(), **kwargs)

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    ax.axis('off')

    # Plot edges
    edges_kw = dict(color='k', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set zoom and angle view
    ax.view_init(30, -45, 0)
    ax.set_box_aspect([shape[1], shape[0], shape[2]])

    if show_colorbar:
        norm = mpl.colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        scmap = plt.cm.ScalarMappable(norm=norm)
        scmap.set_array([])
        plt.colorbar(scmap)


def plot_results(dirnames, labels, vmin=0, vmax=255,
                 inds_cutplanes=None, inds_crop=None):
    """ Plot results related to the different process steps """

    ncol = len(dirnames)
    fig, ax = plt.subplots(2, ncol, sharex='col', figsize=(2.5 * ncol, 5))
    fig.tight_layout(pad=2.5)
    plt.rcParams["image.origin"] = 'lower'
    kwargs = {'vmin': vmin, 'vmax': vmax, 'aspect': 'equal'}

    for i, (dirname, label) in enumerate(zip(dirnames, labels)):
        fnames = sorted(dirname.glob('*.tif'))
        if len(fnames) != 0:
            arr = imread(fnames)
            if inds_cutplanes is None:
                kc, ic, _ = tuple(int(x / 2) for x in arr.shape)
            else:
                kc, ic = inds_cutplanes
            if label == 'target' and inds_crop is not None:
                imin, imax, jmin, jmax = inds_crop
                arr = arr[:, imin:imax, jmin:jmax]

            ax[0, i].set_title(label)
            ax[0, i].imshow(np.flipud(arr[kc, :, :]), **kwargs)
            ax[1, i].imshow(arr[:, ic, :], **kwargs)

            if i == 0:
                ax[0, 0].set_ylabel("Y")
                ax[0, 0].set_xlabel("X")
                add_cutplane_line(ax[0, 0], ic, arr.shape[2], arr.shape[1])
                ax[1, 0].set_ylabel("Z")
                ax[1, 0].set_xlabel("X")
                add_cutplane_line(ax[1, 0], kc, arr.shape[2], -arr.shape[0])

            # # CROPPING
            # from matplotlib.patches import Rectangle
            # rect = Rectangle((5, 2), 40, 40, fc='none', ec='r', ls='--', lw=2)
            # ax[0, 0].add_patch(rect)
            # ax[1, 0].axvline(x=5, c="r", ls="--", lw=2)
            # ax[1, 0].axvline(x=45, c="r", ls="--", lw=2)


def add_cutplane_line(ax, ind, width, height):
    """ Add cutplane representation with related arrows """
    for position_rel in [0.05, 0.95]:
        xy = (position_rel * width, ind)
        xytext = (position_rel * width, ind + 0.1 * height)
        ax.axhline(y=ind, ls='--', lw='0.5', color='w')
        ax.annotate("", xy=xy, xytext=xytext,
                    arrowprops={"arrowstyle": '->', "color": 'w'})
