"""
Functions related to the intensity rescaling processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
from scipy.ndimage import uniform_filter1d

from pystack3d.utils import outputs_saving
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def init_args(params, shape):
    """
    Initialize arguments related to the current processing
    ('intensity_rescaling') and return a specific array to share (histo)

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process

    Returns
    -------
    histo: numpy.ndarray((shape[0], nbins))
        Histogram to be shared during the (multi)processing
    """
    histo = np.zeros((shape[0], params['nbins']), dtype=int)
    return histo


def intensity_rescaling(fnames=None, inds_partition=None, queue_incr=None,
                        nbins=256, range_bins=None, filter_size=10,
                        output_dirname=None):
    """
    Function for image intensity rescaling

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    nbins: int, optional
        Number of bins in the histograms
    range_bins: list of 2 floats, optional
        Range used in the histograms.
        If None, do a preliminary loop on all files to extract [vmin, vmax]
    filter_size: int, optional
        Filter size for the histograms filtering operation (transverse 1d
        uniform filter) used as target for the image intensity rescaling.
        Value filter_size = -1 operates a standard transverse averaging
        through all the slices.
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    assert isinstance(nbins, int) and nbins > 0
    assert filter_size > 0 or filter_size == -1

    # bins range evaluation
    if range_bins is None:
        stats = []
        for fname in fnames:
            with TiffFile(fname) as tiff:
                img = tiff.asarray()
            stats.append([[img.min(), img.max(), None],
                          [None, None, None], [None, None, None]])
        kmin, kmax = inds_partition[0], inds_partition[-1]
        collect_shared_array_parts(stats, kmin, kmax, key='stats')
        stats = get_complete_shared_array(key='stats')
        range_bins = [np.nanmin(stats), np.nanmax(stats)]

    # histograms calculation
    histos_orig = []
    for fname in fnames:
        with TiffFile(fname) as tiff:
            img = tiff.asarray()
        hist, edges = np.histogram(img.flatten(), bins=nbins, range=range_bins)
        histos_orig.append(hist)
        queue_incr.put(0.5)

    # Equivalent 2D-histogram
    histos_orig = np.asarray(histos_orig)
    x = (edges[1:] + edges[:-1]) / 2.

    # array sharing and saving between multiproc
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(histos_orig, kmin, kmax)
    histos_orig = get_complete_shared_array()
    if pid_0:
        np.save(output_dirname / 'outputs' / 'histo_orig.npy', histos_orig)

    # transverse axis averaging or filtering
    if filter_size == -1:
        histos_ref = np.mean(histos_orig, axis=0)
        histos_ref = np.vstack([histos_ref] * histos_orig.shape[0])
    else:
        histos_ref = uniform_filter1d(histos_orig, filter_size, axis=0)
    if pid_0:
        np.save(output_dirname / 'outputs' / 'histos_ref.npy', histos_ref)

    # intensity rescaling
    histos_final = []
    stats = []
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()

        cdf_target = np.cumsum(histos_ref[inds_partition[k]])
        cdf_target = cdf_target / np.max(cdf_target)
        img_res = eval(img, cdf_target, x, nbins, range_bins)

        hist, _ = np.histogram(img_res.flatten(), bins=nbins, range=range_bins)
        histos_final.append(hist)

        outputs_saving(output_dirname, fname, img, img_res, stats)

        queue_incr.put(0.5)

    queue_incr.put('finished')

    # arrays sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(histos_final, kmin, kmax)
    histos_final = get_complete_shared_array()
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        np.save(output_dirname / 'outputs' / 'histos_final.npy', histos_final)
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def eval(img, cdf_target, x_target, nbins, range_bins):
    r"""
    Find closest pixel-matches corresponding to the CDF (Cumulative
    Distribution Function) of the input image, given the value of the CDF H
    of the targeted image at the corresponding x_target location, s.t. :
        cdf_target = H(x_target) <=> x_target = H-1(cdf_target)

    https://stackoverflow.com/questions/32655686

    Parameters
    ----------
    img: numpy.ndarray((m, n))
       input image to rescale
    cdf_target: list of 'nbins' floats
       CDF to target
    x_target: list of 'nbins' floats
       support (gray levels) related to cdf_target values
    nbins: int
        Number of bins to consider
    range_bins: list of 2 floats
        Range associated to the bins

    Returns
    -------
    img_rescaled: numpy.ndarray((m, n))
        the corresponding intensity-rescaled image
    """
    # original cdf
    cdf_img = cdf_calculation(img, nbins, range_bins)

    # intensity rescaling
    shape = img.shape
    vmin, vmax = range_bins
    img_norm = (nbins - 1) * (img - vmin) / (vmax - vmin)
    new_pixels = np.interp(cdf_img, cdf_target, x_target)
    img_rescaled = np.reshape(new_pixels[img_norm.ravel().astype(int)], shape)

    return img_rescaled


def cdf_calculation(img, nbins, range_bins):
    """
    Return the CDF (Cumulative Distribution Function) of an image

    Parameters
    ----------
    img: numpy.ndarray((m, n))
        image to handle
    nbins: int
        Number of bins to consider
    range_bins: list of 2 floats
        Range associated to the bins

    Returns
    -------
    img_cdf: list of 'nbins' floats
        The corresponding CDF
    """
    hist, _ = np.histogram(img, nbins, range=range_bins)
    hist[hist == 0] = 1  # trick to force the cfd to be monotonically increasing
    img_cdf = hist.cumsum().astype(float)
    img_cdf /= img_cdf.max()

    return img_cdf


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    if not os.path.exists(output_dirname / 'outputs' / 'histo_orig.npy'):
        return

    histos_orig = np.load(output_dirname / 'outputs' / 'histo_orig.npy')
    histos_ref = np.load(output_dirname / 'outputs' / 'histos_ref.npy')
    histos_final = np.load(output_dirname / 'outputs' / 'histos_final.npy')

    fig = plt.figure(figsize=(12, 4))
    fig.canvas.manager.set_window_title('intensity_rescaling (maps)')
    plt.subplots_adjust(wspace=0.4)
    plt.subplot(131)
    plt.title("Original histograms")
    plt.imshow(histos_orig, origin='lower')
    plt.xlabel('Bins')
    plt.ylabel('# Frames')
    plt.colorbar()
    plt.axis("auto")
    plt.subplot(132)
    plt.title("Reference histograms")
    plt.imshow(histos_ref, origin='lower')
    plt.colorbar()
    plt.axis("auto")
    plt.subplot(133)
    plt.title("Final histograms")
    plt.imshow(histos_final, origin='lower')
    plt.colorbar()
    plt.axis("auto")
    plt.savefig(output_dirname / 'outputs' / 'intensity_rescaling_maps.png')
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    fig.canvas.manager.set_window_title('intensity_rescaling (profiles)')
    plt.subplots_adjust(wspace=0.4)
    plt.subplot(131)
    plt.title("Original histograms")
    plt.plot(histos_orig.T[:])
    plt.xlabel('Bins')
    plt.ylabel('Occurrences')
    plt.subplot(132)
    plt.title("Reference histograms")
    plt.plot(histos_ref.T[:])
    plt.subplot(133)
    plt.title("Final histograms")
    plt.plot(histos_final.T[:])
    plt.savefig(output_dirname / 'outputs' / 'intensity_rescaling_profiles.png')
    plt.close(fig)
