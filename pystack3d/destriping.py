"""
Functions related to the destriping processing
"""
import numpy as np
from tifffile import TiffFile
from pyvsnr import vsnr2d
import pywt

from pystack3d.utils import outputs_saving
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)


def destriping(fnames=None, inds_partition=None, queue_incr=None,
               maxit=100, cvg_threshold=0, filters=None,
               wavelet_decomposition=None,
               output_dirname=None):
    """
    Function dedicated to destriping from the VSNR algorithm (or wavelet
    decomposition).
    For more details, see: https://github.com/CEA-MetroCarac/pyvsnr

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    wavelet_decomposition: dict, optional
        Parameters provided for destriping using wavelet decomposition.
        See `here <https://cea-metrocarac.github.io/pystack3d/api/pystack3d.destriping.html#pystack3d.destriping.destriping_from_wavelets>`_
        Example:
            wavelet_decomposition = {'wavelet':'coif3', 'level':4, 'sigma':4}
    maxit: int, optional
        Number of maximum iterations used by the VSNR algorithm
    cvg_threshold: float, optional
        Convergence criteria to stop the VSNR iterative process, related to the
        maximum residual variation between 2 iterations
    filters: list of dict, optional
        Filters defined as a list of dictionaries that specifies for each filter the corresponding 'name', 'noise_level', 'sigma' and 'theta'.
        See the pyvsnr documentation for more details.
        Example:
            filters = [{'name': "Gabor", 'noise_level': 10, 'sigma': (1, 20), 'theta': -10}, {'name': "Gabor", 'noise_level': 40, 'sigma': (3, 40), 'theta': -10}]
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if wavelet_decomposition is not None:
        wdec = wavelet_decomposition
        wavelet = wdec['wavelet'] if 'wavelet' in wdec else None
        level = wdec['level'] if 'level' in wdec else None
        sigma = wdec['sigma'] if 'sigma' in wdec else None

    stats = []
    for fname in fnames:
        with TiffFile(fname) as tiff:
            img = tiff.asarray()

        if wavelet_decomposition is not None:
            img_res = destriping_from_wavelets(img,
                                               wavelet=wavelet,
                                               level=level,
                                               sigma=sigma)
        else:
            img_res = vsnr2d(img, filters,
                             maxit=maxit,
                             cvg_threshold=cvg_threshold,
                             norm=False)

        outputs_saving(output_dirname, fname, img, img_res, stats)

        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)


def destriping_from_wavelets(img0, wavelet='coif3', level=4, sigma=4):
    """
    Destriping from wavelet decomposition, considering vertical stripes.

    From: https://doi.org/10.1364/OE.17.008567

    Parameters
    ----------
    img0: numpy.ndarray
        Input image to handle.
    wavelet: Wavelet object or name string, or 2-tuple of wavelets, optional
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
        More information in `pywt.wavelist <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#pywt.wavelist>`_
    level: int, optional
        Decomposition level.
    sigma: float, optional
        Standard deviation to be considered in the gaussian filter applied
        during the wavelet decomposition.

    Returns
    -------
    img: numpy.ndarray
        Output image
    """
    # pylint:disable=invalid-name
    coeffs = pywt.wavedec2(img0, wavelet, level=level)
    new_coeffs = [coeffs[0]]

    for i in range(1, len(coeffs)):
        Ch, Cv, Cd = coeffs[i]
        fCv = np.fft.fft(Cv, axis=0)
        fCv2 = np.fft.fftshift(fCv, axes=[0])
        my = fCv2.shape[0]
        x = np.arange(-my // 2, my // 2)
        gauss1d = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
        fCv3 = fCv2 * gauss1d[:, np.newaxis]
        Cv2 = np.fft.ifftshift(fCv3, axes=[0])
        Cv3 = np.fft.ifft(Cv2, axis=0)
        new_coeffs.append((Ch, Cv3.real, Cd))

    img = pywt.waverec2(new_coeffs, wavelet)

    return img
