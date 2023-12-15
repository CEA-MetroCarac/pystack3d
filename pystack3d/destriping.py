"""
Functions related to the destriping processing
"""
import numpy as np
from tifffile import imread
from pyVSNR import vsnr2d
from common.image.filtering import VSNR

from pystack3d.utils import outputs_saving
from pystack3d.utils_mp import send_shared_array, receive_shared_array


def destriping(fnames=None, inds_partition=None, queue_incr=None,
               vsnr_kwargs=None,
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
    vsnr_kwargs: dict, optional
        Dictionary to manage parameters of the VSNR algorithm.
        Examples (not strictly equivalent in term of results):

        - 'cupy' algorithm :
        vsnr_kwargs = {'algo': "cupy", 'maxit': 100,'cvg_threshold': 1e-4,
                       'nbfilters': 2,
                       'filter_0': {'alpha': 0.4, 'name': "Gabor",
                                    'sigma': (1, 20), 'theta': 350}}
                       'filter_1': {'alpha': 0.4, 'name': "Gabor",
                                    'sigma': (3, 40), 'theta': 350}}
                       }

        - 'cuda' algorithm :
        vsnr_kwargs = {'algo': "cuda", 'maxit': 20,
                       'nbfilters': 2,
                       'filter_0': {'noise_level': 10, 'name': "Gabor",
                                    'sigma': (1, 20), 'theta': 350}}
                       'filter_1': {'noise_level': 40, 'name': "Gabor",
                                    'sigma': (3, 40), 'theta': 350}
                       }
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    if vsnr_kwargs is None:
        raise ValueError("'vsnr_kwargs' has to be given as input")

    algo = vsnr_kwargs['algo']
    assert algo in ['cupy', 'cuda'], "'algo' should be 'cupy' or 'cuda'"

    if algo == 'cupy':
        shape = imread(fnames[0]).shape
        vsnr = VSNR((shape[0], shape[1]))
        for i in range(vsnr_kwargs['nbfilters']):
            filter_i = vsnr_kwargs[f'filter_{i}']
            vsnr.add_filter(alpha=filter_i['alpha'],
                            name=filter_i['name'],
                            sigma=filter_i['sigma'],
                            theta=filter_i['theta'])
        vsnr.initialize()
    else:
        filters = []
        for i in range(vsnr_kwargs['nbfilters']):
            filters.append(vsnr_kwargs[f'filter_{i}'])

    stats = []
    for fname in fnames:
        img = imread(fname)

        vmin, vmax = img.min(), img.max()
        img_norm = (img - vmin) / (vmax - vmin)

        if algo == 'cupy':
            img_res = vsnr.eval(img_norm,
                                maxit=vsnr_kwargs['maxit'],
                                cvg_threshold=vsnr_kwargs['cvg_threshold'])
        else:
            img_res = vsnr2d(img_norm, filters, nite=vsnr_kwargs['maxit'])

        img_res = np.clip(img_res, 0, 1)
        img_res = (img_res - img_res.min()) / (img_res.max() - img_res.min())

        img_res = vmin + img_res * (vmax - vmin)

        outputs_saving(output_dirname, fname, img, img_res, stats)

        queue_incr.put(1)

    queue_incr.put('finished')

    # stats sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    send_shared_array(stats, kmin, kmax, is_stats=True)
    stats = receive_shared_array(is_stats=True)
    if pid_0:
        np.save(output_dirname / 'outputs' / 'stats.npy', stats)
