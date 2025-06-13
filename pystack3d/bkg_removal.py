"""
Functions related to the background removal processing
"""
import os
import itertools
import functools
import operator
import glob
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from skimage.transform import resize
from tifffile import TiffFile
from PIL import Image, ImageDraw

from pystack3d.utils import (imread_3d_skipping, skipping, division,
                             outputs_saving, mask_creation)
from pystack3d.utils_multiprocessing import (collect_shared_array_parts,
                                             get_complete_shared_array)

WEIGHTS = [None, "HuberT", "Hampel"]


def init_args(params, shape):
    """
    Initialize arguments related to the current processing ('bkg_removal') and
    return a specific array to share (coefs)

    Parameters
    ----------
    params: dict
        Dictionary related to the current process.
        See the related documentation for more details.
    shape: tuple of 3 int
        Shape of the stack to process

    Returns
    -------
    coefs: numpy.ndarray((shape[0], len(powers)))
        Coefficients of the polynom to be shared during the (multi)processing
    """
    if 'poly_basis' in params.keys() and params['poly_basis']:
        powers = powers_from_expr(params['poly_basis'], params['dim'],
                                  force_cst_term=True)

    elif 'orders' in params.keys():
        if 'cross_terms' not in params.keys():
            raise IOError("'cross_terms' has to be defined with 'orders'")
        powers = powers_from_orders(params['orders'], params['cross_terms'])

    else:
        raise IOError("'poly_basis' or 'orders' should be defined")

    if 'threshold_min' in params.keys() and 'threshold_max' in params.keys():
        if params['threshold_min'] and params['threshold_max']:
            msg = "threshold_min must be strictly less than threshold_max"
            assert params['threshold_min'] < params['threshold_max'], msg

    if 'weight_func' in params.keys():
        msg = "'weight_func'] should be 'HuberT', 'Hampel', or 'None'"
        assert params['weight_func'] in ['HuberT', 'Hampel', 'None'], msg
        if params['weight_func'] == 'None':
            params['weight_func'] = None

    if 'preserve_avg' in params.keys():
        if params['preserve_avg']:
            msg = "'preserve_avg' is only suitable for '3D' background eval."
            assert len(powers[0]) == 3, msg

    # kwargs updating
    params.update({'powers': powers})
    for key in ['poly_basis', 'orders', 'cross_terms', 'dim']:
        if key in params:
            params.pop(key)

    coefs = np.zeros((shape[0], len(powers)), dtype=float)
    powers_2d = get_powers_2d(powers)
    poly_basis = np.zeros((shape[1] * shape[2], len(powers_2d)), dtype=float)

    return tuple((coefs, poly_basis))


def bkg_removal(fnames=None, inds_partition=None, queue_incr=None,
                powers=None, skip_factors=(10, 10), skip_averaging=False,
                threshold_min=None, threshold_max=None, weight_func="HuberT",
                preserve_avg=False, output_dirname=None):
    """
    Function dedicated to the background removal processing

    Parameters
    ----------
    fnames: list of pathlib.Path, optional
        List of '.tif' filenames to process
    inds_partition: list of ints, optional
        List of indexes to be considered by the global var SHARED_ARRAY when
        working in multiprocessing
    queue_incr: multiprocessing.Queue, optional
        Queue passed to the function to interact with the progress bar
    powers: list of q-lists of 2 or 3 ints, optional
        Powers related to each term of the polynomial basis (in invlex order).
        Example: 1 + y**2 + x + x**2 * y -> [[0, 0], [0, 2], [1, 0], [2, 1]]
    skip_factors: tuple of 2 or 3 ints, optional
        Skipping factors in each direction to perform polynomial fitting on
        a reduced image
    threshold_min, threshold_max: floats, optional
        Relative thresholds used to exclude low and high values in the
        background removal processing
    skip_averaging: bool, optional
        Activation key for averaging when skip_factors is used
    weight_func: str, optional
        Weight function ('HuberT' or 'Hampel') to consider by the M-estimator.
        For more details see : https://www.statsmodels.org/stable/rlm.html
        If None, polynomial fitting is calculated according to the classical
        'leastsq' algorithm when solving : A.X = B
    preserve_avg: bool, optional
        Activation key to preserve the average between the input and the output
        (work only with background evaluated from '3D' coefficients)
    output_dirname: str, optional
        Directory pathname for process results saving
    """
    pid_0 = inds_partition[0] == 0  # first thread

    msg = "'powers' items and 'skip_factors' should have the same length"
    assert len(powers[0]) == len(skip_factors), msg
    msg = "'powers' should have term related to the constants"
    assert [0, 0] in powers or [0, 0, 0] in powers, msg

    # directory for background .tif images saving
    bkg_dirname = output_dirname / 'outputs' / 'bkgs'
    os.makedirs(bkg_dirname, exist_ok=True)

    with TiffFile(fnames[0]) as tiff:
        shape = tiff.pages[0].shape

    # 3D Background coefficients calculation on a down sampled 3D stack
    if len(powers[0]) == 3:

        fnames_full = sorted(glob.glob(os.path.join(fnames[0].parent, "*.tif")))
        nslices = len(fnames_full)
        shape_3d = (shape[0], shape[1], nslices)
        powers_3d = powers

        if pid_0:  # calculation on the 1rst thread only
            arr_3d = imread_3d_skipping(fnames_full, skip_factors=skip_factors)
            arr_3d = np.swapaxes(arr_3d, 0, 1)
            arr_3d = np.swapaxes(arr_3d, 1, 2)

            poly_basis_skip = poly_basis_calculation(shape_3d, powers_3d,
                                                     skip_factors=skip_factors)

            _, _, coefs_3d = bkg_eval(arr_3d, powers_3d,
                                      skip_factors=None,
                                      threshold_min=threshold_min,
                                      threshold_max=threshold_max,
                                      skip_averaging=skip_averaging,
                                      weight_func=weight_func,
                                      poly_basis_precalc=poly_basis_skip,
                                      preserve_avg=preserve_avg)

            np.savetxt(output_dirname / 'outputs' / 'coefs_3d.txt', coefs_3d)

            coefs_3d = np.tile(coefs_3d, nslices).reshape(nslices, len(powers))
            collect_shared_array_parts(coefs_3d)

        coefs_3d = get_complete_shared_array()[0]
        kwargs_3d = {'shape_3d': shape_3d, 'coefs_3d': coefs_3d}
    else:
        kwargs_3d = None

    # polynomial basis pre-calculation
    powers_2d = get_powers_2d(powers)
    args = (shape[:2], powers_2d, skip_factors[:2])
    poly_basis_skip = poly_basis_calculation(*args)
    if pid_0:
        poly_basis = poly_basis_calculation(*args[:-1])
        collect_shared_array_parts(poly_basis, key='array2')
    poly_basis = get_complete_shared_array(key='array2')

    # background calculation and removing
    stats, coefs = [], []
    for k, fname in enumerate(fnames):
        with TiffFile(fname) as tiff:
            img = tiff.asarray()

        if kwargs_3d is not None:
            kwargs_3d['index'] = inds_partition[k]

        img_res, bkg, coef = bkg_eval(img, powers,
                                      skip_factors=skip_factors,
                                      poly_basis_precalc=poly_basis,
                                      poly_basis_precalc_skip=poly_basis_skip,
                                      threshold_min=threshold_min,
                                      threshold_max=threshold_max,
                                      skip_averaging=skip_averaging,
                                      weight_func=weight_func,
                                      kwargs_3d=kwargs_3d,
                                      preserve_avg=preserve_avg)
        coefs.append(coef)

        outputs_saving(output_dirname, fname, img, img_res, stats)
        bkg_saving(bkg_dirname, fname, bkg)

        queue_incr.put(1)

    queue_incr.put('finished')

    # arrays sharing and saving
    kmin, kmax = inds_partition[0], inds_partition[-1]
    collect_shared_array_parts(coefs, kmin, kmax)
    coefs = get_complete_shared_array()
    collect_shared_array_parts(stats, kmin, kmax, key='stats')
    stats = get_complete_shared_array(key='stats')
    if pid_0:
        dirname = output_dirname / 'outputs'
        np.save(dirname / 'stats.npy', stats)
        np.savez(dirname / 'coefs.npz', coefs=coefs, powers=powers)


def bkg_saving(bkg_dirname, fname, bkg):
    """ Save bkg image in .png format with vmin, vmax values displayed """
    vmin, vmax = bkg.min(), bkg.max()
    # image size reduction
    shape = bkg.shape
    skip_factor = max(min(shape[0] // 100, shape[1] // 100), 1)
    bkg_red = bkg[::skip_factor, ::skip_factor]
    # image rescaling into [0, 255]
    if vmax == vmin:
        bkg_res = np.zeros_like(bkg_red)
    else:
        bkg_res = (255 / (vmax - vmin)) * (bkg_red - vmin)
    img = Image.fromarray(bkg_res.astype(np.uint8)).convert('RGB')
    img1 = ImageDraw.Draw(img, 'RGB')
    img1.text((1, 1), f"vmin={vmin:.1f}\nvmax={vmax:.1f}", fill=(255, 0, 0))
    img.save((bkg_dirname / fname.name).with_suffix('.png'), origin='lower')


def get_powers_2d(powers):
    """ Return the 2D powers related to the 'powers' 1rst and 2nd indices """
    return list(dict.fromkeys([(x[0], x[1]) for x in powers]))


def powers_from_orders(orders, cross_terms):
    """
    Return 'powers' from polynomial 'orders'

    Parameters
    ----------
    orders: tuple of 2 or 3 ints
        Polynomial orders in x, y and potentially z directions respectively
    cross_terms: bool, optional
        Activation key to take into account cross terms in the polynom

    Returns
    -------
    powers: list of q-lists of 2 or 3 ints
        Powers related to each term of the polynomial basis (in invlex order).
        Example:
        orders=(2, 1), cross_terms=False : basis = 1 + y + x + x**2
                -> powers = [[0, 0], [0, 1], [1, 0], [2, 0]]
        orders=[2, 1], cross_terms=True : 1 + y + x + x*y + x**2 + x**2*y
                -> powers = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    """
    powers = []
    inds = [range(order + 1) for order in orders]
    for indices in list(itertools.product(*inds)):
        if cross_terms or indices.count(0) >= (len(inds) - 1):
            powers.append(list(indices))
    return powers


def powers_from_expr(expr, dim=3, vars=None, force_cst_term=False):
    """
    Return 'powers' from expression (in invlex order)

    Examples
    --------
    >> expr = "y + x + x*y + x**2 + x**2*y"
    >> print(powers_from_expr(expr))
    >> [[0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    >> print(powers_from_expr(expr, force_cst_term=True))
    >> [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    """
    assert dim in [2, 3]
    if vars is None:
        vars = ['x', 'y', 'z'][:dim]
    assert len(vars) == dim

    if force_cst_term:
        expr += '+ 1'

    expr = expr.replace(" ", "")
    terms = expr.split('+')

    powers_list = []
    for term in terms:
        term = term.replace("**", "^")
        powers = [0] * len(vars)
        for k, var in enumerate(vars):
            terms2 = term.split(var)
            if len(terms2) > 1:
                powers[k] = 1
                for term2 in terms2:
                    if len(term2) > 0 and term2[0] == '^':
                        powers[k] = int(term2[1])
        if powers not in powers_list:
            powers_list.append(powers)
    return sorted(powers_list)


def expr_from_powers(powers, variables=['x', 'y', 'z']):
    """
    Return expression from 'powers'

    Examples
    --------
    >> powers = [[0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    >> print(expr_from_powers(powers, variables=['x', 'y']))
    >> "y + x + x*y + x**2 + x**2*y"
    """
    dim = len(powers[0])
    vars = variables[:dim]
    expr = ""
    for power in powers:
        expr += " + "
        if not any(power):
            expr += "1"
        else:
            term = [f'{var}**{i}' for i, var in zip(power, vars) if i != 0]
            expr += "*".join(term)
    expr = expr.replace("**1", "")
    return expr[3:]


def bkg_eval(arr, powers,
             skip_factors=(10, 10),
             threshold_min=None,
             threshold_max=None,
             skip_averaging=False,
             weight_func="HuberT",
             poly_basis_precalc=None,
             poly_basis_precalc_skip=None,
             kwargs_3d=None,
             preserve_avg=False):
    """
    Background removal function dedicated to stack3d

    Parameters
    ----------
    arr: np.ndarray((m, n)) or np.ndarray((m, n, p))
        2D or 3D array
    powers: list of q-lists of 2 or 3 ints
        Powers related to each term of the polynomial basis (in invlex order).
        Example: 1 + y**2 + x + x**2 * y -> [[0, 0], [0, 2], [1, 0], [2, 1]]
    skip_factors: tuple of 2 or 3 ints, optional
        Skipping factors in each direction to perform polynomial fitting on
        a reduced image
    threshold_min, threshold_max: floats, optional
        Thresholds used to exclude low and high values in the background removal
         processing
    skip_averaging: bool, optional
        Activation key for averaging when skip_factors is used.
    weight_func: str, optional
        Weight function ('HuberT' or 'Hampel') to consider by the M-estimator.
        For more details see : https://www.statsmodels.org/stable/rlm.html
        If None, polynomial fitting is calculated according to the classical
        'leastsq' algorithm when solving : A.X = B
    poly_basis_precalc: np.ndarray((m*n, q), optional
        Pre-calculated polynomial basis to significantly decrease CPU time
        when dealing with huge and repetitive task
    poly_basis_precalc_skip: np.ndarray((m//skip_factors[0]*n//skip_factors[1],
    q), optional
        Pre-calculated polynomial basis taking into account the 'skip_factors'
    kwargs_3d: dict, optional
        Dictionary related to 3D background calculation
    preserve_avg: bool, optional
        Activation key to preserve the average between the input and the output
        (work only with background evaluated from '3D' coefficients)

    Returns
    -------
    arr_bkg: numpy.ndarray((m, n)) or numpy.ndarray((m, n, p))
        Final array with the background removed
    bkg: numpy.ndarray((m, n)) or numpy.ndarray((m, n, p))
        array related to the background
    coefs: numpy.ndarray(q)
        Polynomial coefficients resulting from the fit
    """
    arr_bkg = arr.copy().astype(float)
    mask = mask_creation(arr_bkg,
                         threshold_min=threshold_min,
                         threshold_max=threshold_max)
    arr_bkg[~mask] = np.nan

    # background calculation
    if kwargs_3d is not None:
        bkg = bkg_3d_from_slices(k=kwargs_3d['index'],
                                 shape_3d=kwargs_3d['shape_3d'],
                                 coefs_3d=kwargs_3d['coefs_3d'],
                                 powers_3d=powers,
                                 poly_basis_precalc=poly_basis_precalc)

        coefs = kwargs_3d['coefs_3d']

        if not preserve_avg:
            bkg -= coefs[0]  # to not consider the mean value in the bkg
        arr_bkg -= bkg
        arr_bkg[~mask] = arr[~mask]

    else:
        bkg, coefs, _ = bkg_calculation(arr_bkg,
                                        powers=powers,
                                        skip_factors=skip_factors,
                                        skip_averaging=skip_averaging,
                                        weight_func=weight_func,
                                        poly_basis_precalc=poly_basis_precalc,
                                        poly_basis_precalc_skip=poly_basis_precalc_skip)

        bkg -= coefs[0]  # to not consider the mean value in the bkg
        arr_bkg -= bkg
        arr_bkg[~mask] = arr[~mask]

        if preserve_avg:
            coefs[0] = arr_bkg[mask].mean() - arr[mask].mean()

    return arr_bkg, bkg, coefs


def poly_basis_calculation(shape, powers, skip_factors=None):
    """
    Return the polynomial basis related to 'powers' and associated to a ND-array

    Parameters
    ----------
    shape: tuple of 2 (m,n) or 3 (m,n,p) ints
        shape of the 2D or 3D array
    powers: list of q-lists of 2 or 3 ints
        Powers related to each term of the polynomial basis (in invlex order).
        Example: 1 + y**2 + x + x**2 * y -> [[0, 0], [0, 2], [1, 0], [2, 1]]
    skip_factors: tuple of 2 or 3 ints, optional
        Skipping factors in each direction to perform polynomial fitting on a
        reduced image

    Returns
    -------
    poly_basis: numpy.ndarray((m*n, q) or numpy.ndarray((m*n*p, q)
        The related polynomial basis
    """
    dim = len(shape)

    if skip_factors is None:
        skip_factors = (1,) * dim

    assert (len(powers[0]) == dim), "'powers' terms have not the correct dim."
    assert (len(skip_factors) == dim), "'skip_factors' has not the correct dim."

    # axis permutation to be conform to natural (x, y) axis
    if dim > 1:
        shape = list(shape)
        shape[0], shape[1] = shape[1], shape[0]

    # uniform grid coordinates associated to the array
    coords_1d = []
    for npoints, skip_factor in zip(shape, skip_factors):
        coords_1d.append(np.linspace(0., 1., npoints)[::skip_factor])
    coords = np.meshgrid(*coords_1d)

    # kernels calculation
    kernels = []
    for indices in powers:
        members = []
        for coord, power in zip(coords, indices):
            members.append(coord ** power)
        kernels.append(functools.reduce(operator.mul, members, 1).ravel())
        # equivalent in 2D to : kernels.append((x ** i * y ** j).ravel())

    poly_basis = np.asarray(kernels).T

    return poly_basis


def bkg_calculation(arr, powers,
                    skip_factors=None, skip_averaging=False,
                    weight_func="HuberT",
                    poly_basis_precalc=None, poly_basis_precalc_skip=None):
    """
    Evaluate background of a ND-array from polynomial fitting.


    Parameters
    ----------
    arr: np.ndarray((m, n)) or np.ndarray((m, n, p))
        2D or 3D array
    powers: list of q-lists of 2 or 3 ints
        Powers related to each term of the polynomial basis (in invlex order).
        Example: 1 + y**2 + x + x**2 * y -> [[0, 0], [0, 2], [1, 0], [2, 1]]
    skip_factors: tuple of 2 or 3 ints, optional
        Skipping factors in each direction to perform polynomial fitting on a
        reduced image
    skip_averaging: bool, optional
        Activation key for averaging when skip_factors is used.
    weight_func: str, optional
        Weight function ('HuberT', 'Hampel',...) to consider by the M-estimator.
        For more details see : https://www.statsmodels.org/stable/rlm.html
        If None, polynomial fitting is calculated according to the classical
        'leastsq' algorithm when solving : A.X = B
    poly_basis_precalc: np.ndarray((m*n, q)), optional
        Pre-calculated polynomial basis to significantly decrease CPU time
        when dealing with huge and repetitive task
    poly_basis_precalc_skip: np.ndarray((m//skip_factors[0]*n//skip_factors[1],
    q), optional
        Pre-calculated polynomial basis taking into account the 'skip_factors'

    Returns
    -------
    bkg: numpy.ndarray((m, n)) or numpy.ndarray((m, n, p))
        array related to the background
    coefs: numpy.ndarray(q)
        Polynomial coefficients resulting from the fit where q = len(powers)
    poly_basis: numpy.ndarray((m*n, q) or numpy.ndarray((m*n*p, q)
        The related polynomial basis
    """
    shape = arr.shape
    dim = len(shape)
    assert (weight_func in WEIGHTS), f"'weight_func' has to be in {WEIGHTS}"

    # convert arr to float
    dtype_orig = arr.dtype
    arr = arr.astype(float)

    if poly_basis_precalc is None:
        poly_basis = poly_basis_calculation(shape, powers)
    else:
        poly_basis = poly_basis_precalc

    if skip_factors is not None:
        assert (len(skip_factors) == dim), "'arr' and 'skip_factors' dim differ"

        if skip_averaging:
            new_shape = (np.array(shape) / np.array(skip_factors)).astype('int')
            arr_red = resize(arr, new_shape, preserve_range=True, mode='edge')
        else:
            arr_red = skipping(arr, skip_factors)

        if poly_basis_precalc_skip is None:
            poly_basis_skip = poly_basis_calculation(shape, powers,
                                                     skip_factors=skip_factors)
        else:
            poly_basis_skip = poly_basis_precalc_skip

        _, coefs, _ = bkg_calculation(arr_red, powers,
                                      weight_func=weight_func,
                                      poly_basis_precalc=poly_basis_skip)
    else:

        # array flattening and np.nan values exclusion
        arr = arr.flatten()
        mask = np.isnan([arr])[0]
        arr_clean = arr[~mask]

        # check arr_clean has valid values for coefs calculation
        if arr_clean.size == 0 or arr_clean.min() == arr_clean.max():
            bkg = np.zeros(shape)
            coefs = np.zeros(poly_basis.shape[1])
            return bkg, coefs, poly_basis

        kernels = []
        for i in range(poly_basis.shape[1]):
            kernel = poly_basis[:, i]
            kernels.append(kernel[~mask])
        poly_basis_clean = np.asarray(kernels).T

        # coefficients calculation
        if weight_func is None:
            coefs = np.linalg.lstsq(poly_basis_clean, arr_clean, rcond=None)[0]
        else:
            wfun = eval(f"sm.robust.norms.{weight_func}()")
            rlm_model = sm.RLM(arr_clean, poly_basis_clean, M=wfun)
            rlm_results = rlm_model.fit()
            coefs = rlm_results.params

    bkg = np.dot(poly_basis, coefs).reshape(shape)

    # convert bkg to the original array dtype
    bkg = bkg.astype(dtype_orig)

    return bkg, coefs, poly_basis


def bkg_3d_from_slices(k, shape_3d, coefs_3d, powers_3d,
                       poly_basis_precalc=None):
    """
    Background calculation on a (2D) slice from 3D coefficients previously
    calculated

    Parameters
    ----------
    k: int
        Index of the slice to work with (related to shape_3d[2] axis)
    shape_3d: tuple of 3 ints (m, n, p)
        Shape associated to the 3D Array on which the 3D-Background coefficients
        has been previuosly calculated
    coefs_3d: List of q-floats
        3D-Background pre-calculated coefficients where q = len(powers)
    powers_3d: iterable of q-iterables of 3 ints
        Powers associated to each term of the polynom (in invlex order).
        Example: 1 + x*z + x**2 * y -> [[0, 0, 0], [1, 0, 1], [2, 1, 0]]
    poly_basis_precalc: np.ndarray((m*n, q), optional
        Pre-calculated polynomial basis taking into account the 'skip_factors'

    Returns
    -------
    bkg : numpy.ndarray((m, n))
        3D background estimated on the slice
    """
    assert (len(shape_3d) == 3)

    # z-position calculation
    nslices_tot = shape_3d[2]
    zpos = float(k) / float(nslices_tot - 1)

    # poly_basis_2d definition
    shape_2d = shape_3d[:2]
    powers_2d = list(dict.fromkeys([(x[0], x[1]) for x in powers_3d]))
    if poly_basis_precalc is None:
        poly_basis_2d = poly_basis_calculation(shape_2d, powers_2d)
    else:
        poly_basis_2d = poly_basis_precalc

    # correspondence between (ind_i, ind_j) and k in poly_basis_2d[..., k]
    corr_inds = {}
    compt = 0
    for indices in powers_2d:
        corr_inds[f"{indices[0]}-{indices[1]}"] = compt
        compt += 1

    # 3d-background calculation according to z-position and poly_basis_2d
    bkg = np.zeros(shape_2d).ravel()
    compt = 0
    for indices in powers_3d:
        k = corr_inds[f"{indices[0]}-{indices[1]}"]
        bkg += zpos ** indices[2] * coefs_3d[compt] * poly_basis_2d[..., k]
        compt += 1
    bkg = bkg.reshape(shape_2d)

    return bkg


def plot(output_dirname):
    """ Plot the specific data related to the current process """

    fname = output_dirname / 'outputs' / 'coefs.npz'

    if not os.path.exists(fname):
        return

    npzfile = np.load(fname)
    coefs = npzfile['coefs'].T
    powers = npzfile['powers']

    nrows, ncols = division(len(powers))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols)
    fig.canvas.manager.set_window_title('background removal (coef.)')
    plt.suptitle(f"polynomial basis:\n{expr_from_powers(powers)}")
    for k, (coef, power) in enumerate(zip(coefs, powers)):
        plt.subplot(nrows, ncols, k + 1)
        plt.plot(coef, label=expr_from_powers([power]))
        plt.legend(loc=1)
        if k // ncols != nrows - 1:
            plt.xticks([])
            plt.xlabel('')
        else:
            plt.xlabel('# Frames')
    plt.savefig(output_dirname / 'outputs' / 'coefs.png')
    plt.close(fig)
