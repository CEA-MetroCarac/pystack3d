"""
utilities functions related to multiprocessing
"""
from importlib import import_module
from multiprocessing import Barrier
from multiprocessing.sharedctypes import Array
import numpy as np
from tifffile import TiffFile


def shared_array_init(arr):
    """ Return components related to a multiprocessing.sharedctypes.Array
        created from 'arr' """
    arr_shared = Array(np.ctypeslib.as_ctypes_type(arr.dtype), arr.size)
    arr_shared[:] = arr.ravel()
    return arr_shared, arr.shape, arr.dtype


def worker_init(queue_incr, barrier,
                shared_stats, shape_stats, dtype_stats,
                shared_array, shape_array, dtype_array,
                shared_array2, shape_array2, dtype_array2):
    """ Initialize the workers """
    global QUEUE_INCR, BARRIER  # pylint:disable=W0601
    global SHARED_STATS_PTR, SHARED_STATS  # pylint:disable=W0601
    global SHARED_ARRAY_PTR, SHARED_ARRAY  # pylint:disable=W0601
    global SHARED_ARRAY2_PTR, SHARED_ARRAY2  # pylint:disable=W0601

    QUEUE_INCR = queue_incr
    BARRIER = barrier
    SHARED_STATS_PTR = shared_stats
    SHARED_STATS = np_from_array(shared_stats, shape_stats, dtype_stats)
    SHARED_ARRAY_PTR = shared_array
    SHARED_ARRAY = np_from_array(shared_array, shape_array, dtype_array)
    SHARED_ARRAY2_PTR = shared_array2
    SHARED_ARRAY2 = np_from_array(shared_array2, shape_array2, dtype_array2)


def np_from_array(shared, shape, dtype):
    """ Return array from shared object in buffer """
    if shared is None:
        return None
    else:
        return np.frombuffer(shared.get_obj(), dtype=dtype).reshape(shape)


def step_wrapper(process_step, kwargs):
    """
    Wrapper function to pass kwargs in the case of multiprocessing.
    Must be defined as a 'free' function so all independent workers can have
    access to it

    Parameters
    ----------
    process_step: str
        name of the process step (example 'cropping')
    kwargs: dict
        dictionary arguments passed to the process step
    """
    kwargs.update({'queue_incr': QUEUE_INCR})

    if process_step == 'cropping_final':
        process_step = 'cropping'

    module = import_module(f"pystack3d.{process_step}")
    func = getattr(module, process_step)
    return func(**kwargs)


def initialize_args(process_step, params, nproc, shape):
    """ parameters for shared array in multiprocessing """

    # statistics (min, max, mean) shared array
    arr_stats = np.zeros((shape[0], 3, 3), dtype=float)
    shared_stats, shape_stats, dtype_stats = shared_array_init(arr_stats)

    # specific 'process_step' shared arrays
    shared_array, shape_array, dtype_array = (None, None, None)
    shared_array2, shape_array2, dtype_array2 = (None, None, None)
    if process_step == 'cropping_final':
        process_step = 'cropping'
    module = import_module(f"pystack3d.{process_step}")
    if "init_args" in dir(module):
        func = getattr(module, "init_args")
        obj = func(params, shape)
        if isinstance(obj, np.ndarray):
            arr = obj
            shared_array, shape_array, dtype_array = shared_array_init(arr)
        if isinstance(obj, tuple):
            arr, arr2 = obj
            shared_array, shape_array, dtype_array = shared_array_init(arr)
            shared_array2, shape_array2, dtype_array2 = shared_array_init(arr2)

    args = (Barrier(nproc),
            shared_stats, shape_stats, dtype_stats,
            shared_array, shape_array, dtype_array,
            shared_array2, shape_array2, dtype_array2)

    return args


def collect_shared_array_parts(arr, kmin=0, kmax=0, key='array'):
    """ Collect the array partition sent by workers """
    if kmax == 0:
        kmax = arr.size

    if key == 'stats':
        with SHARED_STATS_PTR.get_lock():
            SHARED_STATS[kmin:kmax + 1] = arr
    elif key == 'array':
        with SHARED_ARRAY_PTR.get_lock():
            SHARED_ARRAY[kmin:kmax + 1] = arr
    elif key == 'array2':
        with SHARED_ARRAY2_PTR.get_lock():
            SHARED_ARRAY2[kmin:kmax + 1] = arr
    else:
        raise IOError


def get_complete_shared_array(key='array'):
    """ Return the shared array resulting from the arrays collect """
    BARRIER.wait()
    if key == 'stats':
        return SHARED_STATS
    elif key == 'array':
        return SHARED_ARRAY
    elif key == 'array2':
        return SHARED_ARRAY2
    else:
        raise IOError
