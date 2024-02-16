"""
Examples with real images
"""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from common import PATH_DATA_METRO_CARAC

from pystack3d import Stack3d
import utils

DIRNAME_REAL = Path(PATH_DATA_METRO_CARAC) / 'Stack3d' / 'SOFC' / 'examples'


def ex_pystack3d_real(process_steps=None, dirfunc=None, nproc=None,
                      serial=True, verbosity=True, show_pbar=True,
                      show_plots=True):
    """ Example with real data """

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # project directory creation with input data
        dirname = utils.init_dir(dirpath, case='real')
        for channel in ["ESB", "SE2"]:
            os.makedirs(dirname / channel, exist_ok=True)
            for src in (DIRNAME_REAL / channel).glob('*.tif'):
                shutil.copy(src, dirname / channel)

        # processing
        stack = Stack3d(input_name=dirname)
        stack.eval(process_steps=process_steps, nproc=nproc, serial=serial,
                   show_pbar=show_pbar)

        # post-processing
        if process_steps is None:
            process_steps = stack.params['process_steps']
        channel = stack.params['channels'][0]
        dirnames, labels, stats = utils.postpro(process_steps, dirname, channel,
                                                verbosity=verbosity)
        if show_plots:
            utils.plot_results(dirnames, labels, stats.min(), stats.max())
            plt.savefig(f"{labels[-1]}.png")
            plt.show()

        return stats[3:]


if __name__ == '__main__':
    # DIRFUNC = utils.UserTempDirectory  # use user temp location
    DIRFUNC = TemporaryDirectory  # use TemporaryDirectory
    NPROC = 1
    REGISTRATION = ['registration_calculation', 'registration_transformation']

    # ex_pystack3d_real('cropping', DIRFUNC, NPROC)
    # ex_pystack3d_real('bkg_removal', DIRFUNC, NPROC)
    # ex_pystack3d_real('intensity_rescaling', DIRFUNC, NPROC)
    # ex_pystack3d_real('destriping', DIRFUNC, NPROC)
    # ex_pystack3d_real(REGISTRATION, DIRFUNC, NPROC)
    # ex_pystack3d_real('resampling', DIRFUNC, NPROC)

    ex_pystack3d_real(['cropping', 'bkg_removal'], DIRFUNC, NPROC)

    # # Launch all process steps
    # ex_pystack3d_real(dirfunc=DIRFUNC, nproc=NPROC)
