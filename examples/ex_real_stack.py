"""
Examples with real images
"""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from examples.utils import init_dir, postpro, plot_results, UserTempDirectory

from pystack3d import Stack3d

ASSETS = Path(__file__).parents[1] / 'assets'
DATA = ASSETS / 'stacks' / 'stack_1'


def ex_real_stack(process_steps=None, dirfunc=None, nproc=None,
                  serial=True, verbosity=True, show_pbar=True,
                  show_plots=True):
    """ Example with real data """

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # project directory creation (copy) with input data
        dirname = init_dir(dirpath, case='real')
        for channel in ["ESB", "SE2"]:
            os.makedirs(dirname / channel, exist_ok=True)
            for src in (DATA / channel).glob('*.tif'):
                shutil.copy(src, dirname / channel)

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


if __name__ == '__main__':
    # DIRFUNC = UserTempDirectory  # use the user temp location
    DIRFUNC = TemporaryDirectory  # use a TemporaryDirectory
    NPROC = 1
    REGISTRATION = ['registration_calculation', 'registration_transformation']

    # ex_real_stack('cropping', DIRFUNC, NPROC)
    # ex_real_stack('bkg_removal', DIRFUNC, NPROC)
    # ex_real_stack('intensity_rescaling', DIRFUNC, NPROC)
    # ex_real_stack('destriping', DIRFUNC, NPROC)
    # ex_real_stack(REGISTRATION, DIRFUNC, NPROC)
    # ex_real_stack('resampling', DIRFUNC, NPROC)

    ex_real_stack(['cropping', 'bkg_removal'], DIRFUNC, NPROC)

    # # Launch all process steps
    # ex_real_stack(dirfunc=DIRFUNC, nproc=NPROC)
