"""
Examples with real images
"""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from pystack3d import Stack3d
from pystack3d.utils_metadata_fibics import params_from_metadata

from examples.utils import init_dir, postpro, plot_results
from examples.utils import UserTempDirectory  # pylint: disable=unused-import

ASSETS = Path(__file__).parents[1] / 'assets'
DATA = ASSETS / 'stacks' / 'stack_1'


def ex_real_stack(process_steps=None, dirfunc=None, nproc=None,
                  serial=True, verbosity=True, show_pbar=True,
                  show_plots=True):
    """ Example with real data """

    toml_from_metadata = False

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # Initialize the project directory in the temporary folder
        dirname = init_dir(dirpath, case='real',
                           copy_params_toml=not toml_from_metadata)

        # copy the slices in the project directory
        for channel in ["ESB", "SE2"]:
            os.makedirs(dirname / channel, exist_ok=True)
            for src in (DATA / channel).glob('*.tif'):
                shutil.copy(src, dirname / channel)

        if toml_from_metadata:
            shutil.copy(DATA / 'Atlas3D.a3d-setup', dirname)
            fname_toml = ASSETS / 'toml' / 'params_real_stack.toml'
            params_from_metadata(dirname, fname_toml_ref=fname_toml, save=True)

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
