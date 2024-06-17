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

from utils import init_dir, postpro, plot_results
from utils import UserTempDirectory  # pylint: disable=unused-import

ASSETS = Path(__file__).parents[1] / 'assets'
DATA = ASSETS / 'stacks' / 'stack_1'


def ex_real_stack(process_steps=None, dirfunc=None, nproc=None,
                  verbosity=True, show_pbar=True, show_plots=True):
    """ Example with real data """

    toml_from_metadata = False

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # Initialize the project directory in the temporary folder
        dir_proj = init_dir(dirpath, case='real',
                            copy_params_toml=not toml_from_metadata)

        # copy the slices in the project directory
        for channel in ["ESB", "SE2"]:
            os.makedirs(dir_proj / channel, exist_ok=True)
            for src in (DATA / channel).glob('*.tif'):
                shutil.copy(src, dir_proj / channel)

        if toml_from_metadata:
            shutil.copy(DATA / 'Atlas3D.a3d-setup', dir_proj)
            fname_toml = ASSETS / 'toml' / 'params_real_stack.toml'
            params_from_metadata(dir_proj, fname_toml_ref=fname_toml, save=True)

        # processing
        stack = Stack3d(input_name=dir_proj)
        stack.eval(process_steps=process_steps, nproc=nproc,
                   show_pbar=show_pbar)

        # post-processing
        if process_steps is None:
            process_steps = stack.params['process_steps']
        channel = stack.params['channels'][0]
        dirnames, labels, stats = postpro(process_steps, dir_proj, channel,
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
    PROCESS_STEPS = [
        'cropping',
        'bkg_removal',
        # 'destriping',
        # 'registration_calculation', 'registration_transformation',
        # 'intensity_rescaling',
        # 'resampling',
    ]

    ex_real_stack(PROCESS_STEPS, DIRFUNC, NPROC)
