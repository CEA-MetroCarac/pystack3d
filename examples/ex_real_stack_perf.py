"""
Example to evaluate PyStack3D multiprocessing performance on real images
"""
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from pystack3d import Stack3d

from examples.utils import init_dir
from examples.utils import UserTempDirectory  # pylint: disable=unused-import

ASSETS = Path(__file__).parents[1] / 'assets'
DATA = ASSETS / 'stacks' / 'stack_1'

POLICY = "slice_{slice_nb}_z={z_coord}um.tif"


def ex_real_stack_perf(process_steps=None, dirfunc=None, nproc=None,
                       nslices=50):
    """ Example with real data """

    with dirfunc() as dirpath:  # open user temp or TemporaryDirectory dirpath

        # Initialize the project directory in the temporary folder
        dir_proj = init_dir(dirpath, case='real_perf')
        os.makedirs(dir_proj, exist_ok=True)

        # duplicate the first slice in the project directory
        slice_ref = list((DATA / 'ESB').glob('*.tif'))[0]
        print('duplication ...')
        for k in range(nslices):
            z = 0.01 * k + 0.002 * (k % 2 - 0.5)
            name = POLICY.format(slice_nb=f'{k:04d}', z_coord=f'{z:.4f}')
            shutil.copy(slice_ref, Path(dir_proj) / name)
        print('done')

        # processing
        stack = Stack3d(input_name=dir_proj)
        stack.eval(process_steps=process_steps, nproc=nproc)


if __name__ == '__main__':
    # DIRFUNC = UserTempDirectory  # use the user temp location
    DIRFUNC = TemporaryDirectory  # use a TemporaryDirectory
    NPROC = 32
    PROCESS_STEPS = [
        'cropping',
        'bkg_removal',
        'destriping',
        'registration_calculation', 'registration_transformation',
        'intensity_rescaling',
        'resampling',
    ]
    NSLICES = 2000

    print("\n\nWARNING: This test requires more than 100GB of disk space.\n"
          "Please ensure that you have enough free space on your disk before "
          "running it.")
    # ex_real_stack_perf(PROCESS_STEPS, DIRFUNC, NPROC, NSLICES)
