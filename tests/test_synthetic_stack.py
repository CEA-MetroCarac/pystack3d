"""
Test pystack3d in the frame of synthetic images
"""
from tempfile import TemporaryDirectory
import numpy as np
from pytest import mark, approx

from examples.ex_synthetic_stack import ex_synthetic_stack as ex_synth

DIRFUNC = TemporaryDirectory
NPROC = [1, 2]


@mark.parametrize("nproc", NPROC)
def test_cropping(nproc):
    stats = ex_synth(process_steps='cropping',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.5, 0.5],
                    [4.72680154, 7.02478134],
                    [2.2953875, 3.89564244]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_bkg_removal(nproc):
    stats = ex_synth(process_steps='bkg_removal',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.5, 0.5],
                    [1., 4.04643856],
                    [0.92, 1.36098475]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_intensity_rescaling(nproc):
    stats = ex_synth(process_steps='intensity_rescaling',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.51367188, 0.51367188],
                    [6.03417601, 7.48632812],
                    [3.39309324, 3.52853035]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_destriping(nproc):
    stats = ex_synth(process_steps='destriping',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.5, 0.5],
                    [5.2020202, 7.5],
                    [2.43828806, 4.29423897]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_registration(nproc):
    stats = ex_synth(process_steps=['registration_calculation',
                                    'registration_transformation'],
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.5, 0.5],
                    [5.2020202, 7.5],
                    [2.43421592, 4.7119125]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_resampling(nproc):
    stats = ex_synth(process_steps='resampling',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0.5, 0.5],
                    [5.2020202, 7.49735938],
                    [2.43421592, 4.28343369]])

    assert stats == approx(ref)
