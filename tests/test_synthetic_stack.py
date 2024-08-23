"""
Test pystack3d in the frame of synthetic images
"""
import os
from tempfile import TemporaryDirectory
import numpy as np
from pytest import mark, approx

from examples.ex_synthetic_stack import ex_synthetic_stack as ex_synth

DIRFUNC = TemporaryDirectory
NPROC = [1, 2] if os.cpu_count() > 1 else [1]


@mark.parametrize("nproc", NPROC)
def test_cropping(nproc):
    stats = ex_synth(process_steps='cropping',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 0.],
                    [127., 127.],
                    [55.33374444, 71.96906667]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_bkg_removal(nproc):
    stats = ex_synth(process_steps='bkg_removal',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 0.],
                    [124., 129.],
                    [52.56102222, 68.74597778]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_intensity_rescaling(nproc):
    stats = ex_synth(process_steps='intensity_rescaling',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 4.],
                    [125., 126.],
                    [61.44863333, 63.37102222]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_destriping(nproc):
    stats = ex_synth(process_steps='destriping',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 0.],
                    [132., 149.],
                    [55.00375556, 77.45533333]])

    assert stats == approx(ref, rel=1e-2)


@mark.parametrize("nproc", NPROC)
def test_registration(nproc):
    stats = ex_synth(process_steps=['registration_calculation',
                                    'registration_transformation'],
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 0.],
                    [127., 127.],
                    [54.14760417, 72.03203704]])

    assert stats == approx(ref)


@mark.parametrize("nproc", NPROC)
def test_resampling(nproc):
    stats = ex_synth(process_steps='resampling',
                     nproc=nproc, dirfunc=DIRFUNC,
                     verbosity=False, show_pbar=False, show_plots=False)

    ref = np.array([[0., 0.],
                    [127., 127.],
                    [55.22061111, 71.96808889]])

    assert stats == approx(ref)
