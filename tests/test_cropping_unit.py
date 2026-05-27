"""
Test pystack3d cropping - unit tests for inds_from_area().

Coordinate conventions
----------------------
area = (xmin, xmax, ymin, ymax)

x-axis -> columns (j) : jmin=xmin, jmax=xmax  (direct mapping)
y-axis -> rows    (i) : imin=H-ymax, imax=H-ymin  (flipped)
"""
import warnings

import numpy as np
import pytest
from tifffile import imwrite

from pystack3d.cropping import inds_from_area


def make_tiffs(tmp_path, shape=(100, 120), n=3):
    """
    Write blank TIFF files of a given shape into a temporary directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory to write files into.
    shape : tuple of int, optional
        Image shape as (height, width). Default is (100, 120).
    n : int, optional
        Number of TIFF files to create. Default is 3.

    Returns
    -------
    list of pathlib.Path
        Sorted list of created file paths.
    """
    paths = []
    for i in range(n):
        p = tmp_path / f"slice_{i:03d}.tif"
        imwrite(p, np.zeros(shape, dtype=np.uint8))
        paths.append(p)
    return paths


def test_no_area_returns_full_image(tmp_path):
    """
    area=None must return the full image extent.

    When no cropping area is specified, inds_from_area should return
    indices spanning the entire image dimensions.
    """
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    imin, imax, jmin, jmax = inds_from_area(
        area=None, fnames=fnames, pid_0=False, output_dirname=None)

    assert (imin, imax, jmin, jmax) == (0, 100, 0, 120)


def test_x_maps_directly_to_columns(tmp_path):
    """
    xmin and xmax must map directly to jmin and jmax without transformation.

    The x-axis corresponds to image columns and requires no coordinate
    conversion.
    """
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    imin, imax, jmin, jmax = inds_from_area(
        area=(10, 80, 0, 100), fnames=fnames, pid_0=False, output_dirname=None)

    assert jmin == 10
    assert jmax == 80


def test_y_axis_is_flipped(tmp_path):
    """
    The y-axis must be flipped when converting to row indices.

    The expected transformation is: imin = H - ymax, imax = H - ymin,
    where H is the image height.
    """
    fnames = make_tiffs(tmp_path, shape=(100, 120))  # H = 100

    imin, imax, jmin, jmax = inds_from_area(
        area=(0, 120, 20, 60), fnames=fnames, pid_0=False, output_dirname=None)

    assert imin == 40  # 100 - 60
    assert imax == 80  # 100 - 20


def test_cropped_size_matches_area(tmp_path):
    """
    The output index range must match the requested area dimensions.

    The number of rows and columns in the cropped region must equal
    the height and width specified in the area parameter.
    """
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    imin, imax, jmin, jmax = inds_from_area(
        area=(10, 90, 10, 70), fnames=fnames, pid_0=False, output_dirname=None)

    assert jmax - jmin == 80  # 90 - 10
    assert imax - imin == 60  # 70 - 10


def test_warning_on_x_overflow(tmp_path):
    """
    A UserWarning must be raised when xmax exceeds the image width.

    Overflow detection is only active for the primary process (pid_0=True).
    """
    (tmp_path / "outputs").mkdir()
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    with pytest.warns(UserWarning):
        inds_from_area(
            area=(0, 999, 0, 50),  # xmax=999 > width=120
            fnames=fnames, pid_0=True, output_dirname=tmp_path)


def test_warning_on_y_overflow(tmp_path):
    """
    A UserWarning must be raised when ymax exceeds the image height.

    Overflow detection is only active for the primary process (pid_0=True).
    """
    (tmp_path / "outputs").mkdir()
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    with pytest.warns(UserWarning):
        inds_from_area(
            area=(0, 50, 0, 999),  # ymax=999 > height=100
            fnames=fnames, pid_0=True, output_dirname=tmp_path)


def test_no_warning_when_not_pid_0(tmp_path):
    """
    Worker processes (pid_0=False) must never emit warnings.

    Even when the area exceeds the image bounds, only the primary process
    is responsible for issuing overflow warnings.
    """
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning causes test failure
        inds_from_area(
            area=(0, 999, 0, 999),  # out of bounds but pid_0=False
            fnames=fnames, pid_0=False, output_dirname=None)


def test_log_file_written_by_pid_0(tmp_path):
    """
    The primary process (pid_0=True) must write a log file with shape info.

    The log file at outputs/log.txt must contain both the original and
    new image shape after cropping.
    """
    (tmp_path / "outputs").mkdir()
    fnames = make_tiffs(tmp_path, shape=(100, 120))

    inds_from_area(
        area=(10, 80, 10, 80),
        fnames=fnames, pid_0=True, output_dirname=tmp_path)

    log = (tmp_path / "outputs" / "log.txt").read_text()
    assert "Original shape" in log
    assert "New shape" in log