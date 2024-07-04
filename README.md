[![PyPI](https://img.shields.io/pypi/v/pystack3d?label=pypi%20package)](https://pypi.org/project/pystack3d/)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](https://github.com/CEA-MetroCarac/pystack3d)
[![Doc](https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg)](https://cea-metrocarac.github.io/pystack3d/doc/index.html)


<p align="center" width="100%">
    <img align="center" width=400 src=https://cea-metrocarac.github.io/pystack3d/logo.png>
</p>

## Introduction

**PyStack3D** is a package dedicated to images correction intended -for instance- to FIB-SEM stack images postprocessing before image segmentation.

The ``pystack3d`` workflow includes the following process steps which can be activated or not and executed in any order:

- ``cropping`` to reduce the image field of view to a ROI (Region Of Interest)

- ``background removal`` to reduce from polynomial approximations artefacts issued for instance from shadowing, charging, ...

- ``intensity rescaling`` to homogenize the 'gray' intensity distribution between successive frames/slices

- ``registration`` to correct the images misalignment due to shifting, drift, rotation, ... during the images acquisition

- ``destriping`` to minimize artefacts like stripes that can appear in some image acquisition technics

- ``resampling`` to correct non uniform spatial steps

An additional step named `cropping_final` can be used to eliminate artefacts produced near the edges during the image processing or to select another ROI at the end.

<p align="center" width="100%">
    <img align="center" width=80%" src=https://cea-metrocarac.github.io/pystack3d/pystack3d.png> <br>
    <em>a) Synthetic case illustrating the defects to be removed by <b>PyStack3D</b>. b) Corrected stack. c) Ground truth.</em>
</p>

<p align="center" width="100%">
    <img align="center" width=100%" src=https://cea-metrocarac.github.io/pystack3d/process_steps_real.png> <br>
    <em>Illustration of a FIB-SEM image correction using some of the <b>PyStack3D</b> process steps.</em>
</p>

## Installation

```
pip install pystack3d
```

## Tests and examples execution

For tests and examples execution, the full ``pystack3d`` project has to be installed via ``git``:

```
    git clone https://github.com/CEA-MetroCarac/pystack3d.git
    cd [path_to_your_pystack3d_project]
```

Once the project has been cloned, the python environment has to be created and completed with the ``pytest`` package (for testing):

```
    pip install poetry
    poetry install
    pip install pytest
```

Then the tests and the examples can be executed as follows:

```
    pytest
    cd examples
    python ex_synthetic_stack.py
    python ex_real_stack.py
```

## Usage

Refer to the [PyStack3D documentation](https://cea-metrocarac.github.io/pystack3d/doc/getting_started.html).


## Contributing / Reporting an issue

Contributions and issue reporting are more than welcome!
Please read through [our Developers notes](https://cea-metrocarac.github.io/pystack3d/doc/dev_notes.html).


## Acknowledgements

This work, carried out on the CEA - Platform for Nanocharacterisation (PFNC), was supported by the “Recherche Technologique de Base” program of the French National Research Agency (ANR).

## Citations

In case you use the results of this code in an article, please cite:

- Quéméré P., David T. (2024). PyStack3D: A Python package for fast image stack correction. *Journal of Open Source Software. (submitted)*

additional citations for the <b>destriping</b>:

- Pavy K., Quéméré P. (2024). Pyvsnr 2.0.0. Zenodo. https://doi.org/10.5281/zenodo.10623640

- Fehrenbach J., Weiss P., Lorenzo C. (2012). Variational algorithms to remove stationary noise: applications to microscopy imaging. *IEEE Transactions on Image Processing 21.10 (2012): 4420-4430.*

additional citation for the <b>registration</b>:

- Thévenaz P., Ruttimann U.E., Unser M. (1998), A Pyramid Approach to Subpixel Registration Based on Intensity, *IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, January 1998*