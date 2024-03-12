[![PyPI](https://img.shields.io/pypi/v/pystack3d?label=pypi%20package)](
https://pypi.org/project/pystack3d/)
[![Github](https://img.shields.io/badge/GitHub-GPL--3.0-informational)](
https://github.dev/CEA-MetroCarac/pystack3d)
[![Doc](https://img.shields.io/badge/%F0%9F%95%AE-docs-green.svg)](
https://cea-metrocarac.github.io/pystack3d/doc/index.html)


<p align="center" width="100%">
    <img align="center" width=250 src=https://cea-metrocarac.github.io/pystack3d/logo.png>
</p>

# Introduction

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
    <img align="center" width=100%" src=doc/_static/process_steps_real.png> <br>
    <em>Illustration of a <b>PyStack3D</b> workflow applied on a FIB-SEM image and using some of the process steps.</em>
</p>

## Installation

```
pip install pystack3d
```

## Tests and examples execution

```
pip install pytest
git clone https://github.com/CEA-MetroCarac/pystack3d.git
cd pystack3d
pytest
python examples/ex_synthetic_stack.py
python examples/ex_real_stack.py
```

### Authors information

In case you use the results of this code in an article, please cite:

- Quéméré P., David T. (2024). PyStack3D: A Python package for fast image stack correction. *Journal of Open Source Software. (submitted)*