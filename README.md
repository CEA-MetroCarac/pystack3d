![](doc/_static/logo.png)

# Introduction

**PyStack3D** is a package dedicated to images correction intended -for instance- to FIB-SEM stack images postprocessing  before image segmentation.

The ``pystack3d`` workflow includes the following process steps which can be activated or not and executed in any order:

- ``cropping`` to reduce the image to the ROI (Region Of Interest)

- ``background removal`` (named "bkg_removal") to reduce artefacts issued for instance from shadowing, charging, ...

- ``intensity rescaling`` (named "intensity_rescaling") to homogoneize the 'gray' intensity distribution between successive frames

- ``registration`` (decomposed in "registration_calculation" and "registration_transformation"), to correct images misalignment due to shifting, drift, rotation, ... during the images acquisition

- ``destriping`` to minimize artefacts like stripes that can appear in some image acquisition technics

- ``resampling`` to correct non uniform spatial steps

An additonal step named `cropping_final` can be used to eliminate artefacts produced near the edges during the image processing or to select another ROI at the end.

<p align="center" width="100%">
    <img align="center" width=100%" src=doc/_static/process_steps_real.png> <br>
    <em>Illustration of a <b>PyStack3D</b> workflow applied on a FIB-SEM image and using some of the process steps.</em>
</p>

<p align="center" width="100%">
    <img align="center" width=100%" src=doc/_static/process_steps.png> <br>
    <em>Y and Z-cutplanes of a non-serialized <b>PyStack3D</b> workflow applied to an analytical case to illustrate each process step individually.
    Input images are the result of shifting, background and stripes addition with periodically altered intensities. The central 'bend-shape' pattern is the consequence of the shifting and non-uniform frame increments. (See examples/ex_pystack3d_synth.py).</em>
</p>

# Install and examples execution

```
git clone https://github.com/CEA-MetroCarac/pystack3d.git
cd pystack3d
python examples/ex_pystack3d_synth.py
python examples/ex_pystack3d_real.py
```

# Quick start

To be executed a **PyStack3D** workflow requires:

 - 1/ an input directory with the ``.tif`` images annoted with their respective ``z``-position
 
 - 2/ a ``prop.toml`` file defining all the process steps parameters.

Once executed, the output directories related to each process step take place inside the input directory as follows:

<pre>
input_dir
    |-- prop.toml
    |-- slice_0000_z=0.0000um.tif
    |-- slice_0001_z=0.0100um.tif
    |   ...
    |-- process
    |       |-- cropping
    |       |       |-- outputs
    |       |       |-- slice_0000_z=0.0000um.tif
    |       |       |-- slice_0001_z=0.0100um.tif
    |       |           ...
    |       |-- bkg_removal
    |       |       |-- outputs
    |       |       |-- slice_0000_z=0.0000um.tif
    |       |       |-- slice_0001_z=0.0100um.tif
    |       |           ...
    |        ...
</pre>

The **PyStack3D** workflow execution is obtained from: 
```
from pystack3d import Stack3d

stack = Stack3d(input_name=input_name)
stack.eval(process_steps=process_steps, nproc=16, show_pbar=True)
```
where `input_name` refers:

- EITHER to the input directory path (as a `str` or a `Path` object) that contains the `.tif` images AND the `prop.toml` file 

- OR the `prop.toml` with the `input_dir` defined as parameter.

All the process steps defined in the `prop.toml` or some of them can be executed as follows: 
```
# execute all the process steps defined in the 'prop.toml' file
stack.eval(nproc=16)

# execute only the 'cropping' processing
stack.eval(process_steps="cropping", nproc=16)

# execute the 'cropping' and the 'background removal' processing
stack.eval(process_steps=["cropping", "bkg_removal"], nproc=16)
```
Note that an additional boolean keyword named `serial` allows to realize non-seralized calculations when setting to `False`.
