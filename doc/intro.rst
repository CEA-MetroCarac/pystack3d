Introduction
============

What is PyStack3D ?
-------------------

PyStack3D is an open source library written in Python which provides tools to do
images stacks corrections, typically before image segmentation for images issued from FIB-SEM 3D acquisition.

The ``pystack3d`` workflow can be applied to multiple channels and consists in the following optional process steps that can be executed in any order:

- ``cropping`` to reduce the image field of view to a ROI (Region Of Interest)


- ``background removal`` to reduce from polynomial approximations artefacts issued for instance from shadowing, charging, ...


- ``intensity rescaling`` to homogenize the 'gray' intensity distribution between successive frames/slices


- ``registration`` to correct the images misalignment due to shifting, drift, rotation, ... during the images acquisition


- ``destriping`` to minimize artefacts like stripes that can appear in some image acquisition technics


- ``resampling`` to correct non uniform spatial steps


- ``cropping_final`` to eliminate artefacts produced near the edges during the image processing or to select another ROI at the end.


Install
-------

The ``pystack3d`` package install can be realized via a git cloning::

    git clone https://github.com/CEA-MetroCarac/pystack3d.git

or by a pypi install::

    pip install pystack3d
