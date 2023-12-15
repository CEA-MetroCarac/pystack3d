Introduction
============

What is PyStack3D
-----------------

PyStack3D is an open source Python library which provides tools to do
images stacks corrections (typically before image segmentation for images issued from FIB-SEM 3D acquisition).

The ``pystack3d`` workflow can be applied to multiple channels and consists in the following optional process steps that can be executed in any order:

- ``cropping`` to reduce the image to the ROI (Region Of Interest)


- ``background removal`` (named "bkg_removal") to reduce artefacts issued for instance from shadowing, charging, ...


- ``intensity rescaling`` (named "intensity_rescaling") to homogoneize the 'gray' intensity distribution between successive frames


- ``registration`` (decomposed in "registration_calculation" and "registration_transformation"), to correct images misalignment due to shifting, drift, rotation, ... during the images acquisition


- ``destriping`` to minimize artefacts like stripes that can appear in some image acquisition technics


- ``resampling`` to correct non uniform spatial steps


- ``cropping_final`` to eliminate artefacts produced near the edges during the image processing or to select another ROI at the end.


Install
-------

Install can be realized via a git cloning::

    git clone https://github.com/CEA-MetroCarac/pystack3d.git

or by a pypi install (to come)::

    pip install pystack3d
