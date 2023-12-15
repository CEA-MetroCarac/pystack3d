Getting started
===============


The project directory
---------------------

To be executed a **PyStack3D** workflow requires:

 - 1/ a project directory with the ``.tif`` images annotated with their respective **z**-positions

 - 2/ a ``params.toml`` file defining all the process steps parameters.

Once executed, the output directories related to each process step take place inside the process directory as follows::

    project_dir
        |-- params.toml
        |-- slice_0000_z=0.0000um.tif
        |-- slice_0001_z=0.0100um.tif
        |   ...
        |-- process (ouputs)
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

Or for a multiple channels::

    project_dir
        |-- params.toml
        |-- channel_1
        |       |-- slice_0000_z=0.0000um.tif
        |       |-- slice_0001_z=0.0100um.tif
        |           ...
        |-- channel_2
        |       |-- slice_0000_z=0.0000um.tif
        |       |-- slice_0001_z=0.0100um.tif
        |           ...
        |-- process (outputs)
        |       |-- cropping
        |       |       |-- outputs
        |       |       |-- channel_1
        |       |       |       |-- slice_0000_z=0.0000um.tif
        |       |       |       |-- slice_0001_z=0.0100um.tif
        |       |       |       |   ...
        |       |       |-- channel_2
        |       |       |       |-- slice_0000_z=0.0000um.tif
        |       |       |       |-- slice_0001_z=0.0100um.tif
        |       |       |       |   ...
        |       |-- bkg_removal
        |       |       |-- outputs
        |       |       |-- channel_1
        |       |       |       |-- slice_0000_z=0.0000um.tif
        |       |       |       |-- slice_0001_z=0.0100um.tif
        |       |       |       |   ...
        |       |       |-- channel_2
        |       |       |       |-- slice_0000_z=0.0000um.tif
        |       |       |       |-- slice_0001_z=0.0100um.tif
        |       |       |       |   ...
        |        ...


Workflows execution
-------------------

A **PyStack3D** workflow execution is obtained with the following instructions::

    from pystack3d import Stack3d

    stack = Stack3d(input_name=input_name)
    stack.eval(process_steps=process_steps, nproc=16, show_pbar=True)

where ```input_name``` refers to:

- **EITHER** the project directory path (as a ```str``` or a ```Path``` object) that contains the `.tif` images **AND** the ```params.toml``` file

- **OR** the ```params.toml``` in which the project directory is defined via the ```input_dir``` parameter.

All the process steps defined in the `params.toml` or some of them can be executed as follows::

    # execute all the process steps defined in the 'params.toml' file
    stack.eval(nproc=16)

    # execute only the 'cropping' process step
    stack.eval(process_steps="cropping", nproc=16)

    # execute the 'cropping' and the 'background removal' process steps
    stack.eval(process_steps=["cropping", "bkg_removal"], nproc=16)

Note that an additional boolean keyword named ```serial``` allows to realize non-seralized calculations when setting to `False`.


Outputs
-------

stats et autres


Examples execution
------------------

Two examples can be launched.

THe first one with synthetic stack composed of small images::

    cd pystack3d
    python examples/ex_pystack3d_synth.py


The seconde one based on a real but reduced stack issued from FIB-SEM images::

    python examples/ex_pystack3d_real.py