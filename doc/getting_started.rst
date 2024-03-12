Getting started
===============


Project directory
-----------------

To be executed a **PyStack3D** workflow requires both:

* a ``params.toml`` file defining all the process steps parameters

* a project directory including the ``.tif`` images annotated with their respective **z**-positions either in the root directory or in sub-folders for a multi-channels acquisition.


Once executed, the **outputs** directories related to each process step take place inside the ``process`` directory as follows::

    project_dir
        |-- params.toml
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

or for a multi-channels acquisition::

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
        |-- process
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


Workflow initialization
-----------------------

All the workflow instructions are provided in a ``params.toml``.

A **raw** ``params.toml`` is given `here <https://github.com/CEA-MetroCarac/pystack3d/blob/main/assets/toml/params.toml>`_ (to be adapted according to needs).


Fibics
~~~~~~

In the frame of **Fibics** (FIB-SEM) acquisitions, **metadata** can be extracted from the .tif files and the ``Atlas3D.a3d-setup`` file as follows::

   from pystack3d.utils_metadata_fibics import params_from_metadata

   params_from_metadata(project_dir, save=True)

where ``project_dir`` refers to the project directory pathname containing the .tif files and the ``Atlas3D.a3d-setup`` file (mandatory file).

By default, the reference ``params.toml`` file considered, modified and saved in the project directory is the **raw** one. But the user can provide another reference .toml file through the ``fname_toml_ref`` argument::

   params_from_metadata(project_dir, save=True, fname_toml_ref=my_toml_ref)


Workflow execution
------------------

A **PyStack3D** workflow execution is obtained with the following instructions::

    from pystack3d import Stack3d

    stack = Stack3d(input_name)
    stack.eval(process_steps, nproc=16, show_pbar=True)

``process_steps`` refers to a list of process to be executed

``input_name`` corresponds:

- either to the **project directory pathname** that contains the ``.tif`` images and the ``params.toml`` file

- or to the ``params.toml`` in which the project directory pathname is defined via the ``input_dir`` parameter.

All the process steps defined in the ``params.toml`` or some of them can be executed as follows::

    # execute all the process steps defined in the 'params.toml' file
    stack.eval(nproc=16)

    # execute only the 'cropping' process step
    stack.eval(process_steps="cropping", nproc=16)

    # execute the 'cropping' and the 'background removal' process steps
    stack.eval(process_steps=["cropping", "bkg_removal"], nproc=16)

Note that an additional boolean keyword named ``serial`` allows to realize non-serialized calculations when setting to ``False``(said differently, with ``serial = False`` the workflow is executed considering the original input data for each process step).


Outputs
-------

Each process steps returns **specific** and **standard** outputs (data and figures) in the related process step **outputs** directory.

**Specific** outputs are related to the each process steps. They are described in each of the process steps sections, if existing.

**Standard** outputs consist in the statistics (min, max, mean) values evolution along the stack axis (z-axis, by convention) before and after the related process step, considering for these last ones the statistics before and after a data reformatting compatible with the input data format. Indeed, some process steps may modify the data type (typically from integer to float) or generate data outside the range of authorized data values. *(This could happen for instance in the **bkg_removal** process step when subtracting the background that could generate negative or positive overflowed values)*.


.. figure:: _static/stats_bkg_removal.png
    :width: 80%
    :align: center

    Example of statistics returned by the **bkg_removal** process step in the `synthetic test case <https://github.com/CEA-MetroCarac/pystack3d/blob/main/pystack3d/examples/ex_pystack3d_synth.py>`_.


Examples
--------

Two examples are provided with the pystack3d package github repository.

The first one corresponds to a synthetic stack composed of small images. It aims at providing quick overviews of the process steps outcomes::

    cd pystack3d
    python examples/ex_synthetic_stack.py

The second one is based on a real but reduced stack (8 slices) issued from a FIB-SEM images acquisition. Although reduced, its execution is longer than the previous one::

    python examples/ex_real_stack.py