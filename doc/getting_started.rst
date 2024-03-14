Getting started
===============


Project directory organization
------------------------------

To be executed a **PyStack3D** workflow requires both:

* a ``params.toml`` file defining all the process steps parameters (see the `parameters settings <file:///C:/Users/PQ177701/PycharmProjects/pystack3d/doc/_build/html/params.html>`_ section)


* a **project directory** including the ``.tif`` images annotated with their respective **slice numbers** and their **z-positions** either in the root directory:

::

    project_dir
        |-- params.toml
        |-- slice_0000_z=0.0000um.tif
        |-- slice_0001_z=0.0100um.tif
        |   ...

or in sub-folders for a multi-channels acquisition::

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

*(The content of each 'output' directory is detailed below)*

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

By default, the reference ``params.toml`` file used by :code:`params_from_metadata` is the **raw** one. But the user can provide another reference .toml file through the ``fname_toml_ref`` argument::

   params_from_metadata(project_dir, save=True, fname_toml_ref=my_toml_ref)


Workflow execution
------------------

A **PyStack3D** workflow execution is obtained with the following instructions::

    from pystack3d import Stack3d

    stack = Stack3d(input_name)
    stack.eval(process_steps, nproc=16)

``input_name`` corresponds either to the **project directory pathname** or to the ``params.toml`` in which the project directory pathname has to be defined via the ``input_dirname`` parameter.

``process_steps`` refers either to a single process step or to a list of process steps or can be omitted leading to the execution of the full process steps defined in the ``params.toml`::

    # execute only the 'cropping' process step
    stack.eval(process_steps="cropping", nproc=16)

    # execute the 'cropping' and the 'background removal' process steps
    stack.eval(process_steps=["cropping", "bkg_removal"], nproc=16)

    # execute all the process steps defined in the 'params.toml' file
    stack.eval(nproc=16)

``nproc`` corresponds to the number of CPU to use for the workflow execution.

Note that an additional boolean keyword named ``serial`` allows to realize non-serialized calculations when setting to ``False`` (said differently, with ``serial = False`` the workflow is executed considering for each process step the original raw input data).


Outputs
-------

Each process steps returns **specific** and **standard** outputs (data and figures) in the related process step **outputs** directory.

**Specific** outputs are related to each process steps. They are described in each of the process steps sections hereafter (if existing).

**Standard** outputs consist in the statistics (min, max, mean) values evolution along the stack axis (z-axis, by convention) **before** and **after** the process step execution, considering for these last ones ('after') the statistics without and with a data reformatting compatible with the input data format. Indeed, some process steps may modify the data type (typically from integer to float) or generate data outside the range of authorized data values. *(This could happen for instance in the **bkg_removal** process step when subtracting the background that could generate negative or positive overflowed values)*.


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