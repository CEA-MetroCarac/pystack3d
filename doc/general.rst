General Parameters
==================

::

    input_dirname = "C:\Users\..."
    ind_min = 150
    ind_max = 1500
    channels = ["ESB", "SE2"]
    nproc = 16
    process_steps = [
        "cropping",
        "bkg_removal",
        "intensity_rescaling",
        "registration_calculation", "registration_transformation",
        "destriping",
        "resampling",
        "cropping_final"
    ]
    history = ["cropping", "bkg_removal"]

``input_dirname`` specifies where the input .tif files are stored (if not provided when creating the pystack3d.Stack() object).

``ind_min`` and ``ind_max`` are related to the frames indices to start and end the workflow. (If not specified, consider the first and the last frame)

``channels`` is used to indicate the channels names in the case of a multiple channels acquisition. In the case of a single channel acquisition, `.tif` files can be directly put at the root, equivalent to  ```channels = ["."]``` (default value)

``nproc`` allows to define the number of processors to work with

``process_steps`` is the list of the process steps to handle

``history`` is a list that can be used to specify the process steps that has already been handled when restarting. Note that during the workflow execution this list is automatically updated after each process step.
