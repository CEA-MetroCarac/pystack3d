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

``input_dirname`` specifies where the input `.tif` files are stored (to be defined if not passed in pystack3d.Stack3d(``input_name``)).

``ind_min`` and ``ind_max`` are related to the frames (slices) indices where to start and end the workflow processing. (If not specified, consider the first and the last frame).

``channels`` is used to indicate the channel names in the case of a multiple channels acquisition. In the case of a single channel acquisition, `.tif` files can be directly put at the root of the input directory, equivalent to  ``channels = ["."]`` (default value).

``nproc`` allows to define the default value for the number of parallel threads to use for the processing.

``process_steps`` is the list of the process steps to execute.

``history`` is a list that can be used to specify the process steps that have already been done in the case of stopping the workflow in the middle. It allows the workflow to restart from the last finished step and with the correct input data (the last 'outputs' directory). Note that during the workflow execution this list is automatically updated after each process step in order to be able to restart in case of any unexpected interruption.
