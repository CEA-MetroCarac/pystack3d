Resampling
==========

The resampling process step recreates slices with a uniform distribution (that is to say, with a constant step size along the z-axis), using linear interpolations between slices/frames.

.. figure:: _static/resampling.png
    :width: 400px
    :align: center

    Illustration of the **resampling** process step in the `synthetic test case <https://github.com/CEA-MetroCarac/pystack3d/blob/main/pystack3d/examples/ex_pystack3d_synth.py>`_.

::

    [resampling]
    policy = "slice_{slice_nb}_z={z_coord}um.tif"
    dz = 0.5

``policy`` refers to the file naming including the number of the slice (``slice_nb``) and its z coordinates (``z_coord``).

``dz`` corresponds to the space step to be considered in the resampling.


Plotting
--------

The special plotting related to the **resampling** process step generates images in the dedicated **outputs**  folder that are named **z_positions.png** and **z_increments.png**


.. figure:: _static/resampling_positions.png
    :width: 400px
    :align: center

    **z_positions.png** shows the slices positions before (in) and after (out) the resampling.


.. figure:: _static/resampling_increments.png
    :width: 400px
    :align: center

    **z_increments.png** shows the slices z-increments before (in) and after (out) the resampling.
