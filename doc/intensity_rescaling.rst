Intensity rescaling
===================

The **intensity rescaling** process aims at homogenize the gray level histograms between consecutive images.

To achieve this, a reference histogram is defined, corresponding either to the average histogram across all slices or to a running average along the slices (z-axis).

.. figure:: _static/intensity_rescaling.png
    :width: 400px
    :align: center

    Illustration of the **intensity_rescaling** process step in the `synthetic test case <https://github.com/CEA-MetroCarac/pystack3d/blob/main/examples/ex_synthetic_stack.py>`_.
    In the present case, the reference histogram has been defined from all the slices. This leads to values after intensity rescaling that are nearly **constant** along the z-axis.


::

    [intensity_rescaling]
    nbins = 256
    #range_bins = [0, 6]
    filter_size = -1

``nbins`` corresponds to the number of bins associated to the histograms.

``range_bins`` defines the range of the gray values to be considered for the binning.
If this parameter is not specified, then a preliminary loop on all the slices are performed to determine the **min** and **max** values associated to the full stack, leading to **range_bins = [min, max]**.

``filter_size`` is related to the running averaging for determining the reference histograms.
A positive integer value is associated to the number of slices/frames to consider in the running average.
``filter_size = -1`` corresponds to an averaging performed across all slices.



Plotting
--------

The special plotting related to the **intensity_rescaling** process step generates images in the dedicated **outputs**  folder that are named **intensity_rescaling_profiles.png** and **intensity_rescaling_maps.png**.

.. figure:: _static/intensity_rescaling_profiles.png
    :align: center

    **intensity_rescaling_profiles.png** gives the histograms profiles calculated for each slice.

.. figure:: _static/intensity_rescaling_maps.png
    :align: center

    **intensity_rescaling_maps.png** returns the same information but according to intensity maps along the z-axis.

Note that in the map representation the maximal intensity values of the 2D-histogram (near 400) are hidden by the y-axis (# Frames).