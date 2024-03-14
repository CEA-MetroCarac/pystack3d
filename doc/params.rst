Parameters settings
===================

The PyStack3D workflow parameters are defined in a ``params.toml`` file that is structured as follows:

* The first parameters are related to the **general** parameters that specify the conditions of the **workflow** execution.

* The following 'encapsulated' parameters are related to the conditions of each **process step** execution.

To illustrate each of the process step hereafter we will consider the `synthetic test case <https://github.com/CEA-MetroCarac/pystack3d/blob/main/examples/ex_synthetic_stack.py>`_ corresponding to a centered square-section pattern imaged with 100 frames and the following defects:

    - an additional background (defined from a polynom),

    - a drift along the x-axis,

    - a non-uniform sampling according to the transverse directon (z-axis) which, coupled with the x-drift, gives the corresponding curved shape of the central square pattern

    - 2 oblique stripes

    - and a periodic change in the gray levels distribution (occurring all the 11 slices)

.. figure:: _static/synth_use_case.png
    :width: 400px
    :align: center

    Front and top views related to the Z=50 and Y=25 cross sections without and with the additional defects.


.. toctree::
    :titlesonly:

    general.rst
    cropping.rst
    bkg_removal.rst
    intensity_rescaling.rst
    registration.rst
    destriping.rst
    resampling.rst
    cropping_final.rst