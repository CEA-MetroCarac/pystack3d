Parameters settings
===================

The PyStack3D workflow parameters are defined in the ``params.toml`` file.

The non encapsulated parameters are related to the "general" parameters that specify the conditions of the **workflow** execution.

The encapsulated parameters are related to the conditions of each **process step** execution.

To illustrate each of the process step we will consider hereafter a centered square-section pattern imaged with 100 frames and the following defects:

    - an additional background (defined from a polynom),

    - a drift along the x-axis,

    - a non-uniform sampling according to the z-axis which, coupled with the x-drift, gives the corresponding curved shape

    - 2 oblique stripes

    - and periodically, a change in the gray levels distribution

.. image:: _static/synth_use_case.png
    :width: 400px
    :align: center
