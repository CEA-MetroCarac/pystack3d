Cropping
--------


Most of the time, the images stack acquisition area is wider than the Region Of Interest (ROI). **Cropping** is then dedicated to "crop" the images in the ROI.

.. figure:: _static/cropping.png
    :width: 400px
    :align: center

    Illustration of the **cropping** process step in the synthetic use case

::

    [cropping]
    area = [5, 45, 2, 42]

``area`` indicates the [xmin, xmax, ymin, ymax] coordinates (in pixel) of the image to crop in the standard coordinates system (with the origin located in the bottom left)
