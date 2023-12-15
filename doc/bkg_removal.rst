Background removal
==================

Most of the images stacks acquisition techniques can engender a gray intensities increasing or decreasing due to **charging** effects (on low conductive material) or to **shadowing** effects (when carving it).
The **background removal** process step aims at removing these effects assuming a polynomial shape to fit.
The fit is realised through the resolution of a minimization problem with a least square algorithm.

.. figure:: _static/bkg_removal.png
    :width: 400px
    :align: center

    Illustration of the **bkg_removal** process step in the synthetic use case

::

    [bkg_removal]
    dim = 3
    poly_basis = "1 + x + y + x*y + x**2 + z"
    #orders = [2, 1, 1]
    #cross_terms = true
    skip_factors = [5, 5, 5]
    # skip_averaging = false
    threshold = 0.5
    # inversion = 0
    # weight_func = "HuberT"


the ``dim`` parameter defines the dimension of the problem to solve (2D or 3D).
In 3D a single one but potentially high CPU cost resolution is performed on all the stack, whereas in 2D the resolution is performed on each slice, independently to the others.

The 2D or 3D polynomial basis can be defined in two ways:

    - ``poly_basis`` which allows to define the polynomial basis **explicitly**, term by term

    - ``orders`` and ``cross_terms`` which define the basis **implicitly** in function to the order associated to each variable (x, y) in 2D or (x, y, z) in 3D.

Note that in the example above, the two approaches define two different basis. Indeed, in the implicit approach, the cross terms parameter (set to ``True``) generates terms like 'z*x*y' and  'z*x**2' that are not present in the literal expression.
In this sense, the explicit approach provides more flexibility in the polynomial basis definition

Since the images sizes could be big as well as the number of frames,  the ``skip_factors`` parameter allows to significantly reduced the array to consider in the fitting processing.
Setting a ``skip_factors = [10, 10, 10]`` on a stack of size ~1000x1000x1000 is a good compromise between accuracy, RAM occupancy and time execution.

When a skip factor is applied, the considered values taken into account are the ones corresponding to the indices positions. Using ``skip_averaging : true`` realizes a local averaging instead. (Note that no real benefit have been observed practically when activating this averaging).


Images may include empty regions (like holes) that are not impacted by the background effects.
For this reason, a mask defined via the ``threshold`` parameter (in term of gray level values) and associated to ``inversion`` can be applied during the fit processing.

    ``inversion: 0`` *(default mode)* creates a "dark" mask from [min, threshold]
    ``inversion: 1`` creates a "bright" mask from [max-threshold, max], and
    ``inversion: 2`` ceates a "symmetrical" mask as the combination of the "dark" and the "bright" mask.

At least, the ``weight_func`` parameter ("**HuberT**" or "**Hampel**") can be used to specifiy a weight function used to not consider outliers during the least squares problem resolution. (default value is HuberT).
See  the `RLM documentation <https://www.statsmodels.org/stable/rlm.html>`_ for more details.