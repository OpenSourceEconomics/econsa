Quantitative sensitivity analysis
=================================


When analyzing (complex) computational models it is often unclear from the model specification alone how the inputs of the model contribute to the outputs. As we've seen in the previous tutorial on *Qualitative sensitivity analysis*, a first step is to sort the inputs by their respective order of importance on the outputs. In many cases however, we would like to learn by how much the individual inputs contribute to the output in relation to the other inputs. This can be done using Sobol indices (:cite:`Sobol.1993`). Classical Sobol indices are designed to work on models with independent input variables. However, since in economics this independence assumption would be very questionable, we focus on so called generalized Sobol indices, as those proposed by :cite:`Kucherenko.2012`, that also work in the case of dependent inputs.

Generalized Sobol indices
^^^^^^^^^^^^^^^^^^^^^^^^^

Say we have a model :math:`\mathcal{M}:\mathbb{R}^n \to \mathbb{R}, x \mapsto \mathcal{M}(x)` and we are interested in analyzing the variance of its output on a given subset :math:`U \subset \mathbb{R}^n`, i.e. we want to analyze

.. math::

  D := \text{Var}(\mathcal{M}|_U) := \int_U (\mathcal{M}(x) - \mu_U)^2 f_X(x) \mathrm{d}x

where :math:`\mu_U := \int_U \mathcal{M}(x) f_X(x) \mathrm{d}x` denotes the restricted mean of the model and :math:`f_X` denotes the probability density function imposed on the input parameters. For the sake of brevity let us assume that :math:`\mathcal{M}` is already restricted so that we can drop the dependence on :math:`S`. To analyze the effect of a single variable, or more general a subset of variable, consider partitioning the model inputs as :math:`(y, z) = x`. The construction of Sobol and generalized Sobol indices starts with noticing that we can decompose the overall variance as

.. math::

  D = \text{Var}_y(\mathbb{E}_z\left[\mathcal{M}(y, z) \mid y \right]) + \mathbb{E}_y\left[\text{Var}_z(\mathcal{M}(y, z) \mid y)\right]

which implies that

.. math::

  1 = \frac{\text{Var}_y(\mathbb{E}_z\left[\mathcal{M}(y, z) \mid y \right])}{D} + \frac{\mathbb{E}_y\left[\text{Var}_z(\mathcal{M}(y, z) \mid y)\right]}{D} =: S_y + S_z^T

We call :math:`S_y` the *first order effect index* of the subset :math:`y` and we call :math:`S_z^T` the *total effect* of the subset :math:`z`. Notice that for each partition of the input space :math:`y` and :math:`z`, the above provides a way of computing the fraction of explained variance. For the sake of clarity, assume :math:`y` represent only a single input variable. Then :math:`S_y` can be interpreted as the effect of :math:`y` on the variability of :math:`\mathcal{M}` **without** considering any interaction effects with other variables. While :math:`S_y^T` can be thought of as representing the effect of :math:`y` on the variance via itself **and** all other input variables.

Again, we now apply this to the **EOQ** model. Given the current limits to our implementation and the fact that the parameters of the model need to remain positive, we specify that the parameters follow a normal distribution with a very small variance.

.. image:: ../../_static/images/fig-eoq-sensitivity-analysis-sobol.png
   :align: center
   :alt: Sobol indices

Shapely values
^^^^^^^^^^^^^^

This part will be written by `lindamaok899 <https://github.com/lindamaok899>`_ as part of her thesis.
