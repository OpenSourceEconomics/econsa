.. role:: raw-math(raw)
    :format: latex html

Uncertainty Propagation
=======================

The estimation step provides us with a probabilistic model for the input parameters. Going forward, we treat the model parameters :math:`\mathbf{X}` as a simple random vector following a normal distribution with mean :math:`\mu`, covariance matrix :math:`\Sigma`, and joint probability density function :math:`f_{\mathbf{X}}`. We are not particularly interested in the uncertainty of each individual parameter of the model. Instead we seek to learn about the induced distribution of the model output :math:`Y` as the uncertainty about the model parameters :math:`\mathbf{X}` propagates through the computational model :math:`\mathbf{M}`. We want to study the statistical properties of :math:`Y`.


Example
-------

.. todo::

  @loikein

  - [ ] add here a description and figure (no code) for uncertainty propagation figure 2 in Borgonovo & al reference. 
  - [x] add the code for that to docs/_static/codes and add its execution to our GitHub Action

Let us take a closer look at the economic order quantity model.

.. math::
  y = \sqrt{\frac{24 x_1 x_3}{rx_2}}

The model input :math:`\mathbf{X}=(X_1,X_2,X_3)` is a random vector with joint density function :math:`f_{\mathbf{X}}`, from which we draw a realisation :math:`\mathbf{x}=(x_1,x_2,x_3)`. The model takes :math:`\mathbf{x}`, and outputs :math:`Y` with probability density function :math:`f_{Y}`.

Below is a reproduction and expansion of Figure 2 from :cite:`Borgonovo.2016`, where :math:`\mathbf{x}` are drew from three independent uniform distributions with plus/minus 10% uncertainty, :math:`X_i\sim U[0.9 x_i^0, 1.1 x_i^0]`.
Through the computational model, the uncertainty propagates and we have what looks like a normally distributed output.

.. figure:: ../../_static/images/fig-borgonovo-cdf.png
   :align: center
   :alt: Figure of uncertainty propagation in Harris EOQ model.

   An expansion of Figure 2 of :cite:`Borgonovo.2016`



Prediction
----------

We want to have an image such as the one below following some simple model. The codes for these graphs are available in _static/codes.

.. image:: ../../_static/images/fig-illustration-density.png
  :width: 250

.. image:: ../../_static/images/fig-illustration-reliability.png
  :width: 250

Forecasting
-----------

We want to have an image such as below produced from some simple model.

.. image:: ../../_static/images/fig-fan-chart.png
  :width: 25
