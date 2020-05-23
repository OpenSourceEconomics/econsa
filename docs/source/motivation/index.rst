==========
Motivation
==========

Computational economic models clearly specify an individual's objective and the institutional and informational constraints of their economic environment under which they operate. Fully-parameterized computational implementations of the economic model are estimated on observed data as to reproduce the observed individual decisions and experiences. Based on the results, researchers can quantify the importance of competing economic mechanisms in determining economic outcomes and forecast the effects of alternative policies before their implementation
(:cite:`Wolpin.2013`).

The uncertainties involved in such an analysis are ubiquitous. Any such model is subject to misspecification, its numerical implementation introduces approximation error, the data is subject to measurement error, and the estimated parameters remain partly uncertain.

A proper accounting of the uncertainty is a prerequisite for the use of computational models in most disciplines (:cite:`Adams.2012,Oberkampf.2010`) and has long been recognized in economics as well (:cite:`Hansen.1996,Canova.1994,Kydland.1992`). However, in practice economists analyze the implications of the estimated model, economists display incredible certitude (:cite:`Manski.2019`) as all uncertainty is disregarded. As a result, flawed findings are accepted as truth and contradictory results are competing. Both have the potential to undermine the public trust in research in the long run.

Any computational economic model :math:`\boldsymbol{M}` provides a mapping between its input parameters :math:`\boldsymbol{x}` and the quantities of interest :math:`y`.

.. math::
  \boldsymbol{M} : \boldsymbol{x} \in \mathcal{D}_\boldsymbol{X} \mapsto y = \boldsymbol{M}(\boldsymbol{x})

We will first discuss uncertainty propagation and then discuss sensitivity analysis.

**Example**

.. todo::

  @loikein Please move here the description of the model along the lines in your notebook and the TESPY docstring. No AUTODOC or anything just a nice write-up that also gives the core equations. This also includes the figure (with description of the basic economics) you created to visualize the tradeoffs. Please add the code for that to docs/_static/codes and add its execution to our GitHub Action. Please make sure all references etc are properly cited in our bibliography section.

Let us consider the economic order quantity model. Economic order quantity model, or Harris model, was developed by Ford W. Harris (:cite:`Harris.1990`) to solve the problem of firms determining the ideal order size for one year, considering inventory costs and ordering costs.

The model takes four variables, and calculates the total ordering cost :math:`T`:

.. math::
  T = \frac{1}{2\times 12R\cdot M}(CX + S) +\frac{S}{X} + C

where :math:`M` denotes the number of units of good needed per month, :math:`C` is the unit price of the good, :math:`X` is the size of order in number of units, and :math:`S` is the cost of placing an order, also known as the setup cost.

:math:`I` is the interest and depreciation cost per month, and is treated as an exogenous parameter in most papers (:cite:`Harris.1990,Borgonovo.2016`).

.. figure:: ../../_static/images/fig-harris-tradeoff.png
   :align: center
   :alt: Figure that shows the set-up cost, interest & depreciation cost, and total cost of Harris EOQ model.

   An increase in the size of order results in a decrease in set-up costs, but an increase in interest & depreciation cost.

Therefore, if we know :math:`M` and :math:`C`, :math:`T` is then a function of :math:`X`, so we can calculate the optimal order size :math:`X^*` by solving:

.. math::
  \begin{aligned}\min_{X} && T &= \frac{1}{24IM}(CX + S) +\frac{S}{X} + C \\
  && \frac{\partial T}{\partial X} &= \frac{C}{24IM} - \frac{S}{X^2} \overset{!}{=} 0 \\
  && X^* &= \sqrt{\frac{24IMS}{C}}\end{aligned}

and :math:`X^*` is called the economic order quantity (EOQ).

In sensitivity analysis, :math:`X^*` is denoted as :math:`y`, the model output, and the model inputs are :math:`M`, :math:`C` and :math:`S`, denoted as :math:`\mathbf{x}=(x_1,x_2,x_3)'`, and :math:`I`, denoted as :math:`r`:

.. math::
  y = \sqrt{\frac{24r x_1 x_3}{x_2}}

We are interested in how :math:`y` changes depending on each :math:`x_i`, in other words, the sensitivity of :math:`y` with regard to each :math:`x_i`. In ``econsa``, 


.. toctree::
   :maxdepth: 1

   uncertainty-propagation
   sensitivity-analysis
