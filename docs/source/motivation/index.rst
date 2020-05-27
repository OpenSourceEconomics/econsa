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

We follow :cite:`Borgonovo.2016` and use the **Economic Order Quantity (EOQ)** model (:cite:`Harris.1990`) as a running example throughout our documentation. We thus start by explaining its basic setup first and then discuss uncertainty propagation and sensitivity analysis.

.. toctree::
   :maxdepth: 1

   eoq-model
   uncertainty-propagation
   sensitivity-analysis
