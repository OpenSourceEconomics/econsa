==========
Motivation
==========

Computational economic models clearly specify an individual's objective and the institutional and informational constraints of their economic environment under which they operate. Fully-parameterized computational implementations of the economic model are estimated on observed data as to reproduce the observed individual decisions and experiences. Based on the results, researchers can quantify the importance of competing economic mechanisms in determining economic outcomes and forecast the effects of alternative policies before their implementation \citep{Wolpin.2013}.

The uncertainties involved in such an analysis are ubiquitous. Any such model is subject to misspecification, its numerical implementation introduces approximation error, the data is subject to measurement error, and the estimated parameters remain partly uncertain.

A proper accounting of the uncertainty is a prerequisite for the use of computational models in most disciplines \citep{Adams.2012,Oberkampf.2010} and has long been recognized in economics as well \citep{Hansen.1996,Canova.1994,Kydland.1992}. However, in practice economists analyze the implications of the estimated model, economists display incredible certitude \citep{Manski.2019} as all uncertainty is disregarded. As a result, flawed findings are accepted as truth and contradictory results are competing. Both have the potential to undermine the public trust in research in the long run.

Any computational economic model :math:`\boldsymbol{M}` provides a mapping between its input parameters :math:`\boldsymbol{x}` and the quantities of interest :math:`y`.

.. math::
  \boldsymbol{M} : \boldsymbol{x} \in \mathcal{D}_\boldsymbol{X} \mapsto y = \boldsymbol{M}(\boldsymbol{x})

We will first discuss uncertainty propagation and then discuss sensitivity analysis.

**Example**

We will use Ford W. Harris’ economic order quantity model (Harris EOQ, henceforth) (Harris, 1913) as our running example throughout this section

.. toctree::
   :maxdepth: 1

   uncertainty-propagation
   sensitivity-analysis
