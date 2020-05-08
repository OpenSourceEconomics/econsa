============
Theory
============

.. todo::

  We will write this once we have a first version of our paper on the topic out. Then we can take most of the material from there.

Here we give an overview of the theory behind uncertainty quantification and
sensitivity analysis with a focus on the intersection between qualitative
(Extended Elementary Effects Method also known as the Morris Method) and
quantitative methods (TF).

Uncertainty quantification and sensitivity analysis provide rigorous procedures
to analyse and characterize the effects of parameter uncertainty on the output
of a model.
The methods for uncertainty quantification and sensitivity analysis can be
divided into global and local methods.
Local methods keep all but one model parameter fixed and explore how much the
model output changes due to variations in that single parameter.
Global methods,
on the other hand, allow the entire parameter space to vary simultaneously.
Global methods can therefore identify complex dependencies between the model
parameters in terms of how they affect the model output.

The global methods can be further divided into intrusive and non-intrusive methods.
Intrusive methods require changes to the underlying model equations,
and are often challenging to implement.
Some models in economics are often created with the use of advanced simulators and
modifying the underlying equations of models using these simulators is a
complicated task best avoided.
Non-intrusive methods, on the other hand, consider the model as a black box,
and can be applied to any model without needing to modify the model equations
or implementation.
Global, non-intrusive methods are therefore the method of preference for econsa
because this allows for the economist to isolate the model and its complexities
and focus on the model parameters.
The uncertainty calculations in econsa is based on (TF) which provides
global non-intrusive methods for uncertainty quantification. Sensitivity analysis
is performed by using the extended morris Method to take the efects of
inputs dependence into account by using the variance
decomposition approach from the variance-based GLobal Sensitivity Analysis
methods.

We start by introducing the problem definition. Next, we introduce the statistical
measurements for uncertainty quantification and sensitivity analysis.

.. toctree::
   :maxdepth: 2


   problem
   sa
   uq
