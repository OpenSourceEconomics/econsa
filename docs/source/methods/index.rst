Methods
=======

Here we explain and document in detail, the methods we implement in the ``econsa`` package to perform sensitivity analysis and uncertainty quantification. An insight into how the calculations are performed is not a prerequisite for using ``econsa``, in most cases, the default settings works fine. Global Sensitivity Analysis can be divided into two categories: quali- and quantitative methods. ``econsa`` implements both methods as 
a comprehensive to ensure flexibility depending on your model requirements, features
and specifications. 

The Elementary Effects (EE), also known as the Morris method, is a qualitative way to screen inputs and helps to determine the set of influential and non-influential inputs. Shapely values on the other hand, ...

.. todo::

  Add information on Shapely values and UQ methods implemented in ``econsa``

.. todo::

  Here we want to distinguish between qualitative approaches such as Morris method and quantitative approaches such as our integration of Shapely values.

.. toctree::
   :maxdepth: 1

   morris
