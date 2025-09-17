##################
Fitting Philosophy
##################

``mcfine`` operates by default in a two-step manner. Firstly, it will try and find an initial solution
using ``lmfit`` (The Initial Fit), before exploring the parameter space more thoroughly using ``emcee``
(The MCMC Fit).

===============
The Initial Fit
===============

Getting a good initial guess is critical for this fitting. ``mcfine`` offers two methods that have similar
accuracy, but different performance. Both of these wrap around ``lmfit``, which provides least-squares minimization.

-------
default
-------

The default method simply tries to fit the spectrum with the number of components supplied. For high-dimensions,
the solver is critical. We recommend ``basinhopping``, which is slow but reliable. This is a "global" minimization
strategy that works well in high dimensions, but is **very slow**. For instance, running this on a 5-parameter
model can take upwards of 5 minutes. As such, when speed is not critical this is probably the optimal choice,
but for large datasets can slow things down significantly.

---------
iterative
---------

Alternatively, to try and sidestep the issue of slow fits in high dimensions, we offer an ``iterative`` method, which
instead uses derivative spectroscopy to estimate where lines are, before fitting them and subtracting them off to
try and find the position of the next line (this is the iterative part). As the complexity increases, the components
are fit simultaneously to try and avoid degeneracies. Because this means the initial velocity guess is better, this
tends to be better behaved and so a faster least-square minimizer can be used. This means this method scales
significantly better than the default with basinhopping - around three times faster. Here, we recommend the ``powell``
minimizer, which works well with noisy data.

============
The MCMC Fit
============

Having got initial parameters from ``lmfit``, we now move onto the thing that sets ``mcfine`` apart: the MCMC fitting.
We again take an iterative approach here, performing a number of "burn-in" runs (the number is configurable, but two
seems to work well) where we distribute the MCMC walkers around the maximum likelihood solution (this either comes
from ``lmfit`` or a previous MCMC burn-in run). These are distributed normally, with a standard deviation of 1% of
the value for the first initialisation run, and then a much tighter ball for subsequent runs. We then run for either
a fixed number of iterations (``fixed``) or an adaptive number where the autocorrelation time is used to decide when
to stop (``adaptive``). The fixed method is generally much faster, but may produce suboptimal results (particularly at
high complexity). We recommend users experiment to see what works best for them.

Following the fitting, parameters are estimated based on a certain chunk of the MCMC chain. We take a multiple of
the autocorrelation time (default: 2) as burn-in, and thin by a multiple of the autocorrelation time (default: 0.5).
This ensures the points are independent and the walkers have settled in. You can check this is sensible by looking
at the diagnostic plots we provide functions to produce.



