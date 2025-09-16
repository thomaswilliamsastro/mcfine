###############
Advanced Topics
###############

===================
Different fit types
===================

We expect that most users will use LTE modelling, there also
exists options for RADEX and pure Gaussian fitting. These
are accessed by the ``fit_type`` parameter:

.. code-block:: toml

  [fitting_params]
  fit_type = "lte"  # Can be "lte", "radex", or "pure_gauss"

===============
Initial guesses
===============

A significant amount of time is spent in producing the initial
guesses via ``lmfit``. This is because by default we use basinhopping,
which works but is slow. The majority of this is because getting initial
guesses for the velocities is critical, but difficult. We also offer
a method called ``iterative``, which finds peaks in the spectrum using
zero-crossings in the derivatives (via specutils). For the total number
of components, it will iteratively subtract off a model for the flux
(from previous fits in this iterative step), find peaks in the
model-subtracted flux (i.e. any unaccounted for emission), and then fit
all the components simultaneously once again (to avoid biases that can arise
by fitting one component at a time) in a small window around the various peaks.
This generally produces identical results in testing to the default method, but
runs significantly (about a factor 5 for a 4-component fit) faster.

If using ``iterative``, the strong recommendation for the minimization algorith
is ``powell``.

======================
``emcee`` optimization
======================

By default, we use a fixed number of walkers and steps in the emcee run. This
may not be optimal, so we also offer a couple of adaptive methods.

The first is you can adapt the number of walkers to the number of parameters
being fitted. A typical rule of thumb is to use **at least** twice the number
of walkers as parameters. This can be specified in the config file by using
a string for ``n_walkers`` under the ``[mcmc]`` section: if you use something like

.. code-block:: toml

   [mcmc]
   n_walkers = "3*n_params"

then the number of walkers will be adapted to the number of components being fit.
This can effectively speed up the fit and reduce sampler filesizes, without affecting
the final fit quality

The second is using an adaptive number of steps, checking against an autocorrelation criterion.
By default, we use a fixed number of steps, using a quarter of these as "burn-in", before a full
run. Then, when calculating parameters, we will discard the first half of this chain as parameters
may still be moving around. However, it may be more optimal to use an adaptive number of steps,
and tune this to some multiple of the autocorrelation length. More details of this can be found
in `the emcee docs <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>`_. You
can switch to this mode like so:

.. code-block:: toml

   [mcmc]
   emcee_run_method = "adaptive"

this will use another few parameters:

.. code-block:: toml

   [mcmc]
   convergence_factor = 100
   tau_change = 0.01
   thin = 0.5
   burn_in = 2
   max_steps = 100000

The first two control when fitting will stop. We will step out of the MCMC run once the chain
is ``convergence_factor`` times the autocorrelation length, and the change in that factor varies
by less than a factor of ``tau_change``. To ensure the MCMC doesn't run forever, the maximum number
of steps is ``max_steps``.

After this, we then use ``thin`` and ``burn_in`` to control how the sampler is pared down for parameter
estimation. We use a factor of ``burn_in`` time the autocorrelation length to define the burn-in, and then
thin by a factor of ``thin`` times the autocorrelation length to thin out the chains. For more details on
these, see the `emcee docs <https://emcee.readthedocs.io/en/stable/tutorials/monitor/>`_.

============
Space saving
============

In some cases it may be worthwhile to try and save as much space as possible
for the output fit dictionaries. This is especially true for huge datasets
where the total space requirements can ramp up into the TB. By default,
the ``fit_dict`` files will contain both the full ``emcee`` sampler and
information for a covariance matrix. If your corner plots look well-behaved
(i.e. mostly Gaussian), then you can save a **significant** amount of space
by setting ``keep_sampler = false`` in ``[fitting_params]``.

=======================
Configurable parameters
=======================

To see the configurable parameters in McFine, there is an
in-built convenience function:

.. code-block:: python

    from mcfine.utils import print_config_params
    print_config_params()

This will list all the parameters, as well as their type
and default values. You can also see the default values
McFine will use in ``mcfine/toml/`` in the GitHub repository,
which may also be useful for those who aren't too familiar
with toml.

==================
Exploring samplers
==================

Although we expose convenience functions for exploring
the fits (see :doc:`here <tutorials/exploring_cube_fits>`), you
can also directly access the `emcee` sampler object:

.. code-block:: python

    from mcfine.utils import load_fit_dict

    fit_dict = load_fit_dict(file_name)
    sampler = fit_dict["sampler"]

from there, you can mess around with this as you'd like.

===================
Adding another line
===================

It is possible to add other lines to McFine relatively
simply. The majority of the info just needs to be put
into ``line_info.py``. For the LTE case, these are
``v_lines`` and ``strength_lines``. For a single-peak
line, this is just 0 and 1. For RT there's
``transition_lines`` and ``freq_lines``. These should
descend from the RADEX naming scheme. Once you've added
those, include your new line in ``ALLOWED_LINES`` in
``fitting.py``, and edit the config file to use this
new line.

============================
Limiting number of processes
============================

McFine is highly multi-processed, but so are a number of
packages that McFine relies on which can cause issues,
especially on larger machines. To limit the number of threads
packages such as numpy will use, before you call your code you can
put (in the shell):

.. code-block:: shell

    setenv MKL_NUM_THREADS 1
    setenv NUMEXPR_NUM_THREADS 1
    setenv OMP_NUM_THREADS 1
    setenv OPENBLAS_NUM_THREADS 1
    setenv GOTO_NUM_THREADS 1

or the equivalent EXPORT call, depending on the shell you use.
