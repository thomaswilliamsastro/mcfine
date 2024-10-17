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
  fit_type = 'lte'  # Can be 'lte', 'radex', or 'pure_gauss'

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

    with open(file_name, 'rb') as f:
        fit_dict = pickle.load(f)
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
