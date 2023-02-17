############################
Tutorial: Fitting a spectrum
############################

We'll now show how to fit a spectrum. This is using the generated spectrum from the
:doc:`generating a spectrum tutorial <tutorial_generating_a_spectrum>`, but feel free
to try out your own.

McFine operates by reading in TOML configuration files -- one to define local parameters
such as where to save the files, and the other various parameters in the fit. We'll call
these ``local.toml`` and ``config.toml``. A local.toml file will look something like:

.. code-block:: toml

    something = 'hello'

And a config.toml file like:

.. code-block:: toml

    config = 'config'

If you need inspiration, the default .toml files are given in the code directory (mcfine/toml).
