############################
Tutorial: Fitting a spectrum
############################

We'll now show how to fit a spectrum. This is using the generated spectrum from the
:doc:`generating a spectrum tutorial <generating_a_spectrum>`, but feel free
to try out your own.

McFine operates by reading in TOML configuration files -- one to define local parameters
such as where to save the files, and the other various parameters in the fit. We'll call
these ``local.toml`` and ``config.toml``. A local.toml file will look something like:

.. code-block:: toml

  [local]

  base_dir = "/Users/username/mcfine_fits"

  fit_dir = "fit"
  plot_dir = "plot"

And a config.toml file like:

.. code-block:: toml

  [fitting_params]
  fit_type = "lte"
  fit_method = "mcmc"

  line = "n2hp10"

  [initial_guess]

  lte = [10, 0, -40, 1]

Taking the ``data``, ``vel``, and ``error_spectrum`` from the
:doc:`generating a spectrum tutorial <generating_a_spectrum>`, we can then run the fit very simply:

.. code-block:: python

    from mcfine import HyperfineFitter

    fit_dir = "fit"

    fit_dict_filename = os.path.join(fit_dir, "fit_dict")

    if not os.path.exists(f"{fit_dict_filename}.pkl") or overwrite_fits:
        hf = HyperfineFitter(
            data=spectrum_obs,
            vel=vel,
            error=error_spectrum,
            config_file=config_file,
            local_file=local_file,
        )

        hf.multicomponent_fitter(fit_dict_filename=fit_dict_filename)

Note that for the initial guess step, the algorithm can be quite slow and will not print out
and progress. Rest assured, things are still running! This is the bottleneck in the code and
is the reason why we **strongly** suggest running on a compute cluster.

If you need inspiration, you can print out all parameters and their default parameters
(see :doc:`advanced topics <../advanced_topics>`).
