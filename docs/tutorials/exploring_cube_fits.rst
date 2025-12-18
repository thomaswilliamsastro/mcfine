#############################
Tutorial: Exploring Cube Fits
#############################

Having fit our cube, we can pull out a bunch of parameters and have a look at the data. Following the
:doc:`previous tutorial <cube_fitting>`, we have an ``hf_fitter`` object, that contains the maps,
which is a dictionary that we can pull things out of:

.. code-block:: python

    maps = hf_fitter.parameter_maps

    # Get the reduced chi-square if it's a numpy array
    chisq_red = maps["chisq_red"]

    # Or if you've input a fits file to start with, then this will be a fits image
    chisq_red = maps["chisq_red"].data

You can then plot this up however you might like:

.. image:: images/cube_chisq.png

The fitted parameters (and errors) are also in here as ``[param]_[comp_no]``, ``[param]_[comp_no]_err_up``, and
``[param]_[comp_no]_err_down``. Note these don't take into account the covariances, if you want to properly sample
for the purposes of a plot then you'll need the full set of walkers, which can be saved in individual fit dictionary
files.

You can also plot things in much the same way as in the :doc:`single spectra plotting <plotting_a_spectrum>` way,
except you can also provide a grid to cut down the number of plots created. For example:

.. code-block:: python

    plot_dir = "plots"

    grid = np.zeros_like(chisq_red)

    grid[10, 10] = 1
    grid[50, 50] = 1

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    mcfine_fitting_filename = os.path.join(
            fit_dir, f"mcfine_fit_backward",
        )

    hf_plotter.plot_step(plot_name=os.path.join(fit_dir, plot_dir, f"{target}_step"),
                         mcfine_output_filename=mcfine_output_filename,
                         #fit_dict_filename=os.path.join(coherence_backward_dir, fit_dict_filename),
                         n_comp_filename=os.path.join(coherence_backward_dir, n_comp_filename),
                         grid=grid)
    hf_plotter.plot_corner(plot_name=os.path.join(plot_dir, f"{target}_corner"),
                           mcfine_output_filename=mcfine_output_filename,
                           #fit_dict_filename=os.path.join(coherence_backward_dir, fit_dict_filename),
                           grid=grid)
    hf_plotter.plot_fit(plot_name=os.path.join(plot_dir, f"{target}"),
                        mcfine_output_filename=mcfine_output_filename,
                        #fit_dict_filename=os.path.join(coherence_backward_dir, fit_dict_filename),
                        grid=grid)

.. image:: images/cube_fit.png

Note here that you can either pass the full fit dictionary (``mcfine_output_filename``), or each individual fit
dictionary (``fit_dict_filename``).
