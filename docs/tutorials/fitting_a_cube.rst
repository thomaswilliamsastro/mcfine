########################
Tutorial: Fitting a Cube
########################

Fitting a cube is a little more complicated than fitting a single spectrum, but has been set up to be
reasonably flexible and simple to interface with. First, we need to load in the config files

.. code-block:: python

   import tomllib

   config_file = "config.toml"
   local_file = "local.toml"

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    with open(local_file, "rb") as f:
        local = tomllib.load(f)

Next, a few filenames for various output files. We need the input cube, as well as files for the output parameter maps,
fit parameter dictionaries, and fit cube. We also have an error cube and a mask cube that we'll load in here, but these
are not strictly necessary. We also set up some directories for the individual fits to go into at this point.

.. code-block:: python

    target = "g316_75"

    fit_dict_filename = f"{target}_fit_dict"
    n_comp_filename = f"{target}_n_comp"
    likelihood_filename = f"{target}_likelihood"
    fit_cube_filename = f"{target}_fit_cube"
    hdu_name = f"{target}.fits"

    original_fit_dir = "original"
    coherence_forward_dir = "coherence_forward"
    coherence_backward_dir = "coherence_backward"

    err_hdu_name = hdu_name.replace(".fits", "_noise.fits")
    mask_hdu_name = hdu_name.replace(".fits", "_mask.fits")

    # Pull out various data we need from HDUs
    with fits.open(hdu_name) as hdu, fits.open(err_hdu_name) as err_hdu, fits.open(mask_hdu_name) as mask_hdu:

       data = hdu[0].data
       err = err_hdu[0].data
       mask = mask_hdu[0].data

       # Also get velocity informations
       vel_delt = hdu[0].header["CDELT3"]
       vel_val = hdu[0].header["CRVAL3"]
       vel_pix = hdu[0].header["CRPIX3"]

       velocity = np.array([vel_val + (i - (vel_pix - 1)) * vel_delt for i in range(hdu.data.shape[0])])
       velocity /= 1e3

We'll set the data up to be read properly into ``mcfine`` now. This involves generating error maps and a mask of
pixels to fit:

.. code-block:: python

    # Add in a calibration uncertainty of 10%
    calibration_uncertainty = 0.1
    err = np.sqrt(err ** 2 + (calibration_uncertainty * data) ** 2)

    # We'll use a pretty unrestrictive mask, which is just any spaxel that has a pixel included in the strict mask
    mask = np.nansum(mask, axis=0)
    mask[mask < 1] = 0
    mask[mask >= 1] = 1

    nan_mask = np.where(mask == 0)

We can now throw this all into ``mcfine``!

.. code-block:: python

    from mcfine import HyperfineFitter

    hf_fitter = HyperfineFitter(data=data,
                                vel=velocity,
                                error=err,
                                mask=mask,
                                config_file=config_file,
                                local_file=local_file,
                                )

We start with a first pass through, fitting all pixels defined by our mask:

.. code-block:: python

    print("First-pass fitting")
    hf_fitter.multicomponent_fitter(fit_dict_filename=os.path.join(original_fit_dir, fit_dict_filename),
                                    n_comp_filename=os.path.join(original_fit_dir, n_comp_filename),
                                    likelihood_filename=os.path.join(original_fit_dir, likelihood_filename),
                                    )

This will take a while if you have a lot of fits to do! Go and enjoy your weekend. After this is done, we will
perform a coherence pass forwards and backwards. This has the effect of removing potentially bad fits by comparing
with neighbours, but typically will only replace 10% or less of the fits

.. code-block:: python

    print("Spatial coherence forwards")
    hf_fitter.encourage_spatial_coherence(fit_dict_filename=fit_dict_filename,
                                          input_dir=original_fit_dir,
                                          output_dir=coherence_forward_dir,
                                          n_comp_filename=n_comp_filename,
                                          likelihood_filename=likelihood_filename,
                                          )
    print("Spatial coherence backwards")
    hf_fitter.encourage_spatial_coherence(fit_dict_filename=fit_dict_filename,
                                          input_dir=coherence_forward_dir,
                                          output_dir=coherence_backward_dir,
                                          n_comp_filename=n_comp_filename,
                                          likelihood_filename=likelihood_filename,
                                          reverse_direction=True,
                                          )

Following this, fitting is complete! We will now generate parameter maps and the fit cube

.. code-block:: python

    print("Creating maps")
    hf_fitter.make_parameter_maps(n_comp_filename=os.path.join(coherence_backward_dir, n_comp_filename),
                                  fit_dict_filename=os.path.join(coherence_backward_dir, fit_dict_filename),
                                  maps_filename=f"{target}_maps.pkl",
                                  )

    print("Creating fit cubes")
    hf_fitter.create_fit_cube(fit_dict_filename=os.path.join(coherence_backward_dir, fit_dict_filename),
                              n_comp_filename=os.path.join(coherence_backward_dir, n_comp_filename),
                              cube_filename=fit_cube_filename)

This is now everything done! We can now explore the cube in :doc:`exploring cube fits <exploring_cube_fits>`
