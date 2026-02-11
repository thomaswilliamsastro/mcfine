import copy
import filecmp
import glob
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import sys
import tomllib
import warnings
from functools import partial

import emcee
import numpy as np
import specutils
import xarray
from astropy import units as u
from astropy.io import fits
from lmfit import minimize, Parameters
from scipy.interpolate import RegularGridInterpolator
from spectral_cube import SpectralCube
from specutils import Spectrum
from specutils.fitting import find_lines_derivative
from threadpoolctl import threadpool_limits
from tqdm import tqdm

# Suppress specutils warning about continuum level
specutils.conf.do_continuum_function_check = False

NDRADEX_IMPORTED = False
try:
    import ndradex

    NDRADEX_IMPORTED = True
except ModuleNotFoundError:
    pass

from .emcee_funcs import (
    get_samples,
    get_samples_from_fit_dict,
)
from .fitting_funcs import (
    chi_square,
    ln_prior,
)
from .line_info import (
    allowed_lines,
    transition_lines,
    freq_lines,
    strength_lines_dict,
    v_lines_dict,
)
from .line_shape_funcs import (
    hyperfine_structure_lte,
    hyperfine_structure_pure_gauss,
    hyperfine_structure_radex,
)
from .radex_funcs import get_nearest_values
from .utils import (
    get_dict_val,
    check_overwrite,
    save_pkl,
    load_pkl,
    CONFIG_DEFAULT_PATH,
    LOCAL_DEFAULT_PATH,
)
from .vars import (
    T_BACKGROUND,
    ALLOWED_LMFIT_METHODS,
    ALLOWED_EMCEE_RUN_METHODS,
    ALLOWED_FIT_TYPES,
    ALLOWED_FIT_METHODS,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s - %(message)s",
)
logger = logging.getLogger("mcfine")

# Define global variables for potentially huge arrays, and various config values
glob_data = np.array([])
glob_error = np.array([])
glob_vel = np.array([])

glob_downsampled_data = np.array([])
glob_downsampled_error = np.array([])

glob_initial_n_comp = np.array([])

glob_mcfine_output = {}

radex_grid = xarray.Dataset()

glob_config = {}

mp.set_start_method("fork")


def multiple_components(
    theta,
    vel,
    strength_lines,
    v_lines,
    props,
    n_comp,
    fit_type="lte",
    log_tau=True,
):
    """Sum intensities for multiple lines.

    Takes `n_comp` distinct lines, and calculates the total intensity of their various hyperfine lines.

    Args:
        theta: [t_ex, tau, vel, vel_width] for each component. Should have a length of 4*`n_comp`
        vel: velocity array
        strength_lines: Array for relative line strengths
        v_lines: Array of relative velocity shifts for the lines
        props: List of properties for the line profiles
        n_comp (int): Number of distinct components to calculate intensities for
        fit_type (str): Should be one of ALLOWED_FIT_TYPES. Defaults to 'lte'
        log_tau (bool): Whether to fit tau in log or linear space. Defaults to True (log space)

    Returns:
        np.ndarray: The sum of the intensities for all the distinct components.

    """

    prop_len = len(props)

    if n_comp == 0:
        return np.zeros_like(vel)

    if fit_type == "lte":

        intensity_model = np.array(
            [
                hyperfine_structure_lte(
                    *theta[prop_len * i : prop_len * i + prop_len],
                    strength_lines,
                    v_lines,
                    vel,
                    log_tau=log_tau,
                )
                for i in range(n_comp)
            ]
        )

    elif fit_type == "pure_gauss":

        intensity_model = np.array(
            [
                hyperfine_structure_pure_gauss(
                    *theta[prop_len * i : prop_len * i + prop_len],
                    strength_lines,
                    v_lines,
                    vel,
                )
                for i in range(n_comp)
            ]
        )

    elif fit_type == "radex":

        qn_ul = np.array(range(len(radex_grid["QN_ul"].values)))

        intensity_model = [
            get_radex_multiple_components(
                theta[prop_len * i : prop_len * i + prop_len], vel, v_lines, qn_ul
            )
            for i in range(n_comp)
        ]

    else:

        raise Warning(f"fit type {fit_type} not understood!")

    intensity_model = np.sum(intensity_model, axis=0)

    return intensity_model


def get_radex_multiple_components(
    theta,
    vel,
    v_lines,
    qn_ul,
):
    """Wrapper around RADEX to get out the multiple components"""

    # Important point here, RADEX uses a square profile so transform the sigma into the right width. Pull out
    # the subset of data around our values

    tau, t_ex = radex_grid_interp(theta, qn_ul)

    intensity_model = hyperfine_structure_radex(
        t_ex, tau, theta[3], theta[4], v_lines, vel
    )

    return intensity_model


def radex_grid_interp(
    theta,
    qn_ul,
    labels=None,
):
    """Interpolate generated RADEX grid to get useful values out without weird edge effects

    Args:
        theta (list): property values
        qn_ul (list): Names for the transitions
        labels (list): Labels for properties

    Returns:
        tau and t_ex
    """

    if labels is None:
        labels = [
            "T_kin",
            "N_mol",
            "n_H2",
            "dv",
        ]

    nearest_values = get_nearest_values(
        radex_grid, labels, [theta[0], 10 ** theta[1], 10 ** theta[2], theta[4] * 2.355]
    )

    # Pull out grid subset of the nearest values
    grid_subset = radex_grid.sel(
        T_kin=nearest_values[0],
        N_mol=nearest_values[1],
        n_H2=nearest_values[2],
        dv=nearest_values[3],
    )
    tau_subset = grid_subset["tau"].values
    t_ex_subset = grid_subset["T_ex"].values

    # Remove any singular (limit) values
    limit_vals = [
        True if type(nearest_value) == np.ndarray else False
        for nearest_value in nearest_values
    ]

    theta_coords = np.array(
        [theta[0], 10 ** theta[1], 10 ** theta[2], theta[4] * 2.355]
    )
    theta_coords = theta_coords[limit_vals]

    # Also remove the singular dimensions from the nearest values in the grid
    nearest_values = np.asarray(nearest_values)[limit_vals]

    # Construct an array to get values for each point in the grid
    coords = np.array([(value, *theta_coords) for value in qn_ul])

    # Setup grid and fast interpolate
    grid_coords = (qn_ul, *nearest_values)
    reg_grid_tau = RegularGridInterpolator(grid_coords, tau_subset)
    reg_grid_t_ex = RegularGridInterpolator(grid_coords, t_ex_subset)

    tau = reg_grid_tau(coords)
    t_ex = reg_grid_t_ex(coords)

    return tau, t_ex


def initial_lmfit(
    params,
    intensity,
    intensity_err,
    vel,
    strength_lines,
    v_lines,
    props,
    n_comp=1,
    fit_type="lte",
    log_tau=True,
):
    """Get initial guess for MC walkers from lmfit

    This is the residual calculation for LMFIT. Just to get our initial
    parameters to instantiate the MC

    Args:
        params: List of lmfit parameters
        intensity: Measured intensity
        intensity_err: Measured error on intensity
        vel: Velocity axis
        strength_lines: Relative line strength
        v_lines: Relative line velocity
        props: List of properties for the line profiles
        n_comp: Number of components to fit. Defaults to 1
        fit_type: Type of fit to do. Either 'lte' or 'radex'. Defaults to 'lte'
        log_tau: Whether to fit tau in log-space (default) or linear space

    Returns:
        Residual (chisq) value from the fit
    """

    theta = np.array([params[key].value for key in params])

    intensity_model = multiple_components(
        theta=theta,
        vel=vel,
        strength_lines=strength_lines,
        v_lines=v_lines,
        props=props,
        n_comp=n_comp,
        fit_type=fit_type,
        log_tau=log_tau,
    )
    diff = intensity - intensity_model
    residuals = diff / intensity_err

    return residuals


def ln_like(
    theta,
    intensity,
    intensity_err,
    vel,
    strength_lines,
    v_lines,
    props,
    n_comp=1,
    fit_type="lte",
    log_tau=True,
):
    """Calculate the (negative) log-likelihood for the model

    This is the main meat of the MCMC fitting, given the
    parameters we calculate the likelihood of the model
    and return that

    Args:
        theta: Parameters for the fit
        intensity: Observed intensity
        intensity_err: Intensity uncertainty
        vel: Observed velocity
        strength_lines: Relative line strength
        v_lines: Relative line velocities
        props: List of properties being fit
        n_comp: Number of components to fit. Defaults to 1
        fit_type: Either 'lte' (default) or 'radex'
        log_tau: Whether to fit tau in log-space (default) or
            linear space

    Returns:
        Negative ln-likelihood for the model.
    """

    intensity_model = multiple_components(
        theta,
        vel,
        strength_lines,
        v_lines,
        props,
        n_comp=n_comp,
        fit_type=fit_type,
        log_tau=log_tau,
    )
    chisq = chi_square(intensity, intensity_model, intensity_err)

    # # Scale the chisq by sqrt(2[N-P]), following Smith+ and others
    # p = n_comp * len(props)
    # n = len(intensity[~np.isnan(intensity)])
    # scale_factor = np.sqrt(2 * (n - p))
    # chisq /= scale_factor

    return -0.5 * chisq


def ln_prob(
    theta,
    intensity,
    intensity_err,
    vel,
    strength_lines,
    v_lines,
    props,
    bounds,
    n_comp=1,
    fit_type="lte",
):
    """Calculate the ln-probability for emcee

    This combines the prior (generally flat) with the ln-likelihood
    to return to emcee

    Args:
        theta: Parameters for the fit
        intensity: Observed intensity
        intensity_err: Intensity uncertainty
        vel: Observed velocity
        strength_lines: Relative line strength
        v_lines: Relative line velocities
        props: List of properties being fit
        bounds: Bounds on parameters
        n_comp: Number of components to fit. Defaults to 1
        fit_type: Either 'lte' (default) or 'radex'

    Returns:
        float: Combined probability (likelihood+prior) for the model.

    """
    lp = ln_prior(theta, vel, props, bounds, n_comp=n_comp)
    if not np.isfinite(lp):
        return -np.inf
    like = ln_like(
        theta,
        intensity,
        intensity_err,
        vel,
        strength_lines,
        v_lines,
        props,
        n_comp=n_comp,
        fit_type=fit_type,
    )
    return lp + like


def calculate_goodness_of_fit(
    data,
    error,
    best_fit_pars,
    n_comp,
):
    """Calculate various goodness of fit metrics."""

    m = len(data[~np.isnan(data)])
    ln_m = np.log(m)
    if best_fit_pars is None:
        k = 0
    else:
        k = len(best_fit_pars)

    if n_comp > 0:
        total_model = multiple_components(
            theta=best_fit_pars,
            vel=glob_vel,
            strength_lines=glob_config["strength_lines"],
            v_lines=glob_config["v_lines"],
            props=glob_config["props"],
            n_comp=n_comp,
            fit_type=glob_config["fit_type"],
        )
    else:
        total_model = np.zeros_like(data)

    chisq = chi_square(
        data,
        total_model,
        error,
    )

    deg_freedom = m - k
    chisq_red = chisq / deg_freedom

    # log-likelihood, BIC, AIC from chi-square
    likelihood = -0.5 * chisq
    bic = k * ln_m - 2 * likelihood
    aic = 2 * k - 2 * likelihood

    goodness_of_fit_metrics = {
        "likelihood": likelihood,
        "bic": bic,
        "aic": aic,
        "chisq": chisq,
        "deg_freedom": deg_freedom,
        "chisq_red": chisq_red,
    }

    return goodness_of_fit_metrics


def sample_tpeak_per_component(
    samples,
    n_comp,
    n_samples=500,
):
    """Get a number of Tpeak samples per component"""

    tpeak = np.zeros([n_samples, n_comp])
    idx_offset = len(glob_config["props"])
    for i in range(n_samples):
        choice_idx = np.random.randint(0, samples.shape[0])
        tpeak_i = [
            np.nanmax(
                multiple_components(
                    theta=samples[
                        choice_idx, idx_offset * j : idx_offset * j + idx_offset
                    ],
                    vel=glob_vel,
                    strength_lines=glob_config["strength_lines"],
                    v_lines=glob_config["v_lines"],
                    props=glob_config["props"],
                    n_comp=1,
                    fit_type=glob_config["fit_type"],
                )
            )
            for j in range(n_comp)
        ]

        tpeak[i, :] = copy.deepcopy(tpeak_i)

    return tpeak


def downsample(
    data,
    chunk_size=10,
    func=np.nanmean,
):
    """Downsample data, given a function

    Args:
        data: Data to downsample. Should be an image
        chunk_size: Chunk size to downsample to
        func: Function to downsample using. Defaults to
            nanmean
    """

    if data.ndim == 2:
        axes = (0, 1)
    elif data.ndim == 3:
        axes = (1, 2)
    else:
        raise ValueError("Input to downsample should be a 2D or 3D array")

    ii = np.arange(chunk_size, data.shape[axes[0]], chunk_size)
    jj = np.arange(chunk_size, data.shape[axes[1]], chunk_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        downsampled_array = np.array(
            [
                [
                    func(i_split, axis=axes)
                    for i_split in np.array_split(j_split, ii, axis=axes[0])
                ]
                for j_split in np.array_split(data, jj, axis=axes[1])
            ]
        ).T

    return downsampled_array


def parallel_fitting(
    ij,
    fit_dict_filename="fit_dict",
    data_type="original",
    fit_method="mcmc",
    overwrite=False,
):
    """Parallel function for MCMC fitting.

    Wraps up the MCMC fitting to pass off to multiple cores. Because of overheads, it's easier to farm out multiple
    fits to multiple cores, rather than run the MCMC with multiple threads.

    Args:
        ij (tuple): tuple containing (i, j) coordinates of the pixel to fit.
        fit_dict_filename (str): Base filename for MCMC ft pickle. Will append coordinates on afterward. Defaults
            to fit_dict.
        data_type: Data type to fit. Can be either "original" or "downsampled"
        fit_method: "mcmc" or "leastsq". "mcmc" will run emcee
            for each component and use fit parameters from that,
            whereas "leastsq" will only run an MCMC after a
            final fit has been found, to explore covariances.
            Defaults to "mcmc".
        overwrite (bool): Overwrite existing files? Defaults to False.

    Returns:
        Number of fitted components and the best-fit likelihood.
    """

    if data_type not in ["original", "downsampled"]:
        raise ValueError("Data type to fit should be original or downsampled")

    i = ij[0]
    j = ij[1]

    cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"

    if not os.path.exists(cube_fit_dict_filename) or overwrite:
        logger.debug(f"Fitting {i}, {j}")

        if data_type == "original":
            data = glob_data[:, i, j]
            error = glob_error[:, i, j]
        elif data_type == "downsampled":
            data = glob_downsampled_data[:, i, j]
            error = glob_downsampled_error[:, i, j]
        else:
            raise ValueError("Data type to fit should be original or downsampled")

        # If somehow we've broken the logic here, then freak out
        if data.size == 0:
            raise ValueError("No data found!")

        initial_n_comp = None
        if glob_initial_n_comp.size != 0:
            initial_n_comp = glob_initial_n_comp[i, j]

        # Limit to a single core to avoid weirdness
        with threadpool_limits(limits=1, user_api=None):
            fit_dict = delta_bic_looper(
                data=data,
                error=error,
                fit_method=fit_method,
                initial_n_comp=initial_n_comp,
            )

        if not glob_config["keep_sampler"]:
            fit_dict.pop("sampler")

        if not glob_config["keep_covariance"]:
            fit_dict.pop("cov_matrix")
            fit_dict.pop("cov_med")

        save_pkl(fit_dict, cube_fit_dict_filename)

    return cube_fit_dict_filename


def delta_bic_looper(
    data,
    error,
    fit_method="mcmc",
    initial_n_comp=None,
    save=False,
    overwrite=True,
    progress=False,
):
    """Increase spectral complexity until we hit diminishing returns

    This will iteratively build up the spectral model one component
    at a time, before looping backwards to remove the weaker components
    which might just be fits to the noise

    Args:
        data: Observed data
        error: Observed error
        fit_method: "mcmc" or "leastsq". "mcmc" will run emcee
            for each component and use fit parameters from that,
            whereas "leastsq" will only run an MCMC after a
            final fit has been found, to explore covariances.
            Defaults to "mcmc".
        initial_n_comp: Initial guess at number of components.
            Defaults to None.
        save: Whether to save out or not, defaults to False
        overwrite: Whether to overwrite existing fits. Defaults to False
        progress: Whether to display progress bars or not. Defaults to False

    Returns:
        fitted number of components, likelihood, and the emcee sampler
    """

    delta_bic = np.inf
    delta_aic = np.inf
    parameter_median_old = None
    sampler_old = None
    sampler = None
    best_fit_pars = None
    best_fit_errs = None
    cov_matrix = None
    cov_med = None
    prop_len = len(glob_config["props"])

    if initial_n_comp is None:
        # We start with a zero component model, i.e. a flat line
        n_comp = 0
        parameter_median = None

        gof = calculate_goodness_of_fit(
            data=data,
            error=error,
            best_fit_pars=parameter_median,
            n_comp=n_comp,
        )
        bic = gof["bic"]
        aic = gof["aic"]

    else:
        # Start with an initial guess of component numbers
        n_comp = int(initial_n_comp)

        # If we're doing MCMC
        if fit_method == "mcmc":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sampler = run_mcmc(
                    data,
                    error,
                    n_comp=n_comp,
                    save=save,
                    overwrite=overwrite,
                    progress=progress,
                )

            # Get flat samples and calculate median parameters
            flat_samples = get_samples(
                sampler,
                burn_in_frac=glob_config["burn_in"],
                thin_frac=glob_config["thin"],
            )
            parameter_median = np.nanmedian(
                flat_samples,
                axis=0,
            )

        # Do a leastsq fit
        elif fit_method == "leastsq":
            parameter_median = get_p0_lmfit(
                data,
                error,
                n_comp=n_comp,
            )

        else:
            raise ValueError(f"fit_method should be one of {ALLOWED_FIT_METHODS}")

        # Calculate goodness of fit, pull out likelihood/BIC/AIC
        gof = calculate_goodness_of_fit(
            data=data,
            error=error,
            best_fit_pars=parameter_median,
            n_comp=n_comp,
        )
        bic = gof["bic"]
        aic = gof["aic"]

    parameter_median_old = parameter_median
    bic_old = bic
    aic_old = aic

    while (
        delta_bic > glob_config["delta_bic_cutoff"]
        or delta_aic > glob_config["delta_aic_cutoff"]
    ):
        # Store the previous BIC and sampler, since we need them later
        parameter_median_old = parameter_median
        sampler_old = sampler
        bic_old = bic
        aic_old = aic

        # Increase the number of components, refit
        n_comp += 1

        # If we're doing MCMC
        if fit_method == "mcmc":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sampler = run_mcmc(
                    data,
                    error,
                    n_comp=n_comp,
                    save=save,
                    overwrite=overwrite,
                    progress=progress,
                )

            # Get flat samples and calculate median parameters
            flat_samples = get_samples(
                sampler,
                burn_in_frac=glob_config["burn_in"],
                thin_frac=glob_config["thin"],
            )
            parameter_median = np.nanmedian(
                flat_samples,
                axis=0,
            )

        # Do a leastsq fit
        elif fit_method == "leastsq":
            parameter_median = get_p0_lmfit(
                data,
                error,
                n_comp=n_comp,
            )

        else:
            raise ValueError(f"fit_method should be one of {ALLOWED_FIT_METHODS}")

        # Calculate goodness of fit, pull out likelihood/BIC/AIC
        gof = calculate_goodness_of_fit(
            data=data,
            error=error,
            best_fit_pars=parameter_median,
            n_comp=n_comp,
        )
        bic = gof["bic"]
        aic = gof["aic"]

        delta_bic = bic_old - bic
        delta_aic = aic_old - aic

    # Move back to the previous values
    parameter_median = parameter_median_old
    sampler = sampler_old
    bic = bic_old
    aic = aic_old
    n_comp -= 1

    # Now loop backwards, iteratively remove the weakest component and refit. Only if we have a >0 order fit!

    if n_comp > 0:
        max_back_loops = n_comp

        for i in range(max_back_loops):

            if glob_config["fit_type"] == "lte":
                line_intensities = np.array(
                    [
                        hyperfine_structure_lte(
                            *parameter_median[prop_len * i : prop_len * i + prop_len],
                            strength_lines=glob_config["strength_lines"],
                            v_lines=glob_config["v_lines"],
                            vel=glob_vel,
                        )
                        for i in range(n_comp)
                    ]
                )

            elif glob_config["fit_type"] == "pure_gauss":
                line_intensities = np.array(
                    [
                        hyperfine_structure_pure_gauss(
                            *parameter_median[prop_len * i : prop_len * i + prop_len],
                            strength_lines=glob_config["strength_lines"],
                            v_lines=glob_config["v_lines"],
                            vel=glob_vel,
                        )
                        for i in range(n_comp)
                    ]
                )

            elif glob_config["fit_type"] == "radex":
                line_intensities = np.array(
                    [
                        hyperfine_structure_radex(
                            *parameter_median[prop_len * i : prop_len * i + prop_len],
                            v_lines=glob_config["v_lines"],
                            vel=glob_vel,
                        )
                        for i in range(n_comp)
                    ]
                )
            else:
                logger.warning(f"Fit type {glob_config['fit_type']} not understood!")
                sys.exit()

            integrated_intensities = np.trapezoid(line_intensities, x=glob_vel, axis=-1)
            component_order = np.argsort(integrated_intensities)

            bic_old = bic
            aic_old = aic
            parameter_median_old = parameter_median
            sampler_old = sampler

            # Remove the weakest component
            n_comp -= 1

            if n_comp == 0:
                parameter_median = None
                gof = calculate_goodness_of_fit(
                    data=data,
                    error=error,
                    best_fit_pars=parameter_median,
                    n_comp=n_comp,
                )
                bic = gof["bic"]
                aic = gof["aic"]
            else:
                idx_to_delete = range(
                    prop_len * component_order[0],
                    prop_len * component_order[0] + prop_len,
                )
                p0 = np.delete(parameter_median, idx_to_delete)

                # If we're doing MCMC
                if fit_method == "mcmc":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sampler = run_mcmc(
                            data,
                            error,
                            n_comp=n_comp,
                            save=save,
                            overwrite=overwrite,
                            progress=progress,
                            p0_fit=p0,
                        )

                    # Get flat samples and calculate median parameters
                    flat_samples = get_samples(
                        sampler,
                        burn_in_frac=glob_config["burn_in"],
                        thin_frac=glob_config["thin"],
                    )
                    parameter_median = np.nanmedian(flat_samples, axis=0)

                # If doing a leastsq fit, we already have p0
                elif fit_method == "leastsq":
                    parameter_median = p0

                else:
                    raise ValueError(
                        f"fit_method should be one of {ALLOWED_FIT_METHODS}"
                    )

                # Calculate goodness of fit, pull out likelihood/BIC/AIC
                gof = calculate_goodness_of_fit(
                    data=data,
                    error=error,
                    best_fit_pars=parameter_median,
                    n_comp=n_comp,
                )
                bic = gof["bic"]
                aic = gof["aic"]

            delta_bic = bic_old - bic
            delta_aic = aic_old - aic

            # If removing and refitting doesn't significantly improve things, then just jump out of here
            if (
                delta_bic < glob_config["delta_bic_cutoff"]
                and delta_aic < glob_config["delta_aic_cutoff"]
            ):
                break

        # Revert to previous sampler/n_comp
        parameter_median = parameter_median_old
        sampler = sampler_old
        n_comp += 1

    # Finally, if we've been using leastsq method and we have fitted
    # components, then run an MCMC here
    if fit_method == "leastsq" and n_comp > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = run_mcmc(
                data,
                error,
                n_comp=n_comp,
                save=save,
                overwrite=overwrite,
                progress=progress,
                p0_fit=parameter_median,
            )

    # Calculate the covariance matrix and the parameter medians/errors.
    # This is only defined if we end up with a 0-component fit

    tpeak_percentiles = None
    tpeak_diff = None

    if sampler is not None:
        samples = get_samples(
            sampler,
            burn_in_frac=glob_config["burn_in"],
            thin_frac=glob_config["thin"],
        )
        cov_matrix = np.cov(samples.T)
        cov_med = np.median(samples, axis=0)

        param_percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        best_fit_pars = param_percentiles[1, :]
        best_fit_errs = np.diff(param_percentiles, axis=0)

        # Calculate Tpeak with errors from the samples
        tpeak = sample_tpeak_per_component(
            samples=samples,
            n_comp=n_comp,
        )

        tpeak_percentiles = np.percentile(tpeak, [16, 50, 84], axis=0)
        tpeak_diff = np.diff(tpeak_percentiles, axis=0)

    # Calculate goodness-of-fit parameters
    goodness_of_fit_metrics = calculate_goodness_of_fit(
        data=data,
        error=error,
        best_fit_pars=best_fit_pars,
        n_comp=n_comp,
    )

    # Put everything into one big ol' dictionary
    fit_dict = {
        "props": {},
        "props_err_down": {},
        "props_err_up": {},
        "n_comp": n_comp,
        "sampler": sampler,
        "cov_matrix": cov_matrix,
        "cov_med": cov_med,
    }
    fit_dict.update(goodness_of_fit_metrics)

    # Put in Tpeak/Tpeak errors
    if n_comp > 0:
        fit_dict["props"]["tpeak"] = tpeak_percentiles[1, :]
        fit_dict["props_err_down"]["tpeak"] = tpeak_diff[0, :]
        fit_dict["props_err_up"]["tpeak"] = tpeak_diff[1, :]

    # Put in property and property errors
    arr = np.zeros(n_comp)
    for i in range(n_comp):
        for prop_idx, prop in enumerate(glob_config["props"]):
            param_idx = i * len(glob_config["props"]) + prop_idx

            if prop not in fit_dict["props"]:
                fit_dict["props"][prop] = copy.deepcopy(arr)
                fit_dict["props_err_down"][prop] = copy.deepcopy(arr)
                fit_dict["props_err_up"][prop] = copy.deepcopy(arr)

            fit_dict["props"][prop][i] = best_fit_pars[param_idx]
            fit_dict["props_err_down"][prop][i] = best_fit_errs[0, param_idx]
            fit_dict["props_err_up"][prop][i] = best_fit_errs[1, param_idx]

    return fit_dict


def run_mcmc(
    data,
    error,
    n_comp=1,
    save=True,
    overwrite=False,
    progress=False,
    fit_dict_filename="fit_dict.pkl",
    p0_fit=None,
):
    """Run emcee to get a fit out

    Args:
        data: Observed data
        error: Observed uncertainty
        n_comp: Number of components to fit. Defaults to 1
        save: Whether to save the sampler out. Defaults to True
        overwrite: Whether to overwrite existing fit. Defaults to False
        progress: Whether to display progress bar. Defaults to False
        fit_dict_filename: Name for the sampler. Defaults to fit_dict.pkl
        p0_fit: Initial guess for the fit. Defaults to None, which will
            use some basic parameters
    """

    if not os.path.exists(fit_dict_filename) or overwrite:

        # If we don't have p0_fit, get from lmfit
        if p0_fit is None:
            p0_fit = get_p0_lmfit(
                data=data,
                error=error,
                n_comp=n_comp,
            )

        # Ensure we have an array here
        if not isinstance(p0_fit, np.ndarray):
            p0_fit = np.array(p0_fit)

        n_dims = len(p0_fit)

        # If we have an adaptive number of walkers, account for that here
        if isinstance(glob_config["n_walkers"], str):
            n_walkers = int(glob_config["n_walkers"].split("*")[0]) * n_dims
        else:
            n_walkers = copy.deepcopy(glob_config["n_walkers"])

        sampler = emcee_wrapper(
            data,
            error,
            p0=p0_fit,
            n_walkers=n_walkers,
            n_dims=n_dims,
            n_comp=n_comp,
            progress=progress,
        )

        if save:
            # Calculate max likelihood
            flat_samples = get_samples(
                sampler,
                burn_in_frac=glob_config["burn_in"],
                thin_frac=glob_config["thin"],
            )
            parameter_median = np.nanmedian(
                flat_samples,
                axis=0,
            )
            likelihood = ln_like(
                theta=parameter_median,
                intensity=data,
                intensity_err=error,
                vel=glob_vel,
                strength_lines=glob_config["strength_lines"],
                v_lines=glob_config["v_lines"],
                props=glob_config["props"],
                n_comp=n_comp,
                fit_type=glob_config["fit_type"],
            )

            fit_dict = {
                "sampler": sampler,
                "n_comp": n_comp,
                "likelihood": likelihood,
            }

            save_pkl(fit_dict, fit_dict_filename)
    else:
        fit_dict = load_pkl(fit_dict_filename)
        sampler = fit_dict["sampler"]

    return sampler


def get_p0_lmfit(
    data,
    error,
    n_comp=1,
):
    """Get initial guesses for parameters using lmfit

    Args:
        data: Observed data
        error: Observed uncertainty
        n_comp: Number of components to fit. Defaults to 1
    """

    prop_len = len(glob_config["props"])

    bounds = glob_config["bounds"] * n_comp

    vel_idx = np.where(np.array(glob_config["props"]) == "v")[0][0]

    # Pull in any relevant kwargs
    kwargs = {}

    for config_dict in [glob_config["config_defaults"], glob_config["config"]]:
        if "lmfit" in config_dict:
            for key in config_dict["lmfit"]:
                kwargs[key] = config_dict["lmfit"][key]

    # We have a default here set to add minimizer_kwargs,
    # which will crash for minimizers that don't support this
    minimizers_with_minimizer_kwargs = [
        "basinhopping",
        "dual_annealing",
        "shgo",
    ]

    if kwargs.get("method", "leastsq") not in minimizers_with_minimizer_kwargs:
        kwargs.pop("minimizer_kwargs", None)

    # Find any NaNs in data or error maps
    good_idx = np.where(np.logical_and(np.isfinite(data), np.isfinite(error)))

    # And the velocity resolution
    dv = np.abs(np.nanmedian(np.diff(glob_vel)))

    p0 = np.array(glob_config["p0"] * n_comp)

    # Do the default method, where we brute force through
    # the least squares
    if glob_config["lmfit_method"] == "default":

        # Move the velocities a channel to encourage parameter space hunting

        for i in range(n_comp):
            p0[prop_len * i + vel_idx] += i * dv

        params = Parameters()
        p0_idx = 0
        for i in range(n_comp):
            for j in range(prop_len):
                params.add(
                    f"{glob_config["props"][j]}_{i}",
                    value=p0[p0_idx],
                    min=bounds[j][0],
                    max=bounds[j][1],
                )
                p0_idx += 1

        # Use lmfit to get an initial fit
        lmfit_result = minimize(
            fcn=initial_lmfit,
            params=params,
            args=(
                data[good_idx],
                error[good_idx],
                glob_vel[good_idx],
                glob_config["strength_lines"],
                glob_config["v_lines"],
                glob_config["props"],
                n_comp,
                glob_config["fit_type"],
                True,
            ),
            **kwargs,
        )

        p0_fit = np.array(
            [lmfit_result.params[key].value for key in lmfit_result.params]
        )

    # Get an initial guess for the velocities via an iterative method
    elif glob_config["lmfit_method"] == "iterative":

        # Start with a flat line model
        it_model = np.zeros(len(data))

        params = Parameters()
        lmfit_result = None
        p0_fit = np.array([])

        # Convolve will fail if there are NaNs in the spectrum,
        # so interpolate over them now
        data_interp = copy.deepcopy(data)

        nan_idx = ~np.isfinite(data_interp)
        x = np.arange(len(data_interp))

        data_interp[nan_idx] = np.interp(
            x[nan_idx],
            x[~nan_idx],
            data_interp[~nan_idx],
        )

        for comp in range(n_comp):

            it_data = data_interp - it_model

            # Find lines. We impose no flux cut here
            # flux_threshold = np.nanmedian(error) * u.K
            flux_threshold = None
            spec = Spectrum(
                flux=it_data * u.K,
                spectral_axis=glob_vel * u.km / u.s,
            )
            found_lines = find_lines_derivative(
                spec,
                flux_threshold=flux_threshold,
            )

            # Only take emission lines if we have any, else take everything
            emission_lines = found_lines[found_lines["line_type"] == "emission"]
            if len(emission_lines) > 0:
                found_lines = copy.deepcopy(emission_lines)

            # Now take these lines and order by flux
            found_line_fluxes = np.array(
                [float(np.abs(it_data[x])) for x in found_lines["line_center_index"]]
            )
            found_line_vels = found_lines["line_center"].value

            # Sort from brightest to faintest, pick the brightest component
            idxs = np.argsort(found_line_fluxes)[::-1]
            found_line_vel = found_line_vels[idxs][0]

            # Having found the velocity, we then fit all components to the data
            p0 = np.array([float(x) for x in glob_config["p0"]])
            p0[vel_idx] = copy.deepcopy(found_line_vel)

            # Update the parameters with any new best guesses
            if lmfit_result is not None:
                for key in lmfit_result.params:
                    params[key].set(value=lmfit_result.params[key].value)

            for j in range(prop_len):
                params.add(
                    f"{glob_config["props"][j]}_{comp}",
                    value=p0[j],
                    min=bounds[j][0],
                    max=bounds[j][1],
                )

            # Get a fit to the actual data
            lmfit_result = minimize(
                fcn=initial_lmfit,
                params=params,
                args=(
                    data[good_idx],
                    error[good_idx],
                    glob_vel[good_idx],
                    glob_config["strength_lines"],
                    glob_config["v_lines"],
                    glob_config["props"],
                    comp + 1,
                    glob_config["fit_type"],
                    True,
                ),
                **kwargs,
            )

            p0_fit = np.array(
                [lmfit_result.params[key].value for key in lmfit_result.params]
            )

            it_model = multiple_components(
                p0_fit,
                vel=glob_vel,
                strength_lines=glob_config["strength_lines"],
                v_lines=glob_config["v_lines"],
                props=glob_config["props"],
                n_comp=comp + 1,
                fit_type=glob_config["fit_type"],
                log_tau=True,
            )

    else:

        raise ValueError(f"lmfit_method should be one of {ALLOWED_LMFIT_METHODS}")

    # Sort p0 so it has monotonically increasing velocities
    v0_values = np.array([p0_fit[prop_len * i + vel_idx] for i in range(n_comp)])
    v0_sort = v0_values.argsort()
    p0_fit_sort = [p0_fit[prop_len * i : prop_len * i + prop_len] for i in v0_sort]
    p0_fit = np.array([item for sublist in p0_fit_sort for item in sublist])

    return p0_fit


def emcee_wrapper(
    data,
    error,
    p0,
    n_walkers,
    n_dims,
    n_comp=1,
    progress=False,
):
    """Light wrapper around emcee

    The runs the emcee part, with some custom moves
    and passes the arguments neatly to functions

    Args:
        data: Observed data
        error: Observed uncertainty
        p0: Initial guess position for the walkers
        n_walkers: Number of emcee walkers
        n_dims: Number of dimensions for the problem
        n_comp: Number of components to fit. Defaults
            to None
        progress: Whether or not to display progress bar.
            Defaults to False
    """

    if glob_config["data_type"] == "spectrum":

        # Multiprocess here for speed

        with mp.Pool(glob_config["n_cores"]) as pool:
            sampler = run_mcmc_sampler(
                data=data,
                error=error,
                p0=p0,
                n_walkers=n_walkers,
                n_dims=n_dims,
                n_comp=n_comp,
                progress=progress,
                pool=pool,
            )

    else:

        # Run in serial since the cube is multiprocessing already (no daemons here, not today satan)
        sampler = run_mcmc_sampler(
            data=data,
            error=error,
            p0=p0,
            n_walkers=n_walkers,
            n_dims=n_dims,
            n_comp=n_comp,
            progress=progress,
        )

    return sampler


def run_mcmc_sampler(
    data,
    error,
    p0,
    n_walkers,
    n_dims,
    n_comp=1,
    progress=False,
    pool=None,
):
    """Tool for actually running emcee

    There are two options here, fixed and adaptive which
    determine how long to run the MCMC walkers for. See
    the docs for more details

    Args:
        data: Observed data
        error: Observed uncertainty
        p0: Initial position guess for the walkers
        n_walkers: Number of emcee walkers
        n_dims: Number of dimensions for the problem
        n_comp: Number of components to fit. Defaults
            to None
        progress: Whether or not to display progress bar.
            Defaults to False
        pool: If running in multiprocessing mode, this
            is the mp Pool
    """

    # Set up the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=n_dims,
        log_prob_fn=ln_prob,
        args=(
            data,
            error,
            glob_vel,
            glob_config["strength_lines"],
            glob_config["v_lines"],
            glob_config["props"],
            glob_config["bounds"],
            n_comp,
            glob_config["fit_type"],
        ),
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
        pool=pool,
    )

    # START INITIALISATION RUNS

    # The initial positions use data percentage
    pos = initialise_positions(
        p0=p0,
        n_walkers=n_walkers,
        n_dims=n_dims,
        n_comp=n_comp,
        data_percentage=True,
    )

    for i in range(glob_config["n_initialisation"]):
        state = sampler.run_mcmc(
            pos,
            glob_config["n_initialisation_steps"],
            progress=progress,
        )

        # Get where we're at the maximum likelihood for the next
        # round, but also keep all the positions around in case
        # we're done
        max_prob_idx = np.argmax(state.log_prob)
        pos = copy.deepcopy(state.coords)
        p0 = pos[max_prob_idx]

        sampler.reset()

        # Initialise the walkers for the next run from the maximum
        # likelihood estimate
        pos = initialise_positions(
            p0=p0,
            n_walkers=n_walkers,
            n_dims=n_dims,
            n_comp=n_comp,
            data_percentage=False,
        )

    # Simple case where we have the fixed steps
    if glob_config["emcee_run_method"] == "fixed":
        sampler.run_mcmc(
            pos,
            glob_config["n_steps"],
            progress=progress,
        )

    elif glob_config["emcee_run_method"] == "adaptive":

        # Set up a tau for testing convergence
        old_tau = np.inf

        for _ in sampler.sample(
            pos,
            iterations=glob_config["max_steps"],
            progress=progress,
        ):

            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the median autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = np.nanmedian(sampler.get_autocorr_time(tol=0))

            # Check convergence
            converged = tau * glob_config["convergence_factor"] < sampler.iteration
            converged &= np.abs(old_tau - tau) / tau < glob_config["tau_change"]
            if converged:
                break
            old_tau = tau

    else:
        raise ValueError(
            f"emcee_run_method should be one of {ALLOWED_EMCEE_RUN_METHODS}, "
            f"not {glob_config['emcee_run_method']}"
        )

    return sampler


def initialise_positions(
    p0,
    n_walkers,
    n_dims,
    n_comp,
    data_percentage=False,
):
    """Initialise positions for the walkers

    Args:
        p0: "best fit" values
        n_walkers: number of walkers
        n_dims: number of dimensions
        n_comp: number of components
        data_percentage: Whether to move walkers
            around based on a percentage of data.
            Defaults to False
    """

    # Shuffle the parameters around a little.

    # If we're using some percentage of the data, include that here,
    # but avoid values less than 1
    if data_percentage:
        p0_movement = np.max(
            np.array([1e-2 * np.abs(p0), 1e-2 * np.ones_like(p0)]),
            axis=0,
        )
    else:
        p0_movement = 1e-4 * np.ones_like(p0)

    # Reinitialise the walkers at this point, wiggle around a small amount
    pos = p0 + p0_movement * np.random.randn(n_walkers, n_dims)

    # Enforce positive values for t_ex, width for the LTE/pure Gaussian fitting
    if glob_config["fit_type"] == "lte":
        positive_idx = [0, 3]
    elif glob_config["fit_type"] == "pure_gauss":
        positive_idx = [0, 2]
    elif glob_config["fit_type"] == "radex":
        positive_idx = [0, 1, 4]
    else:
        logger.warning(f"Fit type {glob_config['fit_type']} not understood!")
        sys.exit()

    prop_len = len(glob_config["props"])

    enforced_positives = [
        [prop_len * i + j] for i in range(n_comp) for j in positive_idx
    ]
    enforced_positives = [item for sublist in enforced_positives for item in sublist]

    for i in enforced_positives:
        pos[:, i] = np.abs(pos[:, i])

    return pos


def parallel_fit_samples(
    ij,
    fit_dict_filename=None,
    consolidate_fit_dict=True,
):
    """Pull fit percentiles from a single pixel

    Will pull from the fit dict if we can, else will
    read in each individual file
    """

    if fit_dict_filename is None:
        logger.warning("fit_dict_filename must be defined!")
        sys.exit()

    i = ij[0]
    j = ij[1]

    if consolidate_fit_dict:
        fit_dict = glob_mcfine_output["fit"].get(i, {}).get(j, {})
    else:
        cube_sampler_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        fit_dict = load_pkl(cube_sampler_filename)
    n_comp = fit_dict["n_comp"]

    if n_comp == 0:
        return np.zeros([3, len(glob_vel)])

    flat_samples = get_samples_from_fit_dict(
        fit_dict, burn_in_frac=glob_config["burn_in"], thin_frac=glob_config["thin"]
    )

    fit_lines = get_fits_from_samples(
        flat_samples,
        vel=glob_vel,
        props=glob_config["props"],
        strength_lines=glob_config["strength_lines"],
        v_lines=glob_config["v_lines"],
        fit_type=glob_config["fit_type"],
        n_draws=100,
        n_comp=n_comp,
    )

    fit_percentiles = np.nanpercentile(
        np.nansum(fit_lines, axis=-1),
        [16, 50, 84],
        axis=1,
    )
    fit_diffs = np.diff(fit_percentiles, axis=0)

    # Put these into down error/nominal/up error array
    fit_final = np.array([fit_diffs[0, :], fit_percentiles[1, :], fit_diffs[1, :]])

    return fit_percentiles


def get_fits_from_samples(
    samples,
    vel,
    props,
    strength_lines,
    v_lines,
    fit_type="lte",
    n_draws=100,
    n_comp=1,
):
    """Get a number of fit lines from an MCMC run

    Args:
        samples: emcee output
        vel: Velocity grid to evaluate the fit on
        props: List of properties
        strength_lines: List of line strengths
        fit_type: Fit type. Defaults to "lte"
        v_lines: List of line velocities
        n_draws (int): Number of draws to pull from samples
        n_comp (int): Number of components in the fit

    Returns:
        array of best fit line
    """

    fit_lines = np.zeros([len(vel), n_draws, n_comp])

    for draw in range(n_draws):
        sample = np.random.randint(low=0, high=samples.shape[0])
        for i in range(n_comp):
            theta_draw = samples[sample, ...][
                len(props) * i : len(props) * i + len(props)
            ]

            if fit_type == "lte":

                fit_lines[:, draw, i] = hyperfine_structure_lte(
                    *theta_draw,
                    strength_lines=strength_lines,
                    v_lines=v_lines,
                    vel=vel,
                )

            elif fit_type == "pure_gauss":

                fit_lines[:, draw, i] = hyperfine_structure_pure_gauss(
                    *theta_draw,
                    strength_lines=strength_lines,
                    v_lines=v_lines,
                    vel=vel,
                )

            elif fit_type == "radex":

                qn_ul = np.array(range(len(radex_grid["QN_ul"].values)))

                fit_lines[:, draw, i] = get_radex_multiple_components(
                    theta_draw,
                    vel=vel,
                    v_lines=v_lines,
                    qn_ul=qn_ul,
                )

    return fit_lines


class HyperfineFitter:

    def __init__(
        self,
        data=None,
        vel=None,
        error=None,
        mask=None,
        config_file=None,
        local_file=None,
    ):
        """Multi-component, hyperfine MCMC line fitting.

        A fully-automated, highly customisable multi-component fitter. For full details, see Williams & Watkins.
        For tutorials, see the docs

        Args:
            data (np.ndarray): Either a 1D array of intensity (spectrum), a 3D array of intensities (cube), or a string,
                which will load in a cube using spectral_cube. If an array, intensities should be in K.
            error (np.ndarray): Array of errors in intensity, or a string to load in using spectral_cube.
                Should have the same shape as `data`. Defaults to None.
            mask (np.ndarray): 1/0 mask to indicate significant emission in the cube (i.e. the pixels to fit). Should
                have shape of `data.shape[1:]`. Defaults to None, which will fit all pixels in a cube.
            vel (np.ndarray): Array of velocity values that correspond to data, in km/s. Not required if loading in
                a spectral cube. Defaults to None.
            config_file (str): Path to config.toml file. Defaults to None, which will use the default settings
            local_file (str): Path to local.toml file. Defaults to None, which will use the default settings

        Todo:
            * Bounds as parameters to input here.
            * Test the radex fitting routines on cubes.
        """

        # Let us know if ndradex has been imported
        if not NDRADEX_IMPORTED:
            logger.warning("ndradex not imported. RT capabilities disabled")

        if data is None:
            logger.warning("data should be provided!")
            sys.exit()
        if vel is None and not isinstance(data, str):
            logger.warning(
                "velocity definition should be provided! Defaulting to monotonically increasing"
            )
            vel = np.arange(data.shape[0])
        if error is None:
            logger.info("No error provided. Defaulting to 0")
            error = np.zeros_like(data)

        self.logger = logger

        # Load in the data spectral cube
        wcs = None
        wcs_2d = None
        if isinstance(data, str):
            data = SpectralCube.read(data)

            # Get WCS out
            wcs = data.wcs
            wcs_2d = data[0, :, :].wcs

            # Get the velocity axis out
            vel = data.spectral_axis.to(u.km / u.s).value

            # Convert to K, pull out values
            data = data.unmasked_data[:].to(u.K).value

        # Load in the error spectral cube
        if isinstance(error, str):
            error = SpectralCube.read(error)

            # Convert to K, pull out values
            error = error.unmasked_data[:].to(u.K).value

        self.data = data
        self.error = error
        self.vel = vel
        self.wcs = wcs
        self.wcs_2d = wcs_2d

        # Define global variables for potentially huge arrays
        global glob_data, glob_error, glob_vel
        glob_data = self.data
        glob_error = self.error
        glob_vel = self.vel

        self.downsampled_data = np.array([])
        self.downsampled_error = np.array([])
        self.downsampled_mask = np.array([])
        self.initial_n_comp = np.array([])

        with open(CONFIG_DEFAULT_PATH, "rb") as f:
            config_defaults = tomllib.load(f)
        with open(LOCAL_DEFAULT_PATH, "rb") as f:
            local_defaults = tomllib.load(f)

        if config_file is not None:
            with open(config_file, "rb") as f:
                config = tomllib.load(f)
        else:
            self.logger.info("No config file provided. Using in-built defaults")
            config = copy.deepcopy(config_defaults)

        if local_file is not None:
            with open(local_file, "rb") as f:
                local = tomllib.load(f)
        else:
            self.logger.info("No local file provided. Using in-built defaults")
            local = copy.deepcopy(local_defaults)

        self.config = config
        self.local = local

        self.config_defaults = config_defaults
        self.local_defaults = local_defaults

        fit_type = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="fit_type",
            logger=self.logger,
        )

        if fit_type not in ALLOWED_FIT_TYPES:
            self.logger.warning(f"Fit type {fit_type} not understood!")
            sys.exit()

        self.fit_type = fit_type

        lmfit_method = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="lmfit_method",
            logger=self.logger,
        )

        if lmfit_method not in ALLOWED_LMFIT_METHODS:
            self.logger.warning(
                f"lmfit_method should be one of {ALLOWED_LMFIT_METHODS}!"
            )
            sys.exit()

        self.lmfit_method = lmfit_method

        if self.fit_type == "radex" and not NDRADEX_IMPORTED:
            raise ValueError("Cannot use mode radex if ndradex is not installed!")

        fit_method = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="fit_method",
            logger=self.logger,
        )

        if fit_method not in ALLOWED_FIT_METHODS:
            self.logger.warning(f"Fitting procedure {fit_method} not understood!")
            sys.exit()

        self.fit_method = fit_method

        # Are we consolidate the fit dictionary into one file?
        consolidate_fit_dict = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="consolidate_fit_dict",
            logger=self.logger,
        )
        self.consolidate_fit_dict = consolidate_fit_dict

        # Are we keeping the sampler/covariance matrix?
        keep_sampler = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="keep_sampler",
            logger=self.logger,
        )
        self.keep_sampler = keep_sampler

        keep_covariance = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="keep_covariance",
            logger=self.logger,
        )
        self.keep_covariance = keep_covariance

        # If we're not keeping anything around, freak out
        if not self.keep_sampler and not self.keep_covariance:
            raise ValueError(
                "You have to keep around at least one of the sampler and covariance matrix!"
            )

        self.dv = np.abs(np.nanmedian(np.diff(self.vel)))

        if self.data.ndim == 1:
            self.data_type = "spectrum"
        else:
            self.data_type = "cube"

        self.logger.info(f"Detected data type as {self.data_type}")

        if self.data_type == "cube":

            if mask is None:
                self.logger.info("No mask provided. Including every pixel")
                mask = np.ones([data.shape[1], data.shape[2]])

            if self.data.shape != self.error.shape:
                self.logger.warning("Data and error should be the same shape")
                sys.exit()
            if self.data.shape[1:] != mask.shape:
                self.logger.warning("Mask should be 2D projection of data")
                sys.exit()

        self.mask = mask

        line = get_dict_val(
            self.config,
            self.config_defaults,
            table="fitting_params",
            key="line",
            logger=self.logger,
        )

        if line not in allowed_lines:
            self.logger.warning(f"Line {line} not understood!")
            sys.exit()

        self.line = line

        if self.fit_type == "lte":

            self.strength_lines = np.array(
                [
                    strength_lines_dict[line_name]
                    for line_name in strength_lines_dict.keys()
                    if self.line in line_name
                ]
            )
            self.v_lines = np.array(
                [
                    v_lines_dict[line_name]
                    for line_name in v_lines_dict.keys()
                    if self.line in line_name
                ]
            )
            self.bounds = [
                (T_BACKGROUND, 1e3),
                (np.log(0.1), np.log(30)),
                (np.nanmin(self.vel), np.nanmax(self.vel)),
                (self.dv / 2.355, 500),
            ]

        elif self.fit_type == "pure_gauss":

            self.strength_lines = np.array(
                [
                    strength_lines_dict[line_name]
                    for line_name in strength_lines_dict.keys()
                    if self.line in line_name
                ]
            )
            self.v_lines = np.array(
                [
                    v_lines_dict[line_name]
                    for line_name in v_lines_dict.keys()
                    if self.line in line_name
                ]
            )
            self.bounds = [
                (0, 1e3),
                (np.nanmin(self.vel), np.nanmax(self.vel)),
                (self.dv / 2.355, 500),
            ]

        elif self.fit_type == "radex":

            rest_freq = get_dict_val(
                self.config,
                self.config_defaults,
                table="fitting_params",
                key="rest_freq",
                logger=self.logger,
            )

            if rest_freq == "":
                self.logger.warning("rest_freq should be defined")
                sys.exit()
            if isinstance(rest_freq, float) or isinstance(rest_freq, int):
                rest_freq = rest_freq * u.GHz

            self.strength_lines = None
            self.transitions = [
                transition_lines[line_name]
                for line_name in transition_lines.keys()
                if self.line in line_name
            ]
            freq = (
                np.array(
                    [
                        freq_lines[line_name]
                        for line_name in freq_lines.keys()
                        if self.line in line_name
                    ]
                )
                * u.GHz
            )
            freq_to_vel = u.doppler_radio(rest_freq)
            self.v_lines = freq.to(u.km / u.s, equivalencies=freq_to_vel).value
            self.bounds = [
                (T_BACKGROUND, 75),
                (13, 15),
                (5, 8),
                (np.nanmin(self.vel), np.nanmax(self.vel)),
                (self.dv / 2.355, 500),
            ]

            radex_datafile = get_dict_val(
                self.local,
                self.local_defaults,
                table="local",
                key="radex_datafile",
                logger=self.logger,
            )

            self.radex_datafile = radex_datafile

        p0 = get_dict_val(
            self.config,
            self.config_defaults,
            table="initial_guess",
            key=fit_type,
            logger=self.logger,
        )
        self.p0 = p0

        if not len(self.bounds) == len(self.p0):
            self.logger.warning("bounds and p0 should have the same length!")
            sys.exit()

        delta_bic_cutoff = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="delta_bic_cutoff",
            logger=self.logger,
        )

        self.delta_bic_cutoff = delta_bic_cutoff

        delta_aic_cutoff = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="delta_aic_cutoff",
            logger=self.logger,
        )

        self.delta_aic_cutoff = delta_aic_cutoff

        emcee_run_method = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="emcee_run_method",
            logger=self.logger,
        )

        if emcee_run_method not in ALLOWED_EMCEE_RUN_METHODS:
            raise ValueError(
                f"emcee_run_method should be one of {ALLOWED_EMCEE_RUN_METHODS}, not {emcee_run_method}"
            )

        self.emcee_run_method = emcee_run_method

        # Variables for the adaptive fitting method
        max_steps = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="max_steps",
            logger=self.logger,
        )
        self.max_steps = max_steps

        convergence_factor = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="convergence_factor",
            logger=self.logger,
        )
        self.convergence_factor = convergence_factor

        tau_change = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="tau_change",
            logger=self.logger,
        )
        self.tau_change = tau_change

        thin = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="thin",
            logger=self.logger,
        )
        self.thin = thin

        burn_in = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="burn_in",
            logger=self.logger,
        )
        self.burn_in = burn_in

        n_initialisation = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="n_initialisation",
            logger=self.logger,
        )
        self.n_initialisation = n_initialisation

        n_initialisation_steps = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="n_initialisation_steps",
            logger=self.logger,
        )
        self.n_initialisation_steps = n_initialisation_steps

        # Variables for fixed fitting method
        n_steps = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="n_steps",
            logger=self.logger,
        )
        self.n_steps = n_steps

        n_walkers = get_dict_val(
            self.config,
            self.config_defaults,
            table="mcmc",
            key="n_walkers",
            logger=self.logger,
        )
        self.n_walkers = n_walkers

        n_cores = get_dict_val(
            self.local,
            self.local_defaults,
            table="local",
            key="n_cores",
            logger=self.logger,
        )

        if n_cores == "":
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = n_cores

        if self.fit_type == "lte":
            self.props = [
                "tex",
                "tau",
                "v",
                "sigma",
            ]
            self.labels = [
                r"$T_\mathrm{ex}$ (%s)",
                r"$\log(\tau)$ (%s)",
                r"$v$ (%s)",
                r"$\sigma$ (%s)",
            ]

        elif self.fit_type == "pure_gauss":
            self.props = [
                "t",
                "v",
                "sigma",
            ]
            self.labels = [
                r"$T$ (%s)",
                r"$v$ (%s)",
                r"$\sigma$ (%s)",
            ]

        elif self.fit_type == "radex":
            self.props = [
                "t_kin",
                "N_col",
                "n_h2",
                "v",
                "sigma",
            ]
            self.labels = [
                r"$T_\mathrm{kin}$ (%s)",
                r"$N_\mathrm{col}$ (%s)",
                r"$n_\mathrm{H2}$ (%s)",
                r"$v$ (%s)",
                r"$\sigma$ (%s)",
            ]
        else:
            self.logger.warning(f"fit_type {self.fit_type} not known")
            sys.exit()

        # Define a global configuration dictionary that we'll use in multiprocessing
        global glob_config

        keys_to_glob = [
            "config_defaults",
            "config",
            "data_type",
            "props",
            "p0",
            "bounds",
            "strength_lines",
            "v_lines",
            "delta_bic_cutoff",
            "delta_aic_cutoff",
            "lmfit_method",
            "fit_type",
            "emcee_run_method",
            "n_cores",
            "n_walkers",
            "n_steps",
            "n_initialisation",
            "n_initialisation_steps",
            "burn_in",
            "thin",
            "convergence_factor",
            "tau_change",
            "max_steps",
            "keep_sampler",
            "keep_covariance",
        ]
        for k in keys_to_glob:
            glob_config[k] = self.__dict__[k]

        self.parameter_maps = None

    def initialise_mcfine_output(self, data=None, data_type="original"):
        """Initialise the mcfine output, which stores useful information about the fit run

        Args:
            data: Array of data
            data_type: Type of data. Should be "original" or "downsampled". Will not take WCS
                for downsampled data
        """

        mcfine_output = {}

        # Start by pulling in the configurations
        mcfine_output["user_configuration_settings"] = copy.deepcopy(self.config)
        mcfine_output["default_configuration_settings"] = copy.deepcopy(
            self.config_defaults
        )

        mcfine_output["user_local_settings"] = copy.deepcopy(self.local)
        mcfine_output["default_local_settings"] = copy.deepcopy(self.local_defaults)

        # Then information about the data, which should only exist if we're fitting a cube
        if data is not None:
            mcfine_output["data"] = {}
            mcfine_output["data"]["shape"] = data.shape

            # Add in the WCS, so we can spit out fits files later
            if self.wcs is not None and data_type != "downsampled":
                mcfine_output["data"]["wcs"] = copy.deepcopy(self.wcs)
                mcfine_output["data"]["wcs_2d"] = copy.deepcopy(self.wcs_2d)

        # And finally, a space for the fits
        if self.consolidate_fit_dict:
            mcfine_output["fit"] = {}

        return mcfine_output

    def generate_radex_grid(
        self,
        output_file=None,
    ):
        """Pre-generate RADEX grid, given various bounds and steps

        This will be the grid we interpolate over for speed later. As such,
        the parameters should be set to cover the parameter range you care about.
        Parameters are stored in config.toml

        Args:
            output_file (str): Output filename. Defaults to None, which will pull
                from config.toml

        """

        if self.fit_type != "radex":
            self.logger.warning("fit_type should be radex")
            sys.exit()

        if output_file is None:
            output_file = get_dict_val(
                self.config,
                self.config_defaults,
                table="generate_radex_grid",
                key="output_file",
                logger=self.logger,
            )

        t_kin = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="t_kin",
            logger=self.logger,
        )
        n_mol = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="n_mol",
            logger=self.logger,
        )
        n_h2 = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="n_h2",
            logger=self.logger,
        )
        dv = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="dv",
            logger=self.logger,
        )

        if t_kin == "":
            t_kin = self.bounds[0]
        if n_mol == "":
            n_mol = self.bounds[1]
        if n_h2 == "":
            n_h2 = self.bounds[2]
        if dv == "":
            dv = self.bounds[4]

        t_kin_step = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="t_kin_step",
            logger=self.logger,
        )
        n_mol_step = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="n_mol_step",
            logger=self.logger,
        )
        n_h2_step = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="n_h2_step",
            logger=self.logger,
        )
        dv_step = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="dv_step",
            logger=self.logger,
        )

        geom = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="geom",
            logger=self.logger,
        )
        progress = get_dict_val(
            self.config,
            self.config_defaults,
            table="generate_radex_grid",
            key="progress",
            logger=self.logger,
        )

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        if not os.path.exists(output_file) or overwrite:

            t_kin_array = np.arange(t_kin[0], t_kin[1] + t_kin_step, t_kin_step)
            n_mol_array = np.arange(n_mol[0], n_mol[1] + n_mol_step, n_mol_step)
            n_h2_array = np.arange(n_h2[0], n_h2[1] + n_h2_step, n_h2_step)
            dv_array = (
                np.arange(dv[0], dv[1] + dv_step, dv_step) * 2.355
            )  # 2.355 since distinction between sigma/dv

            ds = ndradex.run(
                self.radex_datafile,
                self.transitions,
                T_kin=t_kin_array,
                N_mol=10**n_mol_array,
                n_H2=10**n_h2_array,
                dv=dv_array,
                T_bg=T_BACKGROUND,
                geom=geom,
                progress=progress,
                n_procs=self.n_cores,
            )

            ndradex.save_dataset(ds, output_file)

        else:

            ds = ndradex.load_dataset(output_file)

        global radex_grid
        radex_grid = ds

    def downsample_fitter(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        downsample_factor=10,
    ):
        """Run multicomponent fitter on downsampled data

        This is a light wrapper around multicomponent fitter,
        but also takes

        Args:
            fit_dict_filename: Filename to save the fitted emcee walkers
                to. Defaults to None, which will not save anything
            mcfine_output_filename: Filename to save the mcfine output to.
                Defaults to None, which will not save anything
            downsample_factor: Factor to downsample the data down on each spatial axis
        """

        if self.data_type != "cube":
            raise ValueError("Can only do downsample fitting on cubes")

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        # For data and error, we want some representative values within the averaging
        # area. Otherwise, we can hugely boost S/N and end up fitting all the blended
        # components, which is not what we want!
        downsampled_data = downsample(
            self.data, chunk_size=downsample_factor, func=np.nanmedian
        )
        downsampled_error = downsample(
            self.error, chunk_size=downsample_factor, func=np.nanmedian
        )

        # Mask is a little different. Take sum and reduce to bool
        downsampled_mask = downsample(
            self.mask, chunk_size=downsample_factor, func=np.nansum
        )
        downsampled_mask[downsampled_mask < 1] = 0
        downsampled_mask[downsampled_mask >= 1] = 1

        self.downsampled_data = downsampled_data
        self.downsampled_error = downsampled_error
        self.downsampled_mask = downsampled_mask

        global glob_downsampled_data, glob_downsampled_error
        glob_downsampled_data = downsampled_data
        glob_downsampled_error = downsampled_error

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{mcfine_output_filename}.pkl") or overwrite:
            mcfine_output = self.multicomponent_fitter(
                fit_dict_filename=fit_dict_filename,
                mcfine_output_filename=mcfine_output_filename,
                data_type="downsampled",
            )

            # Save the output
            save_pkl(
                mcfine_output,
                f"{mcfine_output_filename}.pkl",
            )

        else:
            mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")

        # Get n_comp out and sample back to the original grid. If we've
        # got the info in the fit dict, this is trivial
        if "fit" in mcfine_output:

            n_comp = np.array(
                [
                    [
                        mcfine_output["fit"].get(i, {}).get(j, {}).get("n_comp", 0)
                        for i in range(downsampled_data.shape[1])
                    ]
                    for j in range(downsampled_data.shape[2])
                ]
            ).T

        # Otherwise, loop over files and pull out info
        else:

            n_comp = np.zeros(downsampled_data.shape[1:])

            if fit_dict_filename is None:
                fit_dict_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="fit_dict_filename",
                    logger=self.logger,
                )

            for i in range(downsampled_data.shape[1]):
                for j in range(downsampled_data.shape[2]):

                    fit_dict_f = f"{fit_dict_filename}_{i}_{j}.pkl"

                    if os.path.exists(fit_dict_f):
                        fit_dict = load_pkl(fit_dict_f)

                        n_comp[i, j] = fit_dict["n_comp"]

        # Get indices for the downsampled and upsampled arrays
        i_us = np.arange(self.data.shape[1])
        j_us = np.arange(self.data.shape[2])
        i_ds = np.arange(downsample_factor, self.data.shape[1], downsample_factor)
        j_ds = np.arange(downsample_factor, self.data.shape[2], downsample_factor)

        n_comp_upsampled = np.zeros(self.data.shape[1:])

        i_split = np.array_split(i_us, i_ds)
        j_split = np.array_split(j_us, j_ds)

        for i_idx, i in enumerate(i_split):
            for j_idx, j in enumerate(j_split):
                # Get min/max indices to map back to the array
                i_low = np.min(i)
                i_high = np.max(i)
                j_low = np.min(j)
                j_high = np.max(j)

                n_comp_upsampled[i_low : i_high + 1, j_low : j_high + 1] = n_comp[
                    i_idx, j_idx
                ]

        self.initial_n_comp = copy.deepcopy(n_comp_upsampled)

        global glob_initial_n_comp
        glob_initial_n_comp = self.initial_n_comp

        return True

    def multicomponent_fitter(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        data_type="original",
    ):
        """Run the multicomponent fitter

        This runs everything, essentially, and is the part that you
        should call. It'll do LMFIT to get guesses, emcee to properly
        sample parameter space and then iteratively add components
        before removing them.

        Args:
            fit_dict_filename: Filename to save the fitted emcee walkers
                to. Defaults to None, which will not save anything
            mcfine_output_filename: Filename to save the mcfine output to.
                Defaults to None, which will not save anything
            data_type: Data type to fit. Can be either "original" or "downsampled"
        """

        if data_type not in ["original", "downsampled"]:
            raise ValueError("Data type to fit should be original or downsampled")

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        self.logger.info("Starting multi-component fitting")

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{mcfine_output_filename}.pkl") or overwrite:

            if fit_dict_filename is None:
                fit_dict_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="fit_dict_filename",
                    logger=self.logger,
                )

            # Create output directory if it doesn't exist
            fit_dict_base_dir = os.path.dirname(fit_dict_filename)
            if not os.path.exists(fit_dict_base_dir):
                os.makedirs(fit_dict_base_dir)

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="chunksize",
                logger=self.logger,
            )

            progress = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="progress",
                logger=self.logger,
            )

            save = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="save",
                logger=self.logger,
            )

            if data_type == "original":
                data = copy.deepcopy(self.data)
                error = copy.deepcopy(self.error)
                mask = copy.deepcopy(self.mask)
            elif data_type == "downsampled":
                data = copy.deepcopy(self.downsampled_data)
                error = copy.deepcopy(self.downsampled_error)
                mask = copy.deepcopy(self.downsampled_mask)

                # If somehow we've broken the logic here, then freak out
                if data.size == 0:
                    raise ValueError(
                        "Must run data downsampling if you want to fit downsampled data"
                    )

            else:
                raise ValueError("Data type to fit should be original or downsampled")

            # Start fitting!
            mcfine_output = None

            if self.data_type == "spectrum":

                mcfine_output = self.initialise_mcfine_output()

                if not os.path.exists(f"{mcfine_output_filename}.pkl") or overwrite:

                    fit_dict = delta_bic_looper(
                        data,
                        error,
                        fit_method=self.fit_method,
                        progress=progress,
                    )

                    if not self.keep_sampler:
                        fit_dict.pop("sampler")
                    if not self.keep_covariance:
                        fit_dict.pop("cov_matrix")
                        fit_dict.pop("cov_med")

                    if self.consolidate_fit_dict:
                        mcfine_output["fit"].update(fit_dict)

                    if save:
                        save_pkl(fit_dict, f"{fit_dict_filename}.pkl")
                        save_pkl(mcfine_output, f"{mcfine_output_filename}.pkl")

            elif self.data_type == "cube":

                mcfine_output = self.initialise_mcfine_output(
                    data=data,
                    data_type=data_type,
                )

                ij_list = [
                    (i, j)
                    for i in range(data.shape[1])
                    for j in range(data.shape[2])
                    if mask[i, j] != 0
                ]

                self.logger.info(f"Fitting using {self.n_cores} cores")

                # Do the fitting. This requires saving out files, since
                # we just return a filename
                with mp.Pool(self.n_cores) as pool:
                    result = list(
                        tqdm(
                            pool.imap(
                                partial(
                                    parallel_fitting,
                                    fit_dict_filename=fit_dict_filename,
                                    data_type=data_type,
                                    fit_method=self.fit_method,
                                    overwrite=overwrite,
                                ),
                                ij_list,
                                chunksize=chunksize,
                            ),
                            total=len(ij_list),
                            dynamic_ncols=True,
                        )
                    )

                for idx, ij in enumerate(ij_list):
                    if ij[0] not in mcfine_output["fit"]:
                        mcfine_output["fit"][ij[0]] = {}
                    if ij[1] not in mcfine_output["fit"][ij[0]]:
                        mcfine_output["fit"][ij[0]][ij[1]] = {}

                    fit_dict = load_pkl(result[idx])

                    # Consolidate fit dictionaries
                    if self.consolidate_fit_dict:
                        mcfine_output["fit"][ij[0]][ij[1]].update(fit_dict)

                if save:
                    save_pkl(mcfine_output, f"{mcfine_output_filename}.pkl")

        else:
            mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")

        return mcfine_output

    def encourage_spatial_coherence(
        self,
        input_dir="fit",
        output_dir="fit_coherence",
        fit_dict_filename=None,
        mcfine_input_filename=None,
        mcfine_output_filename=None,
        reverse_direction=False,
    ):
        """Loop over fits to encourage spatial coherence

        This will loop over RA/Dec to potentially replace fits
        with neighbouring fits. This helps to encourage spatial
        coherence and can also help in the case of catastrophic
        misfits

        Args:
            input_dir: directory for the input fits. Defaults to 'fit'
            output_dir: directory for the output of the coherence checking.
                Defaults to 'fit_coherence'
            fit_dict_filename: Filename structure for the fit dictionary.
                Default to None, which will choose some generic name
            mcfine_input_filename: Name for the consolidated mcfine output to start with.
                Defaults to None, which will pull the default name "mcfine"
            mcfine_output_filename: Name for the consolidated mcfine output to save at the
                end. Defaults to None, which will pull the default name "mcfine_coherence"
            reverse_direction: Whether to reverse how we step through x/y
                in the coherence encouragement. Defaults to False, but likely
                you should run one case forward, then another backward
        """

        if self.data_type != "cube":
            self.logger.warning("Can only do spatial coherence on a cube!")
            sys.exit()

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
                logger=self.logger,
            )

        if mcfine_input_filename is None:
            mcfine_input_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{mcfine_input_filename}.pkl"):
            raise FileNotFoundError(
                f"{mcfine_input_filename}.pkl not found! Make sure this exists"
            )

        # If we don't have an output filename, take the default plus a "coherence" appended
        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )
            mcfine_output_filename += "_coherence"

        if os.path.exists(f"{mcfine_output_filename}.pkl") and not overwrite:
            return True

        if reverse_direction:
            step = -1
        else:
            step = 1

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Test hardlinks work
        hardlinks_supported = True
        hl_input = os.path.join(input_dir, "hl_test.txt")
        hl_output = os.path.join(output_dir, "hl_test.txt")
        os.system(f"touch {hl_input}")
        try:
            os.link(hl_input, hl_output)
        except:
            hardlinks_supported = False

        os.system(f"rm {hl_output}")
        os.system(f"rm {hl_input}")

        if overwrite:

            # Flush out the directory
            files_in_dir = glob.glob(os.path.join(output_dir, "*"))
            for file_in_dir in files_in_dir:
                os.remove(file_in_dir)

            if os.path.exists(f"{mcfine_output_filename}.pkl"):
                os.remove(f"{mcfine_output_filename}.pkl")

        total_found = 0

        ij_list = [
            (i, j)
            for i in range(self.data.shape[1])[::step]
            for j in range(self.data.shape[2])[::step]
            if self.mask[i, j] != 0
        ]

        # Read in the previous output dictionary, to define the new one
        mcfine_output = load_pkl(f"{mcfine_input_filename}.pkl")

        # Set up a dictionary to record best-fit parameters, to
        # reduce I/O
        fit_params = {}

        for ij in tqdm(
            ij_list,
            dynamic_ncols=True,
        ):

            i, j = ij[0], ij[1]

            input_file = os.path.join(input_dir, f"{fit_dict_filename}_{i}_{j}.pkl")
            output_file = os.path.join(output_dir, f"{fit_dict_filename}_{i}_{j}.pkl")

            if not os.path.exists(output_file) or overwrite:

                # Pull original likelihood for the pixel
                fit_dict = load_pkl(input_file)
                bic_original = fit_dict["bic"]
                aic_original = fit_dict["aic"]

                # Pull out the neighbouring pixels
                i_min = max(0, i - 1)
                i_max = min(self.data.shape[1], i + 2)
                j_min = max(0, j - 1)
                j_max = min(self.data.shape[2], j + 2)

                delta_bic = np.zeros([i_max - i_min, j_max - j_min])
                delta_aic = np.zeros_like(delta_bic)
                likelihood_cutout = np.zeros_like(delta_bic)

                for i_cutout, i_full in enumerate(range(i_min, i_max)):
                    for j_cutout, j_full in enumerate(range(j_min, j_max)):

                        if self.mask[i_full, j_full] == 0:
                            continue

                        # If we haven't already looked at this one, pull in
                        # the fit dictionary and get parameters out
                        fit_params_key = f"{i_full}_{j_full}"

                        if fit_params_key not in fit_params:
                            # Check if we already have moved the fit dictionary
                            cutout_fit_dict_filename = os.path.join(
                                output_dir,
                                f"{fit_dict_filename}_{i_full}_{j_full}.pkl",
                            )

                            if not os.path.exists(cutout_fit_dict_filename):
                                cutout_fit_dict_filename = os.path.join(
                                    input_dir,
                                    f"{fit_dict_filename}_{i_full}_{j_full}.pkl",
                                )
                            cutout_fit_dict = load_pkl(cutout_fit_dict_filename)

                            fit_params[fit_params_key] = {}
                            fit_params[fit_params_key].update(cutout_fit_dict)

                            # Pull out the parameters
                            theta = [
                                float(cutout_fit_dict["props"][p][n])
                                for n in range(cutout_fit_dict["n_comp"])
                                for p in self.props
                            ]
                            fit_params[fit_params_key]["theta"] = theta

                        n_comp_new = fit_params[fit_params_key]["n_comp"]
                        theta = fit_params[fit_params_key]["theta"]

                        # Get the various goodness of fit metrics
                        gof_metrics = calculate_goodness_of_fit(
                            data=self.data[:, i, j],
                            error=self.error[:, i, j],
                            best_fit_pars=theta,
                            n_comp=n_comp_new,
                        )

                        delta_bic[i_cutout, j_cutout] = (
                            bic_original - gof_metrics["bic"]
                        )
                        delta_aic[i_cutout, j_cutout] = (
                            aic_original - gof_metrics["aic"]
                        )
                        likelihood_cutout[i_cutout, j_cutout] = gof_metrics[
                            "likelihood"
                        ]

                # Set ones where we don't have a meaningful value to something small
                delta_bic[delta_bic == 0] = -1000
                delta_aic[delta_aic == 0] = -1000

                # Find the position of maximum change in BIC between pixels.
                idx = np.unravel_index(np.argmax(delta_bic, axis=None), delta_bic.shape)

                fit_updated = False
                # Use both the BIC and AIC criterion here to distinguish, we require both to be significant
                if (
                    delta_bic[idx] > self.delta_bic_cutoff
                    and delta_aic[idx] > self.delta_aic_cutoff
                ):
                    total_found += 1

                    # If we're replacing with a file we've already replaced, pull from the output directory. Else
                    # pull from the input directory.
                    input_file = os.path.join(
                        output_dir,
                        f"{fit_dict_filename}_{idx[0] + i_min}_{idx[1] + j_min}.pkl",
                    )
                    if not os.path.exists(input_file):
                        input_file = os.path.join(
                            input_dir,
                            f"{fit_dict_filename}_{idx[0] + i_min}_{idx[1] + j_min}.pkl",
                        )
                    fit_updated = True

                # If the fits have been updated, then update the likelihood and save out
                if fit_updated:
                    updated_fit_dict = load_pkl(input_file)

                    # Update the fit parameter dictionary
                    fit_param_key = f"{i}_{j}"
                    fit_params[fit_param_key].update(updated_fit_dict)

                    # Pull out parameters
                    theta = [
                        float(updated_fit_dict["props"][p][n])
                        for n in range(updated_fit_dict["n_comp"])
                        for p in self.props
                    ]
                    fit_params[fit_param_key]["theta"] = copy.deepcopy(theta)

                    # Update goodness of fit
                    n_comp = updated_fit_dict["n_comp"]
                    gof_metrics = calculate_goodness_of_fit(
                        data=self.data[:, i, j],
                        error=self.error[:, i, j],
                        best_fit_pars=theta,
                        n_comp=n_comp,
                    )

                    updated_fit_dict.update(gof_metrics)

                    # Save out the updated fit dictionary
                    save_pkl(updated_fit_dict, output_file)

                    if self.consolidate_fit_dict:
                        mcfine_output["fit"][i][j].update(updated_fit_dict)

                # Move the right file to the new directory. Use hardlinks to minimize space if possible
                else:
                    if hardlinks_supported:
                        os.link(input_file, output_file)
                    else:
                        os.system(f"cp {input_file} {output_file}")

            # Otherwise, load in and potentially update the dictionary
            else:

                # Do a quick comparison to see if the file has changed
                if not filecmp.cmp(input_file, output_file):
                    total_found += 1

                if self.consolidate_fit_dict:
                    updated_fit_dict = load_pkl(output_file)
                    mcfine_output["fit"][i][j].update(updated_fit_dict)

        self.logger.info(f"Number replaced: {total_found}")

        if not os.path.exists(f"{mcfine_output_filename}.pkl") or overwrite:
            save_pkl(mcfine_output, f"{mcfine_output_filename}.pkl")

        return True

    def create_fit_cube(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        cube_filename=None,
    ):
        """Create upper/lower errors for spectral fit plots

        Will preferentially get this from the mcfine_output, which is quicker and
        reduces I/O. Otherwise, will pull in each individual fit

        If WCS is present, this will be saved out as a multi-extension fits.
        Otherwise, will save as a 4-D numpy array.

        Args:
            fit_dict_filename (str): Name for the filename of fitted parameter dictionary. Defaults
                to None, which will pull from config.toml
            mcfine_output_filename (str): Name for the mcfine output.
                Defaults to None, which will pull from config.toml
            cube_filename (str): Name for the filename of output cube. Defaults
                to None, which will pull from config.toml
        """

        if self.data_type != "cube":
            self.logger.warning("Can only make fit cubes for cubes")
            sys.exit()

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
                logger=self.logger,
            )

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{mcfine_output_filename}.pkl"):
            raise FileNotFoundError(f"{mcfine_output_filename}.pkl does not exist")

        if cube_filename is None:
            cube_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="create_fit_cube",
                key="cube_filename",
                logger=self.logger,
            )

        chunksize = get_dict_val(
            self.config,
            self.config_defaults,
            table="create_fit_cube",
            key="chunksize",
            logger=self.logger,
        )

        # If we have WCS, we create fits files
        if self.wcs is not None:
            cube_filename = f"{cube_filename}.fits"
        else:
            cube_filename = f"{cube_filename}.npy"

        if not os.path.exists(cube_filename) or overwrite:

            # Load in the mcfine output as a global variable
            global glob_mcfine_output
            glob_mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")

            ij_list = [
                (i, j)
                for i in range(self.data.shape[1])
                for j in range(self.data.shape[2])
                if self.mask[i, j] != 0
            ]

            # Setup fit cube. If we have WCS info, then this is 3
            # fits cubes
            if self.wcs is not None:
                init_data = np.ones([*self.data.shape]) * np.nan
                main_hdu = fits.PrimaryHDU(header=self.wcs.to_header())
                cube_hdu = fits.ImageHDU(
                    data=init_data,
                    header=self.wcs.to_header(),
                    name="NOMINAL",
                )
                err_down_hdu = fits.ImageHDU(
                    data=init_data,
                    header=self.wcs.to_header(),
                    name="ERR_DOWN",
                )
                err_up_hdu = fits.ImageHDU(
                    data=init_data,
                    header=self.wcs.to_header(),
                    name="ERR_UP",
                )
                fit_cube = fits.HDUList([main_hdu, cube_hdu, err_down_hdu, err_up_hdu])
            else:
                fit_cube = np.zeros([3, *self.data.shape], dtype=np.float32)

            with mp.Pool(self.n_cores) as pool:
                map_result = list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_fit_samples,
                                fit_dict_filename=fit_dict_filename,
                                consolidate_fit_dict=self.consolidate_fit_dict,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                        dynamic_ncols=True,
                    )
                )

            for idx, ij in enumerate(ij_list):

                if self.wcs is not None:
                    fit_cube["ERR_DOWN"].data[:, ij[0], ij[1]] = map_result[idx][0, :]
                    fit_cube["NOMINAL"].data[:, ij[0], ij[1]] = map_result[idx][1, :]
                    fit_cube["ERR_UP"].data[:, ij[0], ij[1]] = map_result[idx][2, :]
                else:
                    fit_cube[:, :, ij[0], ij[1]] = map_result[idx]

            # Save out the cube
            if self.wcs is not None:
                fit_cube.writeto(cube_filename, overwrite=True)
            else:
                np.save(cube_filename, fit_cube)

        return True

    def make_parameter_maps(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        maps_filename=None,
    ):
        """Make maps of fitted parameters.

        Will preferentially get this from the mcfine_output, which is quicker and
        reduces I/O. Otherwise, will pull in each individual fit

        If WCS is defined, these will be .fits files, otherwise just numpy arrays

        Args:
            fit_dict_filename (str): Name for the filename of fitted parameter dictionary (without i/j suffix).
                Defaults to None, which will pull from config.toml
            mcfine_output_filename (str): Name for the mcfine output.
                Defaults to None, which will pull from config.toml
            maps_filename (str): Name for the filename of output maps. Defaults
                to None, which will pull from config.toml
        """

        if self.data_type != "cube":
            self.logger.warning("Can only make parameter maps for fitted cubes")
            sys.exit()

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
                logger=self.logger,
            )

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{mcfine_output_filename}.pkl"):
            raise FileNotFoundError(f"{mcfine_output_filename}.pkl does not exist")

        if maps_filename is None:
            maps_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="make_parameter_maps",
                key="maps_filename",
                logger=self.logger,
            )

        parameter_maps = {}

        # Define what we'll pull out to maps that aren't
        # coming from the properties themselves
        non_prop_maps = [
            "n_comp",
            "likelihood",
            "bic",
            "aic",
            "chisq",
            "chisq_red",
        ]

        if not os.path.exists(maps_filename) or overwrite:

            # Load in the mcfine output
            mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")

            # Pull 2D data shape out, since we need that later
            data_shape_2d = mcfine_output["data"]["shape"][1:]

            ij_list = [
                [i, j]
                for i in range(self.data.shape[1])
                for j in range(self.data.shape[2])
                if self.mask[i, j] != 0
            ]

            for ij in tqdm(
                ij_list,
                dynamic_ncols=True,
            ):

                i = ij[0]
                j = ij[1]

                # If we've consolidated the fit dictionary,
                # we can pull directly from the output
                if self.consolidate_fit_dict:
                    fit_dict = mcfine_output["fit"].get(i, {}).get(j, {})
                else:
                    fit_dict = load_pkl(f"{fit_dict_filename}_{i}_{j}.pkl")

                # Start looping over the parameters. If we don't have it already,
                # set it up
                for p in non_prop_maps:
                    if p not in parameter_maps:

                        f = self.setup_minimal_array(shape=data_shape_2d)
                        parameter_maps[p] = copy.deepcopy(f)

                    if self.wcs_2d is not None:
                        parameter_maps[p].data[i, j] = fit_dict[p]
                    else:
                        parameter_maps[p][i, j] = fit_dict[p]

                # Loop over properties with associated errors
                prop_key_names = [
                    "props",
                    "props_err_down",
                    "props_err_up",
                ]

                for k in prop_key_names:
                    for p in fit_dict[k]:
                        for n in range(fit_dict["n_comp"]):

                            if f"{p}_{n}" not in parameter_maps:

                                f = self.setup_minimal_array(shape=data_shape_2d)
                                parameter_maps[f"{p}_{n}"] = copy.deepcopy(f)

                            if self.wcs_2d is not None:
                                parameter_maps[f"{p}_{n}"].data[i, j] = fit_dict[
                                    "props"
                                ][p][n]
                            else:
                                parameter_maps[f"{p}_{n}"][i, j] = fit_dict["props"][p][
                                    n
                                ]

                # Also pull out covariance, if we keep that around.
                # This works a little differently
                if self.keep_covariance:
                    if "cov_matrix" not in parameter_maps:
                        parameter_maps["cov_matrix"] = {}
                    if "cov_med" not in parameter_maps:
                        parameter_maps["cov_med"] = {}
                    parameter_maps["cov_matrix"][f"{i}, {j}"] = copy.deepcopy(
                        fit_dict["cov_matrix"]
                    )
                    parameter_maps["cov_med"][f"{i}, {j}"] = copy.deepcopy(
                        fit_dict["cov_med"]
                    )

            # Save maps out
            if maps_filename is not None:
                save_pkl(parameter_maps, maps_filename)

        else:
            with open(maps_filename, "rb") as f:
                parameter_maps = pickle.load(f)

        self.parameter_maps = parameter_maps

    def setup_minimal_array(
        self,
        shape,
    ):
        """Set up a minimal array of NaNs

        If WCS is present, will create a fits file. Otherwise
        a numpy array

        Args:
            shape: Data shape

        TODO:
            For fits files, may be worth pulling in units
        """
        f = np.zeros(shape) * np.nan
        if self.wcs_2d is not None:
            hdr = self.wcs_2d.to_header()

            f = fits.PrimaryHDU(
                data=f,
                header=hdr,
            )

        return f
