import copy
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
import xarray
from astropy import units as u
from lmfit import minimize, Parameters
from scipy.interpolate import RegularGridInterpolator
from specutils import Spectrum
from specutils.fitting import find_lines_derivative
from threadpoolctl import threadpool_limits
from tqdm import tqdm

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
    save_fit_dict,
    load_fit_dict,
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

glob_initial_ncomp = np.array([])

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


def downsample(
    data,
    chunk_size=10,
    func=np.nanmean,
):
    """Downsample data, given a function

    Args:
        data: Data to downsample. Should be an image or a
        chunk_size: Chunk size to downsample to
        func: Function to downsample using. Defaults to
            nanmean
    """

    if data.ndim == 2:
        ii = np.arange(chunk_size, data.shape[0], chunk_size)
        jj = np.arange(chunk_size, data.shape[1], chunk_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downsampled_array = np.array(
                [
                    [
                        func(i_split, axis=(0, 1))
                        for i_split in np.array_split(j_split, ii, axis=0)
                    ]
                    for j_split in np.array_split(data, jj, axis=1)
                ]
            ).T
    elif data.ndim == 3:
        ii = np.arange(chunk_size, data.shape[1], chunk_size)
        jj = np.arange(chunk_size, data.shape[2], chunk_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            downsampled_array = np.array(
                [
                    [
                        func(i_split, axis=(1, 2))
                        for i_split in np.array_split(j_split, ii, axis=1)
                    ]
                    for j_split in np.array_split(data, jj, axis=2)
                ]
            ).T
    else:
        raise ValueError("Input to downsample should be a 2D or 3D array")

    return downsampled_array


def parallel_fitting(
    ij,
    fit_dict_filename="fit_dict",
    data_type="original",
    save=True,
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
        save (bool): Save out files? Defaults to True.
        overwrite (bool): Overwrite existing files? Defaults to False.

    Returns:
        Number of fitted components and the best-fit likelihood.
    """

    if data_type not in ["original", "downsampled"]:
        raise ValueError("Data type to fit should be original or downsampled")

    i = ij[0]
    j = ij[1]

    cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}"

    if not os.path.exists(f"{cube_fit_dict_filename}.pkl") or overwrite:
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

        initial_ncomp = None
        if glob_initial_ncomp.size != 0:
            initial_ncomp = glob_initial_ncomp[i, j]

        # Limit to a single core to avoid weirdness
        with threadpool_limits(limits=1, user_api=None):
            n_comp, likelihood, sampler, cov_matrix, cov_med = delta_bic_looper(
                data=data,
                error=error,
                initial_ncomp=initial_ncomp,
            )

        if save:

            fit_dict = {
                "n_comp": n_comp,
                "likelihood": likelihood,
                "sampler": None,
                "cov": {},
            }
            if glob_config["keep_sampler"]:
                fit_dict["sampler"] = sampler
            else:
                fit_dict.pop("sampler")

            if glob_config["keep_covariance"]:
                fit_dict["cov"] = {
                    "matrix": cov_matrix,
                    "med": cov_med,
                }
            else:
                fit_dict.pop("cov")

            save_fit_dict(fit_dict, f"{cube_fit_dict_filename}.pkl")

    else:

        fit_dict = load_fit_dict(f"{cube_fit_dict_filename}.pkl")

        n_comp = fit_dict["n_comp"]
        likelihood = fit_dict["likelihood"]

    logger.debug(f"N components: {n_comp}, likelihood: {likelihood:.2f}")

    return n_comp, likelihood


def delta_bic_looper(
    data,
    error,
    initial_ncomp=None,
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
        initial_ncomp: Initial guess at number of components.
            Defaults to None.
        save: Whether to save out or not, defaults to False
        overwrite: Whether to overwrite existing fits. Defaults to False
        progress: Whether to display progress bars or not. Defaults to False

    Returns:
        fitted number of components, likelihood, and the emcee sampler
    """

    delta_bic = np.inf
    delta_aic = np.inf
    sampler_old = None
    likelihood_old = None
    sampler = None
    cov_matrix = None
    cov_med = None
    prop_len = len(glob_config["props"])
    ln_m = np.log(len(data[~np.isnan(data)]))

    if initial_ncomp is None:
        # We start with a zero component model, i.e. a flat line
        n_comp = 0
        likelihood = ln_like(
            theta=0,
            intensity=data,
            intensity_err=error,
            vel=glob_vel,
            strength_lines=glob_config["strength_lines"],
            v_lines=glob_config["v_lines"],
            props=glob_config["props"],
            n_comp=n_comp,
            fit_type=glob_config["fit_type"],
        )
        bic = -2 * likelihood
        aic = -2 * likelihood

    else:
        # Start with an initial guess of component numbers
        n_comp = int(initial_ncomp)
        k = n_comp * prop_len
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

        # Calculate max likelihood parameters and BIC
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

        # Calculate BIC and AIC
        bic = k * ln_m - 2 * likelihood
        aic = 2 * k - 2 * likelihood

    while (
        delta_bic > glob_config["delta_bic_cutoff"]
        or delta_aic > glob_config["delta_aic_cutoff"]
    ):
        # Store the previous BIC and sampler, since we need them later
        bic_old = bic
        aic_old = aic
        sampler_old = sampler
        likelihood_old = likelihood

        # Increase the number of components, refit
        n_comp += 1
        k = n_comp * prop_len
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

        # Calculate max likelihood parameters and BIC
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

        # Calculate BIC and AIC
        bic = k * ln_m - 2 * likelihood
        delta_bic = bic_old - bic

        aic = 2 * k - 2 * likelihood
        delta_aic = aic_old - aic

    sampler = sampler_old
    likelihood = likelihood_old
    n_comp -= 1

    # Now loop backwards, iteratively remove the weakest component and refit. Only if we have a >0 order fit!

    if n_comp > 0:
        max_back_loops = n_comp

        for i in range(max_back_loops):
            flat_samples = get_samples(
                sampler,
                burn_in_frac=glob_config["burn_in"],
                thin_frac=glob_config["thin"],
            )
            parameter_median = np.nanmedian(
                flat_samples,
                axis=0,
            )

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
            sampler_old = sampler
            likelihood_old = likelihood

            # Remove the weakest component
            n_comp -= 1

            if n_comp == 0:
                likelihood = ln_like(
                    theta=0,
                    intensity=data,
                    intensity_err=error,
                    vel=glob_vel,
                    strength_lines=glob_config["strength_lines"],
                    v_lines=glob_config["v_lines"],
                    props=glob_config["props"],
                    n_comp=n_comp,
                    fit_type=glob_config["fit_type"],
                )
                bic = -2 * likelihood
            else:
                k = n_comp * prop_len
                idx_to_delete = range(
                    prop_len * component_order[0],
                    prop_len * component_order[0] + prop_len,
                )
                p0 = np.delete(parameter_median, idx_to_delete)
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

                # Calculate max likelihood parameters and BIC
                flat_samples = get_samples(
                    sampler,
                    burn_in_frac=glob_config["burn_in"],
                    thin_frac=glob_config["thin"],
                )
                parameter_median = np.nanmedian(flat_samples, axis=0)
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
                bic = k * ln_m - 2 * likelihood
                aic = 2 * k - 2 * likelihood

            delta_bic = bic_old - bic
            delta_aic = aic_old - aic

            # If removing and refitting doesn't significantly improve things, then just jump out of here
            if (
                delta_bic < glob_config["delta_bic_cutoff"]
                and delta_aic < glob_config["delta_aic_cutoff"]
            ):
                break

        sampler = sampler_old
        likelihood = likelihood_old
        n_comp += 1

    # Finally, calculate the covariance matrix and the parameter medians.
    # This is only defined if we end up with a 0-component fit
    if sampler is not None:
        samples = get_samples(
            sampler,
            burn_in_frac=glob_config["burn_in"],
            thin_frac=glob_config["thin"],
        )
        cov_matrix = np.cov(samples.T)
        cov_med = np.median(samples, axis=0)

    return n_comp, likelihood, sampler, cov_matrix, cov_med


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
            use some basic parameters that are likely to be suboptimal
    """

    prop_len = len(glob_config["props"])

    bounds = glob_config["bounds"] * n_comp

    if not os.path.exists(fit_dict_filename) or overwrite:

        if p0_fit is None:

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

                # Start with flat line model
                it_model = np.zeros(len(data))

                lmfit_result = None
                params = Parameters()
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

                    # Only take emission lines
                    found_lines = found_lines[found_lines["line_type"] == "emission"]

                    # Now take these lines and order by flux
                    found_line_fluxes = np.array(
                        [
                            float(np.abs(it_data[x]))
                            for x in found_lines["line_center_index"]
                        ]
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

                raise ValueError(
                    f"lmfit_method should be one of {ALLOWED_LMFIT_METHODS}"
                )

            # Sort p0 so it has monotonically increasing velocities
            v0_values = np.array(
                [p0_fit[prop_len * i + vel_idx] for i in range(n_comp)]
            )
            v0_sort = v0_values.argsort()
            p0_fit_sort = [
                p0_fit[prop_len * i : prop_len * i + prop_len] for i in v0_sort
            ]
            p0_fit = np.array([item for sublist in p0_fit_sort for item in sublist])

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

            save_fit_dict(fit_dict, fit_dict_filename)
    else:
        fit_dict = load_fit_dict(fit_dict_filename)
        sampler = fit_dict["sampler"]

    return sampler


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
    n_comp=None,
):
    """Pull fit percentiles from a single pixel"""

    if not fit_dict_filename or n_comp is None:
        logger.warning("Fit dict filename and n_comp must be defined!")
        sys.exit()

    i = ij[0]
    j = ij[1]

    n_comp_pix = int(n_comp[i, j])

    cube_sampler_filename = f"{fit_dict_filename}_{i}_{j}.pkl"

    if n_comp_pix == 0:
        return np.zeros([3, len(glob_vel)])
    fit_dict = load_fit_dict(cube_sampler_filename)

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
        n_comp=n_comp_pix,
    )

    fit_percentiles = np.nanpercentile(
        np.nansum(fit_lines, axis=-1),
        [50, 16, 84],
        axis=1,
    )

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


def parallel_map_making(
    ij,
    n_comp=None,
    fit_dict_filename="fit_dict",
    n_samples=500,
):
    """Pull parameters for map out of a single pixel"""

    if n_comp is None:
        logger.warning("n_comp should be defined!")
        sys.exit()

    i, j = ij[0], ij[1]
    n_comps_pix = int(n_comp[i, j])

    par_dict = {}

    obs = glob_data[:, i, j]
    obs_err = glob_error[:, i, j]

    if n_comps_pix > 0:

        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        fit_dict = load_fit_dict(cube_fit_dict_filename)

        flat_samples = get_samples_from_fit_dict(
            fit_dict,
            burn_in_frac=glob_config["burn_in"],
            thin_frac=glob_config["thin"],
        )

        # Pull out median and errors for each parameter and each component
        param_percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        param_diffs = np.diff(param_percentiles, axis=0)

        # Pull out model for reduced chi-square
        total_model = multiple_components(
            theta=param_percentiles[1, :],
            vel=glob_vel,
            strength_lines=glob_config["strength_lines"],
            v_lines=glob_config["v_lines"],
            props=glob_config["props"],
            n_comp=n_comps_pix,
            fit_type=glob_config["fit_type"],
        )

        for n_comp_pix in range(n_comps_pix):

            # Pull out peak intensities and errors for each component

            tpeak = np.zeros(n_samples)
            idx_offset = len(glob_config["props"])
            for sample in range(n_samples):
                choice_idx = np.random.randint(0, flat_samples.shape[0])
                theta = flat_samples[
                    choice_idx,
                    idx_offset * n_comp_pix : idx_offset * n_comp_pix + idx_offset,
                ]

                if glob_config["fit_type"] == "lte":
                    model = hyperfine_structure_lte(
                        *theta,
                        strength_lines=glob_config["strength_lines"],
                        v_lines=glob_config["v_lines"],
                        vel=glob_vel,
                    )

                elif glob_config["fit_type"] == "pure_gauss":
                    model = hyperfine_structure_pure_gauss(
                        *theta,
                        strength_lines=glob_config["strength_lines"],
                        v_lines=glob_config["v_lines"],
                        vel=glob_vel,
                    )

                elif glob_config["fit_type"] == "radex":

                    model = hyperfine_structure_radex(
                        *theta,
                        v_lines=glob_config["v_lines"],
                        vel=glob_vel,
                    )

                else:
                    logger.warning(
                        f"Fit type {glob_config['fit_type']} not understood!"
                    )
                    sys.exit()

                tpeak[sample] = np.nanmax(model)

            tpeak_percentiles = np.percentile(tpeak, [16, 50, 84], axis=0)
            tpeak_diff = np.diff(tpeak_percentiles)

            par_dict[f"tpeak_{n_comp_pix}"] = tpeak_percentiles[1]
            par_dict[f"tpeak_{n_comp_pix}_err_down"] = tpeak_diff[0]
            par_dict[f"tpeak_{n_comp_pix}_err_up"] = tpeak_diff[1]

            # Pull out fitted properties and errors for each component

            for prop_idx, prop in enumerate(glob_config["props"]):
                param_idx = n_comp_pix * len(glob_config["props"]) + prop_idx
                par_dict[f"{prop}_{n_comp_pix}"] = param_percentiles[1, param_idx]
                par_dict[f"{prop}_{n_comp_pix}_err_down"] = param_diffs[0, param_idx]
                par_dict[f"{prop}_{n_comp_pix}_err_up"] = param_diffs[1, param_idx]

            # Pull out covariance
            if glob_config["keep_covariance"]:
                par_dict["cov"] = copy.deepcopy(fit_dict["cov"])

    else:
        total_model = np.zeros_like(obs)

    # Add in reduced chisq
    chisq = chi_square(obs, total_model, obs_err)
    deg_freedom = len(obs[~np.isnan(obs)]) - (n_comps_pix * len(glob_config["props"]))
    par_dict["chisq_red"] = chisq / deg_freedom

    return par_dict


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
            data (np.ndarray): Either a 1D array of intensity (spectrum) or a 3D array of intensities (cube).
                Intensities should be in K.
            vel (np.ndarray): Array of velocity values that correspond to data, in km/s.
            error (np.ndarray): Array of errors in intensity. Should have the same shape as `data`. Defaults to None.
            mask (np.ndarray): 1/0 mask to indicate significant emission in the cube (i.e. the pixels to fit). Should
                have shape of `data.shape[1:]`. Defaults to None, which will fit all pixels in a cube.
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
        if vel is None:
            logger.warning(
                "velocity definition should be provided! Defaulting to monotonically increasing"
            )
            vel = np.arange(data.shape[0])
        if error is None:
            logger.info("No error provided. Defaulting to 0")
            error = np.zeros_like(data)

        self.logger = logger

        self.data = data
        self.error = error
        self.vel = vel

        # Define global variables for potentially huge arrays
        global glob_data, glob_error, glob_vel
        glob_data = data
        glob_error = error
        glob_vel = vel

        self.downsampled_data = np.array([])
        self.downsampled_error = np.array([])
        self.downsampled_mask = np.array([])
        self.initial_ncomp = np.array([])

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
            self.mask = mask

            if self.data.shape != self.error.shape:
                self.logger.warning("Data and error should be the same shape")
                sys.exit()
            if self.data.shape[1:] != self.mask.shape:
                self.logger.warning("Mask should be 2D projection of data")
                sys.exit()

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
        self.covariance_dict = None
        self.max_n_comp = None

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
        n_comp_filename=None,
        likelihood_filename=None,
        downsample_factor=10,
    ):
        """Run multicomponent fitter on downsampled data

        This is a light wrapper around multicomponent fitter,
        but also takes

        Args:
            fit_dict_filename: Filename to save the fitted emcee walkers
                to. Defaults to None, which will not save anything
            n_comp_filename: Filename to save out fitted number of components.
                Defaults to None, which will not save anything
            likelihood_filename: Filename to save out likelihood for the
                best fit. Defaults to None, which will not save anything
            downsample_factor: Factor to downsample the data down on each spatial axis
        """

        if self.data_type != "cube":
            raise ValueError("Can only do downsample fitting on cubes")

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        # For data and error, take means
        downsampled_data = downsample(self.data, chunk_size=downsample_factor)
        downsampled_error = downsample(self.error, chunk_size=downsample_factor)

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

        if n_comp_filename is None:
            n_comp_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="n_comp_filename",
                logger=self.logger,
            )
        if likelihood_filename is None:
            likelihood_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="likelihood_filename",
                logger=self.logger,
            )

        if not os.path.exists(f"{n_comp_filename}.npy") or overwrite:
            self.multicomponent_fitter(
                fit_dict_filename=fit_dict_filename,
                n_comp_filename=n_comp_filename,
                likelihood_filename=likelihood_filename,
                data_type="downsampled",
            )

        # Get n_comp out and sample back to the original grid
        n_comp = np.load(f"{n_comp_filename}.npy")

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

        self.initial_ncomp = copy.deepcopy(n_comp_upsampled)

        global glob_initial_ncomp
        glob_initial_ncomp = self.initial_ncomp

        return True

    def multicomponent_fitter(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        likelihood_filename=None,
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
            n_comp_filename: Filename to save out fitted number of components.
                Defaults to None, which will not save anything
            likelihood_filename: Filename to save out likelihood for the
                best fit. Defaults to None, which will not save anything
            data_type: Data type to fit. Can be either "original" or "downsampled"
        """

        if data_type not in ["original", "downsampled"]:
            raise ValueError("Data type to fit should be original or downsampled")

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        self.logger.info("Starting multi-component fitting")

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
                logger=self.logger,
            )

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

        if self.data_type == "spectrum":

            if not os.path.exists(f"{fit_dict_filename}.pkl") or overwrite:

                n_comp, likelihood, sampler, cov_matrix, cov_med = delta_bic_looper(
                    data,
                    error,
                    progress=progress,
                )
                if save:

                    fit_dict = {
                        "n_comp": n_comp,
                        "likelihood": likelihood,
                        "sampler": None,
                        "cov": {},
                    }
                    if self.keep_sampler:
                        fit_dict["sampler"] = sampler
                    else:
                        fit_dict.pop("sampler")

                    if self.keep_covariance:
                        fit_dict["cov"] = {
                            "matrix": cov_matrix,
                            "med": cov_med,
                        }
                    else:
                        fit_dict.pop("cov")

                    save_fit_dict(fit_dict, f"{fit_dict_filename}.pkl")

        elif self.data_type == "cube":

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="n_comp_filename",
                    logger=self.logger,
                )
            if likelihood_filename is None:
                likelihood_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="likelihood_filename",
                    logger=self.logger,
                )

            if not os.path.exists(f"{n_comp_filename}.npy") or overwrite:
                n_comp = np.zeros([data.shape[1], data.shape[2]])
            else:
                n_comp = np.load(f"{n_comp_filename}.npy")
            if not os.path.exists(f"{likelihood_filename}.npy") or overwrite:
                likelihood = np.zeros_like(n_comp)
            else:
                likelihood = np.load(f"{likelihood_filename}.npy")

            ij_list = [
                (i, j)
                for i in range(data.shape[1])
                for j in range(data.shape[2])
                if mask[i, j] != 0
            ]

            self.logger.info(f"Fitting using {self.n_cores} cores")

            with mp.Pool(self.n_cores) as pool:
                map_result = list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_fitting,
                                fit_dict_filename=fit_dict_filename,
                                data_type=data_type,
                                save=save,
                                overwrite=overwrite,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                    )
                )

            for idx, ij in enumerate(ij_list):
                n_comp[ij[0], ij[1]] = map_result[idx][0]
                likelihood[ij[0], ij[1]] = map_result[idx][1]

            if save:
                np.save(f"{n_comp_filename}.npy", n_comp)
                np.save(f"{likelihood_filename}.npy", likelihood)

    def encourage_spatial_coherence(
        self,
        input_dir="fit",
        output_dir="fit_coherence",
        fit_dict_filename=None,
        n_comp_filename=None,
        likelihood_filename=None,
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
            n_comp_filename: Filename for the n_comp map. Defaults to
                None, which will choose some generic name
            likelihood_filename: Filename for the likelihood map. Defaults to
                None, which will choose some generic name
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

        if n_comp_filename is None:
            n_comp_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="n_comp_filename",
                logger=self.logger,
            )
        if likelihood_filename is None:
            likelihood_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="likelihood_filename",
                logger=self.logger,
            )

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

        n_comp = np.load(os.path.join(input_dir, f"{n_comp_filename}.npy"))
        likelihood = np.load(os.path.join(input_dir, f"{likelihood_filename}.npy"))
        total_found = 0

        ij_list = [
            (i, j)
            for i in range(self.data.shape[1])[::step]
            for j in range(self.data.shape[2])[::step]
            if self.mask[i, j] != 0
        ]

        for ij in tqdm(ij_list):

            i, j = ij[0], ij[1]

            # Pull out the neighbouring pixels
            i_min = max(0, i - 1)
            i_max = min(self.data.shape[1], i + 2)
            j_min = max(0, j - 1)
            j_max = min(self.data.shape[2], j + 2)

            n_comp_cutout = n_comp[i_min:i_max, j_min:j_max]

            input_file = os.path.join(input_dir, f"{fit_dict_filename}_{i}_{j}.pkl")
            output_file = os.path.join(output_dir, f"{fit_dict_filename}_{i}_{j}.pkl")

            if not os.path.exists(output_file) or overwrite:

                n_comp_original = int(n_comp[i, j])
                ln_m = np.log(len(self.data[:, i, j][~np.isnan(self.data[:, i, j])]))

                # Pull original likelihood for the pixel
                fit_dict = load_fit_dict(input_file)
                like_original = fit_dict["likelihood"]

                bic_original = (
                    ln_m * n_comp_original * len(self.props) - 2 * like_original
                )
                aic_original = 2 * n_comp_original * len(self.props) - 2 * like_original

                delta_bic = np.zeros_like(n_comp_cutout)
                delta_aic = np.zeros_like(delta_bic)
                likelihood_cutout = np.zeros_like(delta_bic)

                for i_cutout, i_full in enumerate(range(i_min, i_max)):
                    for j_cutout, j_full in enumerate(range(j_min, j_max)):

                        if self.mask[i_full, j_full] == 0:
                            continue

                        # Pull out the parameters from the neighbouring fits, and see if that works better for the
                        # original pixel
                        n_comp_new = int(n_comp[i_full, j_full])
                        if n_comp_new > 0:

                            # Check if we already have moved the sampler file
                            cutout_fit_dict_filename = os.path.join(
                                output_dir,
                                f"{fit_dict_filename}_{i_full}_{j_full}.pkl",
                            )

                            if not os.path.exists(cutout_fit_dict_filename):
                                cutout_fit_dict_filename = os.path.join(
                                    input_dir,
                                    f"{fit_dict_filename}_{i_full}_{j_full}.pkl",
                                )
                            cutout_fit_dict = load_fit_dict(cutout_fit_dict_filename)

                            # If we have the full emcee sampler, prefer that here
                            if "sampler" in cutout_fit_dict:
                                cutout_sampler = cutout_fit_dict["sampler"]
                                flat_samples = get_samples(
                                    cutout_sampler,
                                    burn_in_frac=self.burn_in,
                                    thin_frac=self.thin,
                                )
                                pars_new = np.nanmedian(
                                    flat_samples,
                                    axis=0,
                                )

                            # Else we have these calculated as part of the covariance matrix
                            else:
                                pars_new = copy.deepcopy(cutout_fit_dict["cov"]["med"])

                            like_new = ln_like(
                                theta=pars_new,
                                intensity=self.data[:, i, j],
                                intensity_err=self.error[:, i, j],
                                vel=self.vel,
                                strength_lines=self.strength_lines,
                                v_lines=self.v_lines,
                                props=self.props,
                                n_comp=n_comp_new,
                                fit_type=self.fit_type,
                            )
                        else:
                            like_new = ln_like(
                                theta=0,
                                intensity=self.data[:, i, j],
                                intensity_err=self.error[:, i, j],
                                vel=self.vel,
                                strength_lines=self.strength_lines,
                                v_lines=self.v_lines,
                                props=self.props,
                                n_comp=n_comp_new,
                                fit_type=self.fit_type,
                            )
                        bic_new = ln_m * n_comp_new * len(self.props) - 2 * like_new
                        aic_new = 2 * n_comp_new * len(self.props) - 2 * like_new
                        delta_bic[i_cutout, j_cutout] = bic_original - bic_new
                        delta_aic[i_cutout, j_cutout] = aic_original - aic_new
                        likelihood_cutout[i_cutout, j_cutout] = like_new

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
                    n_comp[i, j] = n_comp_cutout[idx]
                    likelihood[i, j] = likelihood_cutout[idx]

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
                    updated_fit_dict = load_fit_dict(input_file)
                    updated_fit_dict["likelihood"] = likelihood[i, j]
                    save_fit_dict(updated_fit_dict, output_file)

                # Move the right file to the new directory. Use hardlinks to minimize space if possible
                else:
                    if hardlinks_supported:
                        os.link(input_file, output_file)
                    else:
                        os.system(f"cp {input_file} {output_file}")

        self.logger.info(f"Number replaced: {total_found}")

        n_comp_output_filename = os.path.join(output_dir, f"{n_comp_filename}.npy")
        likelihood_output_filename = os.path.join(
            output_dir, f"{likelihood_filename}.npy"
        )

        if not os.path.exists(n_comp_output_filename) or overwrite:
            np.save(n_comp_output_filename, n_comp)
            np.save(likelihood_output_filename, likelihood)

    def create_fit_cube(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        cube_filename=None,
    ):
        """Create upper/lower errors for spectral fit plots

        Args:
            fit_dict_filename (str): Name for the filename of fitted parameter dictionary. Defaults
                to None, which will pull from config.toml
            n_comp_filename (str): Name for the filename of component number map. Defaults
                to None, which will pull from config.toml
            cube_filename (str): Name for the filename of output cube. Defaults
                to None, which will pull from config.toml
        """

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
        if n_comp_filename is None:
            n_comp_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="n_comp_filename",
                logger=self.logger,
            )

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

        if not os.path.exists(f"{cube_filename}.npy") or overwrite:

            ij_list = [
                (i, j)
                for i in range(self.data.shape[1])
                for j in range(self.data.shape[2])
                if self.mask[i, j] != 0
            ]

            n_comp = np.load(f"{n_comp_filename}.npy")

            # Setup fit cube

            fit_cube = np.zeros([3, *self.data.shape], dtype=np.float32)

            with mp.Pool(self.n_cores) as pool:
                map_result = list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_fit_samples,
                                fit_dict_filename=fit_dict_filename,
                                n_comp=n_comp,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                    )
                )

            for idx, ij in enumerate(ij_list):
                fit_cube[:, :, ij[0], ij[1]] = map_result[idx]

            np.save(f"{cube_filename}.npy", fit_cube)

    def make_parameter_maps(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        maps_filename=None,
        cov_filename=None,
    ):
        """Make maps of fitted parameters

        Args:
            fit_dict_filename (str): Name for the filename of fitted parameter dictionary. Defaults
                to None, which will pull from config.toml
            n_comp_filename (str): Name for the filename of component number map. Defaults
                to None, which will pull from config.toml
            maps_filename (str): Name for the filename of output maps. Defaults
                to None, which will pull from config.toml
            cov_filename (str): Name for the filename of output covariance map.
                Defaults to None, which will pull from config.toml
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

        if n_comp_filename is None:
            n_comp_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="n_comp_filename",
                logger=self.logger,
            )

        if maps_filename is None:
            maps_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="make_parameter_maps",
                key="maps_filename",
                logger=self.logger,
            )

        if cov_filename is None:
            cov_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="make_parameter_maps",
                key="cov_filename",
                logger=self.logger,
            )

        n_samples = get_dict_val(
            self.config,
            self.config_defaults,
            table="make_parameter_maps",
            key="n_samples",
            logger=self.logger,
        )

        chunksize = get_dict_val(
            self.config,
            self.config_defaults,
            table="make_parameter_maps",
            key="chunksize",
            logger=self.logger,
        )

        n_comp = np.load(f"{n_comp_filename}.npy")
        max_n_comp = int(np.nanmax(n_comp))

        parameter_maps = {}
        cov_dict = {}
        if not os.path.exists(maps_filename) or overwrite:

            # Set up arrays in a dictionary

            parameter_maps["chisq_red"] = np.zeros(
                [self.data.shape[1], self.data.shape[2]]
            )
            parameter_maps["chisq_red"][parameter_maps["chisq_red"] == 0] = np.nan

            for i in range(max_n_comp):

                keys = [
                    f"tpeak_{i}",
                    f"tpeak_{i}_err_up",
                    f"tpeak_{i}_err_down",
                ]
                for key in keys:
                    parameter_maps[key] = np.zeros(
                        [self.data.shape[1], self.data.shape[2]]
                    )
                    parameter_maps[key][parameter_maps[key] == 0] = np.nan

                for prop in self.props:

                    keys = [
                        f"{prop}_{i}",
                        f"{prop}_{i}_err_up",
                        f"{prop}_{i}_err_down",
                    ]
                    for key in keys:
                        parameter_maps[key] = np.zeros(
                            [self.data.shape[1], self.data.shape[2]]
                        )
                        parameter_maps[key][parameter_maps[key] == 0] = np.nan

            # Loop over each pixel, pulling out the properties for each component, as well as the peak intensity and
            # errors for everything. Parallelize this up for speed

            ij_list = [
                [i, j]
                for i in range(self.data.shape[1])
                for j in range(self.data.shape[2])
                if self.mask[i, j] != 0
            ]

            with mp.Pool(self.n_cores) as pool:
                par_dicts = list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_map_making,
                                n_comp=n_comp,
                                fit_dict_filename=fit_dict_filename,
                                n_samples=n_samples,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                    )
                )

            # Pull the parameters into the arrays

            for dict_idx, par_dict in enumerate(par_dicts):

                i, j = ij_list[dict_idx][0], ij_list[dict_idx][1]
                for key in par_dict.keys():
                    if key in parameter_maps:
                        parameter_maps[key][i, j] = par_dict[key]

                # Also pull out covariance, if we keep that around.
                # This works a little differently since arrays might be
                # different sizes
                if self.keep_covariance:
                    cov_dict[f"{i}, {j}"] = copy.deepcopy(par_dict["cov"])

            # Save out
            if maps_filename is not None:
                with open(maps_filename, "wb") as f:
                    pickle.dump(parameter_maps, f)

            if self.keep_covariance:
                if cov_filename is not None:
                    with open(cov_filename, "wb") as f:
                        pickle.dump(cov_dict, f)

        else:
            with open(maps_filename, "rb") as f:
                parameter_maps = pickle.load(f)

            if self.keep_covariance:
                with open(cov_filename, "rb") as f:
                    cov_dict = pickle.load(f)

        self.parameter_maps = parameter_maps
        if self.keep_covariance:
            self.covariance_dict = cov_dict

        self.max_n_comp = max_n_comp
