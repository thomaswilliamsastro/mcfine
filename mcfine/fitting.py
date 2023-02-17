import copy
import glob
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from functools import partial

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import astropy.units as u
import emcee
import ndradexhyperfine as ndradex
import numpy as np
from lmfit import minimize, Parameters
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from .line_info import transition_lines, freq_lines, strength_lines, v_lines
from .utils import get_dict_val, check_overwrite, save_fit_dict, load_fit_dict

T_BACKGROUND = 2.73

ALLOWED_FIT_TYPES = [
    'lte',
    'radex',
]

ALLOWED_FIT_METHODS = [
    'mcmc',
    'leastsq',
]

ALLOWED_LINES = [
    'n2hp10',
]

CONFIG_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toml', 'config_defaults.toml')
LOCAL_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toml', 'local_defaults.toml')

radex_grid = {}

mp.set_start_method('fork')


def round_nearest(x,
                  a,
                  ):
    return round(round(x / a) * a, -int(np.floor(np.log10(a))))


def get_nearest_value(data,
                      value,
                      ):
    # Find the nearest below and above
    diff = data - value
    less = np.where(diff <= 0)
    greater = np.where(diff >= 0)

    # Get the position and value for above and below. If we're at the edge but somehow this doesn't work, just
    # go for the lowest or highest values
    if len(less[0]) == 0:
        nearest_below = data[0]
    else:
        nearest_below_idx = np.argsort(np.abs(diff[less]))[0]
        nearest_below = data[less][nearest_below_idx]

    if len(greater[0]) == 0:
        nearest_above = data[-1]
    else:
        nearest_above_idx = np.argsort(diff[greater])[0]
        nearest_above = data[greater][nearest_above_idx]

    nearest_value = np.array([nearest_below, nearest_above])

    # If we're actually choosing a grid value, just return that singular value
    if np.diff(nearest_value) == 0:
        nearest_value = nearest_value[0]

    return nearest_value


def get_nearest_values(dataset,
                       keys,
                       values,
                       ):
    nearest_values_list = [get_nearest_value(dataset[keys[i]].values, values[i])
                           for i in range(len(values))]

    return nearest_values_list


def gaussian(x,
             amp,
             centre,
             width,
             ):
    """Evaluate a Gaussian on a 1D grid.

    Calculates a Gaussian, using :math:`f(x) = A \\exp[-0.5(x - \\mu)^2/\\sigma^2]`.

    Args:
        x (np.ndarray): Grid to calculate Gaussian on.
        amp (float or np.ndarray): Height(s) of curve peak(s), :math:`A`.
        centre (float or np.ndarray): Peak centre(s), :math:`\\mu`.
        width (float or np.ndarray): Standard deviation(s), :math:`\\sigma`.

    Returns:
        np.ndarray: Gaussian model array
    """

    y = amp * np.exp(- (x - centre) ** 2 / (2 * width ** 2))
    return y


def residual(observed,
             model,
             observed_error=None,
             ):
    """TODO

    """

    if observed_error is not None:
        res = (observed - model) / observed_error
    else:
        res = (observed - model) / model

    return res


def chi_square(observed,
               model,
               observed_error=None,
               ):
    """Calculate standard chi-square.

    If errors are provided, then this is the sum of :math:`({\\rm obs}-{\\rm model})^2/{\\rm error}^2`. Else the sum of
    :math:`({\\rm obs}-{\\rm model})^2/{\\rm model}`.

    Args:
        observed (np.ndarray): Observed values.
        model (np.ndarray): Model values.
        observed_error (float or np.ndarray): The error in the observed values. Defaults to None.

    Returns:
        float: The chi-square value.

    """

    res = residual(observed, model, observed_error)
    chisq = np.nansum(res ** 2)

    return chisq


def hyperfine_structure_lte(t_ex,
                            tau,
                            v_centre,
                            line_width,
                            strength_lines,
                            v_lines,
                            vel,
                            return_hyperfine_components=False,
                            log_tau=True,
                            ):
    """Create hyperfine intensity profile.

    Takes line strengths and relative velocity centres, along with excitation temperature and optical depth to
    produce a hyperfine intensity profile.

    Args:
        t_ex (float): Excitation temperature (K).
        tau (float): Total optical depth of the line.
        v_centre (float): Central velocity of strongest component (km/s).
        line_width (float): Width of components (assumed to be the same for each hyperfine component; km/s).
        return_hyperfine_components (bool): Return the intensity for each hyperfine component. Defaults to False.

    Returns:
        Intensity for each individual hyperfine component (if `return_hyperfine_components` is True), and the total
            intensity for all components

    """

    if log_tau:
        strength = np.exp(tau) * strength_lines
    else:
        strength = tau * strength_lines

    tau_components = gaussian(vel[:, np.newaxis], strength, v_lines + v_centre,
                              line_width)

    total_tau = np.nansum(tau_components, axis=-1)
    intensity_total = (1 - np.exp(-total_tau)) * (t_ex - T_BACKGROUND)

    if not return_hyperfine_components:
        return intensity_total
    else:
        intensity_components = (1 - np.exp(-tau_components)) * (t_ex - T_BACKGROUND)
        return intensity_components, intensity_total


def hyperfine_structure_radex(t_ex,
                              tau,
                              v_centre,
                              line_width,
                              v_lines,
                              vel,
                              return_hyperfine_components=False,
                              ):
    tau_components = gaussian(vel[:, np.newaxis], tau, v_lines + v_centre,
                              line_width)

    intensity_components = (1 - np.exp(-tau_components)) * (t_ex - T_BACKGROUND)

    intensity_total = np.nansum(intensity_components, axis=-1)

    if not return_hyperfine_components:
        return intensity_total
    else:
        return intensity_components, intensity_total


def multiple_components(theta,
                        vel,
                        strength_lines,
                        v_lines,
                        props,
                        n_comp,
                        fit_type='lte',
                        log_tau=True,
                        ):
    """Sum intensities for multiple lines.

    Takes `n_comp` distinct lines, and calculates the total intensity of their various hyperfine lines.

    Args:
        theta (list): [t_ex, tau, vel, vel_width] for each component. Should have a length of 4*`n_comp`.
        vel: TODO
        props: TODO
        n_comp (int): Number of distinct components to calculate intensities for.
        fit_type: TODO

    Returns:
        np.ndarray: The sum of the intensities for all the distinct components.

    """

    prop_len = len(props)

    if n_comp == 0:
        return np.zeros_like(vel)

    if fit_type == 'lte':

        intensity_model = np.array([hyperfine_structure_lte(*theta[prop_len * i: prop_len * i + prop_len],
                                                            strength_lines, v_lines, vel, log_tau=log_tau)
                                    for i in range(n_comp)])

    elif fit_type == 'radex':

        qn_ul = np.array(range(len(radex_grid['QN_ul'].values)))

        intensity_model = [get_hyperfine_multiple_components(theta[prop_len * i: prop_len * i + prop_len],
                                                             vel, v_lines, qn_ul) for i in range(n_comp)]

    else:

        raise Warning('fit type %s not understood!' % fit_type)

    intensity_model = np.sum(intensity_model, axis=0)

    return intensity_model


def get_hyperfine_multiple_components(theta,
                                      vel,
                                      v_lines,
                                      qn_ul,
                                      ):
    # Important point here, RADEX uses a square profile so transform the sigma into the right width. Pull out
    # the subset of data around our values

    tau, t_ex = radex_grid_interp(theta, qn_ul)

    intensity_model = hyperfine_structure_radex(t_ex, tau, theta[3], theta[4], v_lines, vel)

    return intensity_model


def radex_grid_interp(theta,
                      qn_ul,
                      labels=None,
                      ):
    if labels is None:
        labels = ['T_kin', 'N_mol', 'n_H2', 'dv']

    nearest_values = get_nearest_values(
        radex_grid, labels,
        [theta[0], 10 ** theta[1], 10 ** theta[2], theta[4] * 2.355])

    # Pull out grid subset of nearest values
    grid_subset = radex_grid.sel(T_kin=nearest_values[0], N_mol=nearest_values[1], n_H2=nearest_values[2],
                                 dv=nearest_values[3])
    tau_subset = grid_subset['tau'].values
    t_ex_subset = grid_subset['T_ex'].values

    # Remove any singular (limit) values
    limit_vals = [True if type(nearest_value) == np.ndarray else False for nearest_value in nearest_values]

    theta_coords = np.array([theta[0], 10 ** theta[1], 10 ** theta[2], theta[4] * 2.355])
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


def initial_fit(*args):
    """TODO

    """

    return -ln_like(*args)


def initial_lmfit(params,
                  intensity,
                  intensity_err,
                  vel,
                  strength_lines,
                  v_lines,
                  props,
                  n_comp=1,
                  fit_type='lte',
                  log_tau=True,
                  ):
    theta = np.array([params[key].value for key in params])

    intensity_model = multiple_components(theta, vel, strength_lines, v_lines, props, n_comp=n_comp, fit_type=fit_type,
                                          log_tau=log_tau)
    residual = (intensity - intensity_model) / intensity_err

    return residual


def ln_like(theta,
            intensity,
            intensity_err,
            vel,
            strength_lines,
            v_lines,
            props,
            n_comp=1,
            fit_type='lte',
            log_tau=True,
            ):
    """TODO: Docstring.

    Args:
        theta:
        intensity:
        intensity_err:
        n_comp:

    Returns:
        float: Negative ln-likelihood for the model.

    """

    intensity_model = multiple_components(theta, vel, strength_lines, v_lines, props, n_comp=n_comp, fit_type=fit_type,
                                          log_tau=log_tau)
    chisq = chi_square(intensity, intensity_model, intensity_err)

    # # Scale the chisq by sqrt(2[N-P]), following Smith+ and others
    # p = n_comp * len(props)
    # n = len(intensity[~np.isnan(intensity)])
    # scale_factor = np.sqrt(2 * (n - p))
    # chisq /= scale_factor

    return -0.5 * chisq


def ln_prob(theta,
            intensity,
            intensity_err,
            vel,
            strength_lines,
            v_lines,
            props,
            bounds,
            n_comp=1,
            fit_type='lte',
            ):
    """TODO: Docstring

    Args:
        theta:
        intensity:
        intensity_err:
        n_comp:

    Returns:
        float: Combined probability (likelihood+prior) for the model.

    """
    lp = ln_prior(theta, vel, props, bounds, n_comp=n_comp)
    if not np.isfinite(lp):
        return -np.inf
    like = ln_like(theta, intensity, intensity_err, vel, strength_lines, v_lines, props,
                   n_comp=n_comp, fit_type=fit_type)
    return lp + like


def ln_prior(theta,
             vel,
             props,
             bounds,
             n_comp=1,
             ):
    """TODO: Docstring

    Args:
        theta:
        n_comp:

    Returns:

    """

    prop_len = len(props)

    for prop in range(prop_len):

        values = np.array([theta[prop_len * i + prop] for i in range(n_comp)])

        if not np.logical_and(bounds[prop][0] <= values, values <= bounds[prop][1]).all():
            return -np.inf

    if n_comp > 1:

        dv = np.abs(np.nanmedian(np.diff(vel)))

        # Insist on monotonically increasing velocity components TODO?, and separated by at least one channel
        v_idx = np.where(np.asarray(props) == 'v')[0][0]

        vels = np.array([theta[prop_len * i + v_idx] for i in range(n_comp)])
        vel_diffs = np.diff(vels)

        if np.any(vel_diffs < dv):
            # if np.any(vel_diffs < 0):
            return -np.inf

    return 0


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

        TODO: Long description

        Args:
            data (np.ndarray): Either a 1D array of intensity (spectrum) or a 3D array of intensities (cube).
                Intensities should be in K.
            vel (np.ndarray): Array of velocity values that correspond to data, in km/s.
            error (np.ndarray): Array of errors in intensity. Should have the same shape as `data`. Defaults to None.
            mask (np.ndarray): 1/0 mask to indicate significant emission in the cube (i.e. the pixels to fit). Should
                have shape of `data.shape[1:]`. Defaults to None, which will fit all pixels in a cube.
            config_file (str): TODO
            local_file (str): TODO

        Todo:
            * Bounds as parameters to input here.
            * Test the radex fitting routines on cubes.
            * Covariance matrices for the MCMC chains so we can plot errors for each component
            * Add support for different lines

        """

        logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)s] %(asctime)s - %(name)s - %(funcName)s - %(message)s',
                            )
        self.logger = logging.getLogger(__name__)

        if data is None:
            self.logger.warning('data should be provided!')
            sys.exit()
        if vel is None:
            self.logger.warning('velocity definition should be provided! Defaulting to monotonically increasing')
            vel = np.arange(data.shape[0])
        if error is None:
            self.logger.info('No error provided. Defaulting to 0')
            error = np.zeros_like(data)

        self.data = data
        self.vel = vel
        self.error = error

        with open(CONFIG_DEFAULT_PATH, 'rb') as f:
            config_defaults = tomllib.load(f)
        with open(LOCAL_DEFAULT_PATH, 'rb') as f:
            local_defaults = tomllib.load(f)

        if config_file is not None:
            with open(config_file, 'rb') as f:
                config = tomllib.load(f)
        else:
            self.logger.info('No config file provided. Using in-built defaults')
            config = copy.deepcopy(config_defaults)

        if local_file is not None:
            with open(local_file, 'rb') as f:
                local = tomllib.load(f)
        else:
            self.logger.info('No local file provided. Using in-built defaults')
            local = copy.deepcopy(local_defaults)

        self.config = config
        self.local = local

        self.config_defaults = config_defaults
        self.local_defaults = local_defaults

        fit_type = get_dict_val(self.config,
                                self.config_defaults,
                                table='fitting_params',
                                key='fit_type',
                                logger=self.logger,
                                )

        if fit_type not in ALLOWED_FIT_TYPES:
            self.logger.warning('Fit type %s not understood!' % fit_type)
            sys.exit()

        self.fit_type = fit_type

        fit_method = get_dict_val(self.config,
                                  self.config_defaults,
                                  table='fitting_params',
                                  key='fit_method',
                                  logger=self.logger,
                                  )

        if fit_method not in ALLOWED_FIT_METHODS:
            self.logger.warning('Fitting procedure %s not understood!' % fit_method)
            sys.exit()

        self.fit_method = fit_method

        self.dv = np.abs(np.nanmedian(np.diff(self.vel)))

        if self.data.ndim == 1:
            self.data_type = 'spectrum'
        else:
            self.data_type = 'cube'

        self.logger.info('Detected data type as %s' % self.data_type)

        if self.data_type == 'cube':

            if mask is None:
                self.logger.info('No mask provided. Including every pixel')
                mask = np.ones([data.shape[1], data.shape[2]])
            self.mask = mask

            if self.data.shape != self.error.shape:
                self.logger.warning('Data and error should be the same shape')
                sys.exit()
            if self.data.shape[1:] != self.mask.shape:
                self.logger.warning('Mask should be 2D projection of data')
                sys.exit()

        line = get_dict_val(self.config,
                            self.config_defaults,
                            table='fitting_params',
                            key='line',
                            logger=self.logger,
                            )

        if line not in ALLOWED_LINES:
            self.logger.warning('Line %s not understood!' % line)
            sys.exit()

        self.line = line

        if self.fit_type == 'lte':

            self.strength_lines = np.array(
                [strength_lines[line_name] for line_name in strength_lines.keys() if self.line in line_name]
            )
            self.v_lines = np.array(
                [v_lines[line_name] for line_name in v_lines.keys() if self.line in line_name]
            )
            self.bounds = [
                (T_BACKGROUND, 1e3),
                (np.log(0.1), np.log(30)),
                (np.nanmin(self.vel), np.nanmax(self.vel)),
                (self.dv / 2.355, 10),
            ]

        elif self.fit_type == 'radex':

            rest_freq = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='fitting_params',
                                     key='rest_freq',
                                     logger=self.logger,
                                     )

            if rest_freq == '':
                self.logger.warning('rest_freq should be defined')
                sys.exit()
            if isinstance(rest_freq, float) or isinstance(rest_freq, int):
                rest_freq = rest_freq * u.GHz

            self.strength_lines = None
            self.transitions = [
                transition_lines[line_name] for line_name in transition_lines.keys() if self.line in line_name
            ]
            freq = np.array(
                [freq_lines[line_name] for line_name in freq_lines.keys() if self.line in line_name]
            ) * u.GHz
            freq_to_vel = u.doppler_radio(rest_freq)
            self.v_lines = freq.to(u.km / u.s, equivalencies=freq_to_vel).value
            self.bounds = [
                (T_BACKGROUND, 75),
                (13, 15), (5, 8),
                (np.nanmin(self.vel), np.nanmax(self.vel)),
                (self.dv / 2.355, 10),
            ]

            radex_datafile = get_dict_val(self.local,
                                          self.local_defaults,
                                          table='local',
                                          key='radex_datafile',
                                          logger=self.logger,
                                          )

            self.radex_datafile = radex_datafile

        p0 = get_dict_val(self.config,
                          self.config_defaults,
                          table='initial_guess',
                          key=fit_type,
                          logger=self.logger,
                          )
        self.p0 = p0

        if not len(self.bounds) == len(self.p0):
            self.logger.warning('bounds and p0 should have the same length!')
            sys.exit()

        delta_bic_cutoff = get_dict_val(self.config,
                                        self.config_defaults,
                                        table='mcmc',
                                        key='delta_bic_cutoff',
                                        logger=self.logger,
                                        )

        self.delta_bic_cutoff = delta_bic_cutoff

        n_steps = get_dict_val(self.config,
                               self.config_defaults,
                               table='mcmc',
                               key='n_steps',
                               logger=self.logger,
                               )

        self.n_steps = n_steps

        n_walkers = get_dict_val(self.config,
                                 self.config_defaults,
                                 table='mcmc',
                                 key='n_walkers',
                                 logger=self.logger,
                                 )

        self.n_walkers = n_walkers

        n_cores = get_dict_val(self.local,
                               self.local_defaults,
                               table='local',
                               key='n_cores',
                               logger=self.logger,
                               )

        if n_cores == '':
            self.n_cores = mp.cpu_count()
        else:
            self.n_cores = n_cores

        if self.fit_type == 'lte':
            self.props = [
                'tex',
                'tau',
                'v',
                'sigma',
            ]
            self.labels = [
                r'$T_\mathrm{ex}$ (%s)',
                r'$\log(\tau)$ (%s)',
                r'$v$ (%s)',
                r'$\sigma$ (%s)',
            ]

        elif self.fit_type == 'radex':
            self.props = [
                't_kin',
                'N_col',
                'n_h2',
                'v',
                'sigma',
            ]
            self.labels = [
                r'$T_\mathrm{kin}$ (%s)',
                r'$N_\mathrm{col}$ (%s)',
                r'$n_\mathrm{H2}$ (%s)',
                r'$v$ (%s)',
                r'$\sigma$ (%s)',
            ]
        else:
            self.logger.warning('fit_type %s not known' % self.fit_type)
            sys.exit()

        self.parameter_maps = None
        self.max_n_comp = None

    def generate_radex_grid(self,
                            output_file='radex_output.nc',
                            t_kin=None,
                            t_kin_step=5,
                            n_mol=None,
                            n_mol_step=0.2,
                            n_h2=None,
                            n_h2_step=0.2,
                            dv=None,
                            dv_step=1,
                            geom='uni',
                            progress=True,
                            ):
        """TODO: This needs to be optimized. Doesn't matter if the grid is kinda massive so long as it works, hey

        Args:
            output_file:
            t_kin:
            t_kin_step:
            n_mol:
            n_mol_step:
            n_h2:
            n_h2_step:
            dv:
            dv_step:

        """

        if self.fit_type != 'radex':
            self.logger.warning('fit_type should be radex')
            sys.exit()

        if t_kin is None:
            t_kin = self.bounds[0]
        if n_mol is None:
            n_mol = self.bounds[1]
        if n_h2 is None:
            n_h2 = self.bounds[2]
        if dv is None:
            dv = self.bounds[4]

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        if not os.path.exists(output_file) or overwrite:

            t_kin_array = np.arange(t_kin[0], t_kin[1] + t_kin_step, t_kin_step)
            n_mol_array = np.arange(n_mol[0], n_mol[1] + n_mol_step, n_mol_step)
            n_h2_array = np.arange(n_h2[0], n_h2[1] + n_h2_step, n_h2_step)
            dv_array = np.arange(dv[0], dv[1] + dv_step, dv_step) * 2.355  # 2.355 since distinction between sigma/dv

            ds = ndradex.run(self.radex_datafile,
                             self.transitions,
                             T_kin=t_kin_array,
                             N_mol=10 ** n_mol_array,
                             n_H2=10 ** n_h2_array,
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

    def multicomponent_fitter(self,
                              fit_dict_filename=None,
                              n_comp_filename=None,
                              likelihood_filename=None,
                              ):
        """TODO: Docstring.

        """

        f_name = inspect.currentframe().f_code.co_name
        overwrite = check_overwrite(self.config, f_name)

        self.logger.info('Starting multi-component fitting')

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(self.config,
                                             self.config_defaults,
                                             table='multicomponent_fitter',
                                             key='fit_dict_filename',
                                             logger=self.logger,
                                             )

        progress = get_dict_val(self.config,
                                self.config_defaults,
                                table='multicomponent_fitter',
                                key='progress',
                                logger=self.logger,
                                )

        save = get_dict_val(self.config,
                            self.config_defaults,
                            table='multicomponent_fitter',
                            key='save',
                            logger=self.logger,
                            )

        if self.data_type == 'spectrum':

            if not os.path.exists(fit_dict_filename + '.pkl') or overwrite:
                data = self.data
                error = self.error

                n_comp, likelihood, sampler = self.delta_bic_looper(data,
                                                                    error,
                                                                    progress=progress,
                                                                    )
                if save:
                    fit_dict = {'sampler': sampler,
                                'n_comp': n_comp,
                                'likelihood': likelihood,
                                }

                    save_fit_dict(fit_dict, fit_dict_filename + '.pkl')

        elif self.data_type == 'cube':

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(self.config,
                                               self.config_defaults,
                                               table='multicomponent_fitter',
                                               key='n_comp_filename',
                                               logger=self.logger,
                                               )
            if likelihood_filename is None:
                likelihood_filename = get_dict_val(self.config,
                                                   self.config_defaults,
                                                   table='multicomponent_fitter',
                                                   key='likelihood_filename',
                                                   logger=self.logger,
                                                   )

            if not os.path.exists(n_comp_filename + '.npy') or overwrite:
                n_comp = np.zeros([self.data.shape[1], self.data.shape[2]])
            else:
                n_comp = np.load(n_comp_filename + '.npy')
            if not os.path.exists(likelihood_filename + '.npy') or overwrite:
                likelihood = np.zeros_like(n_comp)
            else:
                likelihood = np.load(likelihood_filename + '.npy')

            ij_list = [(i, j)
                       for i in range(self.data.shape[1])
                       for j in range(self.data.shape[2])
                       if self.mask[i, j] != 0
                       ]

            with mp.Pool(self.n_cores) as pool:
                map_result = list(
                    tqdm(
                        pool.imap(
                            partial(self.parallel_fitting,
                                    fit_dict_filename=fit_dict_filename,
                                    save=save,
                                    overwrite=overwrite),
                            ij_list),
                        total=len(ij_list)
                    )
                )

            for idx, ij in enumerate(ij_list):
                n_comp[ij[0], ij[1]] = map_result[idx][0]
                likelihood[ij[0], ij[1]] = map_result[idx][1]

            if save:
                np.save(n_comp_filename + '.npy', n_comp)
                np.save(likelihood_filename + '.npy', likelihood)

    def parallel_fitting(self,
                         ij,
                         fit_dict_filename='fit_dict',
                         save=True,
                         overwrite=False,
                         ):
        """Parallel function for MCMC fitting.

        Wraps up the MCMC fitting to pass off to multiple cores. Because of overheads, it's easier to farm out multiple
        fits to multiple cores, rather than run the MCMC with multiple threads.

        Args:
            ij (tuple): tuple containing (i, j) coordinates of the pixel to fit.
            fit_dict_filename (str): Base filename for MCMC ft pickle. Will append coordinates on afterwards. Defaults
                to fit_dict.
            save (bool): Save out files? Defaults to True.
            overwrite (bool): Overwrite existing files? Defaults to False.

        Returns:
            Number of fitted components and the best-fit likelihood.

        """

        i = ij[0]
        j = ij[1]

        cube_fit_dict_filename = fit_dict_filename + '_%s_%s' % (i, j)

        if not os.path.exists(cube_fit_dict_filename + '.pkl') or overwrite:
            self.logger.info('Fitting %s, %s' % (i, j))
            data = self.data[:, i, j]
            error = self.error[:, i, j]

            # Limit to a single core to avoid weirdness
            with threadpool_limits(limits=1, user_api='blas'):
                n_comp, likelihood, sampler = self.delta_bic_looper(data=data,
                                                                    error=error,
                                                                    )

            if save:
                fit_dict = {'sampler': sampler,
                            'n_comp': n_comp,
                            'likelihood': likelihood,
                            }

                save_fit_dict(fit_dict, fit_dict_filename + '.pkl')

        else:

            fit_dict = load_fit_dict(fit_dict_filename + '.pkl')

            n_comp = fit_dict['n_comp']
            likelihood = fit_dict['likelihood']

        self.logger.info('N components: %d, likelihood: %.2f' % (n_comp, likelihood))

        return n_comp, likelihood

    def delta_bic_looper(self,
                         data,
                         error,
                         save=False,
                         overwrite=True,
                         progress=False,
                         ):
        """TODO: Docstring

        """

        # We start with a zero component model, i.e. a flat line

        delta_bic = np.inf
        sampler_old = None
        likelihood_old = None
        sampler = None
        n_comp = 0
        prop_len = len(self.props)
        ln_m = np.log(len(data[~np.isnan(data)]))
        likelihood = ln_like(theta=0,
                             intensity=data,
                             intensity_err=error,
                             vel=self.vel,
                             strength_lines=self.strength_lines,
                             v_lines=self.v_lines,
                             props=self.props,
                             n_comp=n_comp,
                             fit_type=self.fit_type,
                             )
        bic = - 2 * likelihood

        while delta_bic > self.delta_bic_cutoff:
            # Store the previous BIC and sampler, since we need them later
            bic_old = bic
            sampler_old = sampler
            likelihood_old = likelihood

            # Increase the number of components, refit
            n_comp += 1
            k = n_comp * prop_len
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sampler = self.run_mcmc(data,
                                        error,
                                        n_comp=n_comp,
                                        save=save,
                                        overwrite=overwrite,
                                        progress=progress,
                                        )

            # Calculate max likelihood parameters and BIC
            flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                             flat=True,
                                             )
            parameter_median = np.nanmedian(flat_samples,
                                            axis=0,
                                            )
            likelihood = ln_like(theta=parameter_median,
                                 intensity=data,
                                 intensity_err=error,
                                 vel=self.vel,
                                 strength_lines=self.strength_lines,
                                 v_lines=self.v_lines,
                                 props=self.props,
                                 n_comp=n_comp,
                                 fit_type=self.fit_type,
                                 )
            bic = k * ln_m - 2 * likelihood
            delta_bic = bic_old - bic

        sampler = sampler_old
        likelihood = likelihood_old
        n_comp -= 1

        # Now loop backwards, iteratively remove the weakest component and refit. Only if we have a >0 order fit!

        if n_comp > 0:
            max_back_loops = n_comp

            for i in range(max_back_loops):
                flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                                 flat=True,
                                                 )
                parameter_median = np.nanmedian(flat_samples,
                                                axis=0,
                                                )

                if self.fit_type == 'lte':
                    line_intensities = np.array(
                        [hyperfine_structure_lte(*parameter_median[prop_len * i: prop_len * i + prop_len],
                                                 strength_lines=self.strength_lines,
                                                 v_lines=self.v_lines,
                                                 vel=self.vel,
                                                 )
                         for i in range(n_comp)]
                    )

                elif self.fit_type == 'radex':
                    line_intensities = np.array(
                        [hyperfine_structure_radex(*parameter_median[prop_len * i: prop_len * i + prop_len],
                                                   v_lines=self.v_lines,
                                                   vel=self.vel,
                                                   )
                         for i in range(n_comp)]
                    )
                else:
                    self.logger.warning('Fit type %s not understood!' % self.fit_type)
                    sys.exit()

                integrated_intensities = np.trapz(line_intensities, x=self.vel, axis=-1)
                component_order = np.argsort(integrated_intensities)

                bic_old = bic
                sampler_old = sampler
                likelihood_old = likelihood

                # Remove the weakest component
                n_comp -= 1

                if n_comp == 0:
                    likelihood = ln_like(theta=0,
                                         intensity=data,
                                         intensity_err=error,
                                         vel=self.vel,
                                         strength_lines=self.strength_lines,
                                         v_lines=self.v_lines,
                                         props=self.props,
                                         n_comp=n_comp,
                                         fit_type=self.fit_type)
                    bic = - 2 * likelihood
                else:
                    k = n_comp * prop_len
                    idx_to_delete = range(prop_len * component_order[0], prop_len * component_order[0] + prop_len)
                    p0 = np.delete(parameter_median, idx_to_delete)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        sampler = self.run_mcmc(data,
                                                error,
                                                n_comp=n_comp,
                                                save=save,
                                                overwrite=overwrite,
                                                progress=progress,
                                                p0_fit=p0,
                                                )

                    # Calculate max likelihood parameters and BIC
                    flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                                     flat=True,
                                                     )
                    parameter_median = np.nanmedian(flat_samples, axis=0)
                    likelihood = ln_like(theta=parameter_median,
                                         intensity=data,
                                         intensity_err=error,
                                         vel=self.vel,
                                         strength_lines=self.strength_lines,
                                         v_lines=self.v_lines,
                                         props=self.props,
                                         n_comp=n_comp,
                                         fit_type=self.fit_type,
                                         )
                    bic = k * ln_m - 2 * likelihood
                delta_bic = bic_old - bic

                # If removing and refitting doesn't significantly improve things, then just jump out of here
                if delta_bic < self.delta_bic_cutoff:
                    break

            sampler = sampler_old
            likelihood = likelihood_old
            n_comp += 1

        return n_comp, likelihood, sampler

    def run_mcmc(self,
                 data,
                 error,
                 n_comp=1,
                 save=True,
                 overwrite=False,
                 progress=False,
                 fit_dict_filename='fit_dict.pkl',
                 p0_fit=None,
                 ):

        prop_len = len(self.props)

        bounds = self.bounds * n_comp

        if not os.path.exists(fit_dict_filename) or overwrite:

            if p0_fit is None:

                vel_idx = np.where(np.array(self.props) == 'v')[0][0]

                p0 = np.array(self.p0 * n_comp)

                # Move the velocities a little to encourage parameter space hunting

                for i in range(n_comp):
                    p0[prop_len * i + vel_idx] += i

                # Use lmfit to get an initial fit

                params = Parameters()
                p0_idx = 0
                for i in range(n_comp):
                    for j in range(prop_len):
                        params.add('%s_%d' % (self.props[j], i),
                                   value=p0[p0_idx],
                                   min=bounds[j][0],
                                   max=bounds[j][1],
                                   )
                        p0_idx += 1

                # Pull in any relevant kwargs

                kwargs = {}

                for config_dict in [self.config_defaults, self.config]:
                    if 'lmfit' in config_dict:
                        for key in config_dict['lmfit']:
                            kwargs[key] = config_dict['lmfit'][key]

                lmfit_result = minimize(
                    fcn=initial_lmfit,
                    params=params,
                    args=(data,
                          error,
                          self.vel,
                          self.strength_lines,
                          self.v_lines,
                          self.props,
                          n_comp,
                          self.fit_type,
                          True,
                          ),
                    **kwargs,
                )

                p0_fit = np.array([lmfit_result.params[key].value for key in lmfit_result.params])

                # Sort p0 so it has monotonically increasing velocities
                v0_values = np.array(
                    [p0_fit[prop_len * i + vel_idx] for i in range(n_comp)]
                )
                v0_sort = v0_values.argsort()
                p0_fit_sort = [p0_fit[prop_len * i: prop_len * i + prop_len]
                               for i in v0_sort]
                p0_fit = [item
                          for sublist in p0_fit_sort
                          for item in sublist]

            n_dims = len(p0_fit)

            # Shuffle the parameters around a little. For values very close to 0, don't base this on the value itself
            p0_movement = np.max(np.array([0.01 * np.abs(p0_fit), 0.01 * np.ones_like(p0_fit)]),
                                 axis=0,
                                 )
            pos = np.array(p0_fit) + p0_movement * np.random.randn(self.n_walkers, n_dims)

            # Enforce positive values for t_ex, width for the LTE fitting and

            if self.fit_type == 'lte':
                positive_idx = [0, 3]
            elif self.fit_type == 'radex':
                positive_idx = [0, 1, 4]
            else:
                self.logger.warning('Fit type %s not understood!' % self.fit_type)
                sys.exit()

            prop_len = len(self.props)

            enforced_positives = [[prop_len * i + j]
                                  for i in range(n_comp)
                                  for j in positive_idx]
            enforced_positives = [item
                                  for sublist in enforced_positives
                                  for item in sublist]

            for i in enforced_positives:
                pos[:, i] = np.abs(pos[:, i])

            sampler = self.emcee_wrapper(data,
                                         error,
                                         pos=pos,
                                         n_dims=n_dims,
                                         n_comp=n_comp,
                                         progress=progress,
                                         )

            if save:
                # Calculate max likelihood
                flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                                 flat=True,
                                                 )
                parameter_median = np.nanmedian(flat_samples,
                                                axis=0,
                                                )
                likelihood = ln_like(theta=parameter_median,
                                     intensity=data,
                                     intensity_err=error,
                                     vel=self.vel,
                                     strength_lines=self.strength_lines,
                                     v_lines=self.v_lines,
                                     props=self.props,
                                     n_comp=n_comp,
                                     fit_type=self.fit_type,
                                     )

                fit_dict = {'sampler': sampler,
                            'n_comp': n_comp,
                            'likelihood': likelihood,
                            }

                save_fit_dict(fit_dict, fit_dict_filename)
        else:
            fit_dict = load_fit_dict(fit_dict_filename)
            sampler = fit_dict['sampler']

        return sampler

    def emcee_wrapper(self,
                      data,
                      error,
                      pos,
                      n_dims,
                      n_comp=1,
                      progress=False,
                      ):

        if self.data_type == 'spectrum':

            # Multiprocess here for speed

            with mp.Pool(self.n_cores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers,
                                                ndim=n_dims,
                                                log_prob_fn=ln_prob,
                                                args=(
                                                    data,
                                                    error,
                                                    self.vel,
                                                    self.strength_lines,
                                                    self.v_lines,
                                                    self.props,
                                                    self.bounds,
                                                    n_comp,
                                                    self.fit_type,
                                                ),
                                                moves=[(emcee.moves.DEMove(), 0.8),
                                                       (emcee.moves.DESnookerMove(), 0.2)
                                                       ],
                                                pool=pool,
                                                )

                # Run burn-in
                state = sampler.run_mcmc(pos,
                                         self.n_steps // 4,
                                         progress=progress,
                                         )
                sampler.reset()

                # Do the full run
                sampler.run_mcmc(state,
                                 self.n_steps,
                                 progress=progress,
                                 )

        else:

            # Run in serial since the cube is multiprocessing already (no daemons here, not today satan)
            sampler = emcee.EnsembleSampler(nwalkers=self.n_walkers,
                                            ndim=n_dims,
                                            log_prob_fn=ln_prob,
                                            args=(
                                                data,
                                                error,
                                                self.vel,
                                                self.strength_lines,
                                                self.v_lines,
                                                self.props,
                                                self.bounds,
                                                n_comp,
                                                self.fit_type,
                                            ),
                                            moves=[(emcee.moves.DEMove(), 0.8),
                                                   (emcee.moves.DESnookerMove(), 0.2)
                                                   ],
                                            )

            # Run burn-in
            state = sampler.run_mcmc(pos,
                                     self.n_steps // 4,
                                     progress=progress,
                                     )
            sampler.reset()

            # Do the full run
            sampler.run_mcmc(state,
                             self.n_steps,
                             progress=progress,
                             )

        return sampler

    def encourage_spatial_coherence(self, sampler_filename='sampler',
                                    original_fit_dir='original', output_fit_dir='coherence',
                                    n_comp_filename='n_comp', likelihood_filename='likelihood',
                                    reverse_direction=False, overwrite=False):

        if self.data_type != 'cube':
            self.logger.warning('Can only do spatial coherence on a cube!')
            sys.exit()

        if reverse_direction:
            step = -1
        else:
            step = 1

        if overwrite:

            # Flush out the directory
            files_in_dir = glob.glob(output_fit_dir + '/*')
            for file_in_dir in files_in_dir:
                os.remove(file_in_dir)

        n_comp = np.load(os.path.join(original_fit_dir, n_comp_filename + '.npy'))
        likelihood = np.load(os.path.join(original_fit_dir, likelihood_filename + '.npy'))
        total_found = 0

        ij_list = [(i, j) for i in range(self.data.shape[1])[::step] for j in range(self.data.shape[2])[::step]
                   if self.mask[i, j] != 0]

        for ij in tqdm(ij_list):

            i, j = ij[0], ij[1]

            # Pull out the neighbouring pixels
            i_min = max(0, i - 1)
            i_max = min(self.data.shape[1], i + 2)
            j_min = max(0, j - 1)
            j_max = min(self.data.shape[2], j + 2)

            n_comp_cutout = n_comp[i_min: i_max, j_min: j_max]

            input_file = os.path.join(original_fit_dir, sampler_filename + '_%s_%s.pkl' % (i, j))
            output_file = os.path.join(output_fit_dir, sampler_filename + '_%s_%s.pkl' % (i, j))

            if not os.path.exists(output_file) or overwrite:

                n_comp_original = int(n_comp[i, j])
                ln_m = np.log(len(self.data[:, i, j][~np.isnan(self.data[:, i, j])]))

                # Pull original likelihood for the pixel
                if n_comp_original > 0:
                    sampler = load_fit_dict(input_file)
                    flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
                    pars_original = np.nanmedian(flat_samples, axis=0)
                    like_original = ln_like(pars_original, self.data[:, i, j], self.error[:, i, j], self.vel,
                                            self.strength_lines, self.v_lines, self.props, n_comp_original,
                                            self.fit_type)
                else:
                    like_original = ln_like(0, self.data[:, i, j], self.error[:, i, j], self.vel,
                                            self.strength_lines, self.v_lines, self.props, n_comp_original,
                                            self.fit_type)

                bic_original = ln_m * n_comp_original * len(self.props) - 2 * like_original

                delta_bic = np.zeros_like(n_comp_cutout)
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
                            cube_sampler_filename = os.path.join(output_fit_dir,
                                                                 sampler_filename + '_%s_%s.pkl' % (i_full, j_full))

                            if not os.path.exists(cube_sampler_filename):
                                cube_sampler_filename = os.path.join(original_fit_dir,
                                                                     sampler_filename + '_%s_%s.pkl'
                                                                     % (i_full, j_full))
                            sampler = load_fit_dict(cube_sampler_filename)
                            flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
                            pars_new = np.nanmedian(flat_samples, axis=0)
                            like_new = ln_like(pars_new, self.data[:, i, j], self.error[:, i, j], self.vel,
                                               self.strength_lines, self.v_lines, self.props, n_comp_new,
                                               self.fit_type)
                        else:
                            like_new = ln_like(0, self.data[:, i, j], self.error[:, i, j], self.vel,
                                               self.strength_lines, self.v_lines, self.props, n_comp_new,
                                               self.fit_type)
                        bic_new = ln_m * n_comp_new * len(self.props) - 2 * like_new
                        delta_bic[i_cutout, j_cutout] = bic_original - bic_new
                        likelihood_cutout[i_cutout, j_cutout] = like_new

                # Find the maximum change in BIC between pixels. Set ones where we don't have a meaningful value to
                # something small
                delta_bic[delta_bic == 0] = -1000
                idx = np.unravel_index(np.argmax(delta_bic, axis=None), delta_bic.shape)
                if delta_bic[idx] > self.delta_bic_cutoff:
                    total_found += 1
                    n_comp[i, j] = n_comp_cutout[idx]
                    likelihood[i, j] = likelihood_cutout[idx]

                    # If we're replacing with a file we've already replaced, pull from the output directory. Else
                    # pull from the input directory.

                    input_file = os.path.join(output_fit_dir,
                                              sampler_filename + '_%s_%s.pkl' % (idx[0] + i_min, idx[1] + j_min))
                    if not os.path.exists(input_file):
                        input_file = os.path.join(original_fit_dir, sampler_filename + '_%s_%s.pkl' %
                                                  (idx[0] + i_min, idx[1] + j_min))

                # Move the right file to the new directory

                os.system('cp %s %s' % (input_file, output_file))

        self.logger.info('Number replaced: %d' % total_found)

        n_comp_output_filename = os.path.join(output_fit_dir, n_comp_filename + '.npy')
        likelihood_output_filename = os.path.join(output_fit_dir, likelihood_filename + '.npy')

        if not os.path.exists(n_comp_output_filename) or overwrite:
            np.save(n_comp_output_filename, n_comp)
            np.save(likelihood_output_filename, likelihood)

    def get_fits_from_samples(self, samples, vel, n_draws=100, n_comp=1):
        """TODO: Docstring

        """

        fit_lines = np.zeros([len(vel), n_draws, n_comp])

        if self.fit_type == 'radex':
            qn_ul = np.array(range(len(radex_grid['QN_ul'].values)))

        for draw in range(n_draws):
            sample = np.random.randint(low=0, high=samples.shape[0])
            for i in range(n_comp):
                theta_draw = samples[sample, ...][len(self.props) * i: len(self.props) * i + len(self.props)]

                if self.fit_type == 'lte':

                    fit_lines[:, draw, i] = hyperfine_structure_lte(*theta_draw, self.strength_lines, self.v_lines,
                                                                    vel=vel)

                elif self.fit_type == 'radex':

                    fit_lines[:, draw, i] = get_hyperfine_multiple_components(theta_draw, vel, self.v_lines, qn_ul)

        return fit_lines

    def create_fit_cube(self, sampler_filename='sampler', n_comp_filename='n_comp', cube_filename='fit_cube',
                        chunksize=256):
        """Create upper/lower errors for spectral fit plots

        Returns:

        """

        ij_list = [(i, j) for i in range(self.data.shape[1]) for j in range(self.data.shape[2])
                   if self.mask[i, j] != 0]

        n_comp = np.load(n_comp_filename + '.npy')

        # Setup fit cube

        fit_cube = np.zeros([3, *self.data.shape], dtype=np.float32)

        with mp.Pool(self.n_cores) as pool:
            map_result = list(tqdm(pool.imap(partial(self.parallel_fit_samples,
                                                     sampler_filename=sampler_filename,
                                                     n_comp=n_comp),
                                             ij_list, chunksize=chunksize), total=len(ij_list)))

        for idx, ij in enumerate(ij_list):
            fit_cube[:, :, ij[0], ij[1]] = map_result[idx]

        np.save(cube_filename + '.npy', fit_cube)

    def parallel_fit_samples(self, ij, sampler_filename=None, n_comp=None):
        """TODO: Docstring"""

        if not sampler_filename or n_comp is None:
            self.logger.warning('Sampler filename and n_comp must be defined!')
            sys.exit()

        i = ij[0]
        j = ij[1]

        n_comp_pix = int(n_comp[i, j])

        cube_sampler_filename = sampler_filename + '_%s_%s.pkl' % (i, j)

        if n_comp_pix == 0:
            return np.zeros([3, len(self.vel)])
        sampler = load_fit_dict(cube_sampler_filename)
        flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)

        fit_lines = self.get_fits_from_samples(flat_samples, self.vel, n_draws=100, n_comp=n_comp_pix)

        fit_percentiles = np.nanpercentile(np.nansum(fit_lines, axis=-1), [50, 16, 84], axis=1)

        return fit_percentiles

    def make_parameter_maps(self, n_comp_filename='n_comp', sampler_filename='sampler', chunksize=256, n_samples=500,
                            output_file=None, overwrite=False):

        if self.data_type != 'cube':
            self.logger.warning('Can only make parameter maps for fitted cubes')
            sys.exit()

        n_comp = np.load(n_comp_filename + '.npy')
        max_n_comp = int(np.nanmax(n_comp))

        if not os.path.exists(output_file) or overwrite:

            # Set up arrays in a dictionary

            parameter_maps = {}

            parameter_maps['chisq_red'] = np.zeros([self.data.shape[1],
                                                    self.data.shape[2]])
            parameter_maps['chisq_red'][parameter_maps['chisq_red'] == 0] = np.nan

            for i in range(max_n_comp):

                keys = ['tpeak_%s' % i, 'tpeak_%s_err_up' % i, 'tpeak_%s_err_down' % i]
                for key in keys:
                    parameter_maps[key] = np.zeros([self.data.shape[1],
                                                    self.data.shape[2]])
                    parameter_maps[key][parameter_maps[key] == 0] = np.nan

                for prop in self.props:

                    keys = [prop + '_%s' % i, prop + '_%s_err_up' % i, prop + '_%s_err_down' % i]
                    for key in keys:
                        parameter_maps[key] = np.zeros([self.data.shape[1],
                                                        self.data.shape[2]])
                        parameter_maps[key][parameter_maps[key] == 0] = np.nan

            # Loop over each pixel, pulling out the properties for each component, as well as the peak intensity and
            # errors for everything. Parallelise this up for speed

            ij_list = [[i, j] for i in range(self.data.shape[1]) for j in range(self.data.shape[2])
                       if self.mask[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                par_dicts = list(tqdm(pool.imap(partial(self.parallel_map_making,
                                                        n_comp=n_comp,
                                                        sampler_filename=sampler_filename,
                                                        n_samples=n_samples),
                                                ij_list, chunksize=chunksize), total=len(ij_list)))

            # Pull the parameters into the arrays

            for dict_idx, par_dict in enumerate(par_dicts):

                i, j = ij_list[dict_idx][0], ij_list[dict_idx][1]
                for key in par_dict.keys():
                    parameter_maps[key][i, j] = par_dict[key]

            # Save out

            if output_file is not None:
                with open(output_file, 'wb') as f:
                    pickle.dump(parameter_maps, f)

        else:
            with open(output_file, 'rb') as f:
                parameter_maps = pickle.load(f)

        self.parameter_maps = parameter_maps
        self.max_n_comp = max_n_comp

    def parallel_map_making(self, ij, n_comp=None, sampler_filename='sampler', n_samples=500):

        if n_comp is None:
            self.logger.warning('n_comp should be defined!')
            sys.exit()

        i, j = ij[0], ij[1]
        n_comps_pix = int(n_comp[i, j])

        par_dict = {}

        obs = self.data[:, i, j]
        obs_err = self.error[:, i, j]

        if n_comps_pix > 0:

            cube_sampler_filename = sampler_filename + '_%s_%s.pkl' % (i, j)
            sampler = load_fit_dict(cube_sampler_filename)

            # Pull out median and errors for each parameter and each component
            flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
            param_percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
            param_diffs = np.diff(param_percentiles, axis=0)

            # Pull out model for reduced chi-square
            total_model = multiple_components(param_percentiles[1, :], self.vel, self.strength_lines, self.v_lines,
                                              self.props, n_comps_pix, self.fit_type)

            for n_comp_pix in range(n_comps_pix):

                # Pull out peak intensities and errors for each component

                tpeak = np.zeros(n_samples)
                for sample in range(n_samples):
                    choice_idx = np.random.randint(0, flat_samples.shape[0])
                    theta = flat_samples[choice_idx, 4 * n_comp_pix: 4 * n_comp_pix + 4]
                    model = hyperfine_structure_lte(*theta, self.strength_lines, self.v_lines, self.vel)

                    tpeak[sample] = np.nanmax(model)

                tpeak_percentiles = np.percentile(tpeak, [16, 50, 84], axis=0)
                tpeak_diff = np.diff(tpeak_percentiles)

                par_dict['tpeak_%s' % n_comp_pix] = tpeak_percentiles[1]
                par_dict['tpeak_%s_err_down' % n_comp_pix] = tpeak_diff[0]
                par_dict['tpeak_%s_err_up' % n_comp_pix] = tpeak_diff[1]

                # Pull out fitted properties and errors for each component

                for prop_idx, prop in enumerate(self.props):
                    param_idx = n_comp_pix * len(self.props) + prop_idx
                    par_dict[prop + '_%s' % n_comp_pix] = param_percentiles[1, param_idx]
                    par_dict[prop + '_%s_err_down' % n_comp_pix] = param_diffs[0, param_idx]
                    par_dict[prop + '_%s_err_up' % n_comp_pix] = param_diffs[1, param_idx]

        else:
            total_model = np.zeros_like(obs)

        chisq = chi_square(obs, total_model, obs_err)
        deg_freedom = len(obs[~np.isnan(obs)]) - (n_comps_pix * len(self.props))
        par_dict['chisq_red'] = chisq / deg_freedom

        return par_dict
