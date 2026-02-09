import logging
import multiprocessing as mp
import sys
import warnings
from functools import partial
from itertools import cycle

import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from .emcee_funcs import get_burn_in_thin, get_samples_from_fit_dict
from .fitting import HyperfineFitter, chi_square
from .fitting import get_fits_from_samples
from .utils import load_pkl, get_dict_val

ALLOWED_FILE_EXTS = [
    "png",
    "pdf",
    "ps",
    "eps",
    "svg",
]

logger = logging.getLogger("mcfine")

# Define global variables for potentially huge arrays, and various config values
glob_data = np.array([])
glob_error = np.array([])
glob_vel = np.array([])

glob_config = {}

glob_mcfine_output = {}


def parallel_step(
    ij,
    plot_name="step",
    fit_dict_filename="fit_dict",
    consolidate_fit_dict=True,
):
    """Wrapper to parallelise step plotting"""

    i, j = ij[0], ij[1]

    if consolidate_fit_dict:
        fit_dict = glob_mcfine_output["fit"].get(i, {}).get(j, {})
    else:
        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        fit_dict = load_pkl(cube_fit_dict_filename)

    n_comp = fit_dict["n_comp"]

    if n_comp == 0:
        return True

    if "sampler" not in fit_dict:
        logger.warning("Can only produce step plots when emcee sampler is present")
        return False
    sampler = fit_dict["sampler"]

    cube_plot_name = f"{plot_name}_{i}_{j}"
    plot_step(
        sampler,
        plot_name=cube_plot_name,
        n_comp=n_comp,
    )

    return True


def plot_step(
    sampler,
    plot_name="step_plot",
    n_comp=1,
):
    """Make a step plot"""

    file_exts = get_dict_val(
        glob_config["config"],
        glob_config["config_defaults"],
        table="plotting",
        key="file_exts",
    )

    # Get samples from the chain
    samples = sampler.get_chain()

    # And get the nominal "burn-in" range to put on the
    # plot
    burn_in, _ = get_burn_in_thin(
        sampler,
        burn_in_frac=glob_config["burn_in"],
        thin_frac=glob_config["thin"],
    )

    # Load up the labels
    labels = []
    for i in range(n_comp):
        for label in glob_config["labels"]:
            labels.append(label % i)

    fig, axes = plt.subplots(
        nrows=len(glob_config["labels"]),
        ncols=n_comp,
        squeeze=False,
        figsize=(6 * n_comp, 2 * len(glob_config["labels"])),
        sharex="all",
    )
    plt.subplots_adjust(hspace=0.1)

    axes = axes.T.flatten()

    colour = cycle(iter(plt.cm.rainbow(np.linspace(0, 1, len(glob_config["labels"])))))

    ax_i = 0

    for j in range(n_comp):

        for i in range(len(glob_config["labels"])):

            c = next(colour)

            # sample_no = i * n_comp + j

            ax = axes[ax_i]
            ax.plot(samples[:, :, ax_i], c=c, alpha=0.3)
            ax.set_xlim(0, len(samples))

            ax.axvline(
                burn_in,
                color="k",
                linestyle="--",
            )

            plt.text(
                0.05,
                0.9,
                glob_config["labels"][i] % j,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="black", alpha=1),
                transform=ax.transAxes,
            )

            if i == len(glob_config["labels"]) - 1:
                ax.set_xlabel("Step Number")

            ax_i += 1

    for file_ext in file_exts:
        if file_ext not in ALLOWED_FILE_EXTS:
            logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
            sys.exit()

        plt.savefig(
            f"{plot_name}.{file_ext}",
            bbox_inches="tight",
        )

    plt.close()

    return True


def parallel_corner(
    ij,
    plot_name="corner",
    fit_dict_filename=None,
    consolidate_fit_dict=True,
):
    """Wrapper to parallelise corner plotting"""

    i, j = ij[0], ij[1]

    if consolidate_fit_dict:
        fit_dict = glob_mcfine_output["fit"].get(i, {}).get(j, {})
    else:
        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        fit_dict = load_pkl(cube_fit_dict_filename)

    n_comp = fit_dict["n_comp"]

    if n_comp == 0:
        return True

    flat_samples = get_samples_from_fit_dict(
        fit_dict,
        burn_in_frac=glob_config["burn_in"],
        thin_frac=glob_config["thin"],
    )

    cube_plot_name = f"{plot_name}_{i}_{j}"
    plot_corner(
        flat_samples,
        plot_name=cube_plot_name,
        n_comp=n_comp,
    )

    return True


def plot_corner(
    flat_samples,
    plot_name,
    n_comp=1,
):
    """Make a corner plot"""

    file_exts = get_dict_val(
        glob_config["config"],
        glob_config["config_defaults"],
        table="plotting",
        key="file_exts",
    )

    # Load up the labels
    labels = []
    for i in range(n_comp):
        for label in glob_config["labels"]:
            labels.append(label % i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corner.corner(
            flat_samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
        )

    for file_ext in file_exts:
        if file_ext not in ALLOWED_FILE_EXTS:
            logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
            sys.exit()

        plt.savefig(
            f"{plot_name}.{file_ext}",
            bbox_inches="tight",
        )

    plt.close()

    return True


def parallel_plot_fit(
    ij,
    fit_dict_filename=None,
    consolidate_fit_dict=True,
    loc_image=None,
    n_comp=None,
    plot_name="fit",
    show_individual_components=True,
    show_hyperfine_components=False,
    n_points=1000,
    n_draws=100,
    figsize=(10, 4),
    x_label=r"Velocity (km s$^{-1}$)",
    y_label="Intensity (K)",
):
    """Wrapper to parallelise fit spectrum plotting"""

    i, j = ij[0], ij[1]

    if consolidate_fit_dict:
        fit_dict = glob_mcfine_output["fit"].get(i, {}).get(j, {})
    else:
        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        fit_dict = load_pkl(cube_fit_dict_filename)

    n_comp = fit_dict["n_comp"]

    data = glob_data[:, i, j]
    error = glob_error[:, i, j]
    if n_comp > 0:
        flat_samples = get_samples_from_fit_dict(
            fit_dict,
            burn_in_frac=glob_config["burn_in"],
            thin_frac=glob_config["thin"],
        )
    else:
        flat_samples = None

    cube_plot_name = f"{plot_name}_{i}_{j}"

    # Figure out the optimum figure size
    if loc_image is not None:
        ratio = loc_image.shape[0] / loc_image.shape[1]
        figsize = (3 * 5, 5 * ratio)

    else:
        figsize = figsize

    plot_fit(
        flat_samples=flat_samples,
        data=data,
        error=error,
        loc_image=loc_image,
        i=i,
        j=j,
        n_comp=n_comp,
        plot_name=cube_plot_name,
        show_individual_components=show_individual_components,
        show_hyperfine_components=show_hyperfine_components,
        n_points=n_points,
        n_draws=n_draws,
        figsize=figsize,
        x_label=x_label,
        y_label=y_label,
    )


def plot_fit(
    flat_samples,
    data,
    error,
    loc_image=None,
    i=None,
    j=None,
    n_comp=1,
    plot_name="fit",
    show_individual_components=True,
    show_hyperfine_components=False,
    n_points=1000,
    n_draws=100,
    figsize=(10, 4),
    x_label=r"Velocity (km s$^{-1}$)",
    y_label="Intensity (K)",
):
    """Plot a fit spectrum"""

    file_exts = get_dict_val(
        glob_config["config"],
        glob_config["config_defaults"],
        table="plotting",
        key="file_exts",
    )

    vel_min, vel_max = np.nanmin(glob_vel), np.nanmax(glob_vel)

    vel_plot_mcmc = np.linspace(vel_min, vel_max, n_points)

    fit_percentiles_components = None
    if flat_samples is not None:
        fit_mcmc = get_fits_from_samples(
            samples=flat_samples,
            vel=vel_plot_mcmc,
            props=glob_config["props"],
            strength_lines=glob_config["strength_lines"],
            v_lines=glob_config["v_lines"],
            fit_type=glob_config["fit_type"],
            n_draws=n_draws,
            n_comp=n_comp,
        )
        fit_chisq = get_fits_from_samples(
            samples=flat_samples,
            vel=glob_vel,
            props=glob_config["props"],
            strength_lines=glob_config["strength_lines"],
            v_lines=glob_config["v_lines"],
            fit_type=glob_config["fit_type"],
            n_draws=n_draws,
            n_comp=n_comp,
        )
        model = np.nanmedian(np.nansum(fit_chisq, axis=-1), axis=1)

        if show_individual_components:
            fit_percentiles_components = np.nanpercentile(
                fit_mcmc, [50, 16, 84], axis=1
            )
        fit_percentiles = np.nanpercentile(
            np.nansum(fit_mcmc, axis=-1), [50, 16, 84], axis=1
        )
    else:
        fit_percentiles = np.zeros([3, n_points])
        model = np.zeros_like(glob_vel)

    # Calculate reduced chisq
    chisq = chi_square(data, model, observed_error=error)
    deg_freedom = len(data[~np.isnan(data)]) - (n_comp * len(glob_config["props"]))
    chisq_red = chisq / deg_freedom

    fig = plt.figure(figsize=figsize)

    if loc_image is not None:
        gs = GridSpec(1, 3, figure=fig)
        ax = fig.add_subplot(gs[0, :-1])
    else:
        gs = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])

    plt.step(
        glob_vel,
        data,
        where="mid",
        c="k",
    )

    y_lim = plt.ylim()
    y_min, y_max = y_lim[0], 1.2 * np.nanmax(data)

    plt.plot(vel_plot_mcmc, fit_percentiles[0, :], c="r", zorder=99)
    if flat_samples is not None:
        plt.fill_between(
            vel_plot_mcmc,
            fit_percentiles[1, :],
            fit_percentiles[2, :],
            color="r",
            alpha=0.75,
            zorder=99,
        )

        if show_individual_components and fit_percentiles_components is not None:
            plt.plot(
                vel_plot_mcmc,
                fit_percentiles_components[0, :, :],
                c="k",
            )
            for i_fit_percentiles in range(fit_percentiles_components.shape[-1]):
                plt.fill_between(
                    vel_plot_mcmc,
                    fit_percentiles_components[1, :, i_fit_percentiles],
                    fit_percentiles_components[2, :, i_fit_percentiles],
                    color="k",
                    alpha=0.75,
                )

    plt.xlim([vel_min, vel_max])
    plt.ylim([y_min, y_max])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    plt.grid()

    plt.text(
        0.95,
        0.95,
        rf"$\chi_\nu^2={chisq_red:.2f}$",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", alpha=1),
        transform=ax.transAxes,
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if loc_image is not None:
        ax = fig.add_subplot(gs[0, -1])

        vmin, vmax = np.nanpercentile(loc_image, [1, 99])

        ax.imshow(
            loc_image,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        )

        # Put a cross on the image to show where the pixel is
        if i is not None and j is not None:
            plt.scatter(
                j,
                i,
                c="lime",
                marker="x",
            )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        plt.subplots_adjust(hspace=0, wspace=0)

    else:
        plt.tight_layout()

    for file_ext in file_exts:
        if file_ext not in ALLOWED_FILE_EXTS:
            logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
            sys.exit()

        plt.savefig(
            f"{plot_name}.{file_ext}",
            bbox_inches="tight",
        )

    plt.close()


class HyperfinePlotter(HyperfineFitter):

    def __init__(
        self,
        data=None,
        error=None,
        mask=None,
        vel=None,
        config_file=None,
        local_file=None,
    ):
        """Plotting support for McFine

        Allows for plotting step/corner/fit spectrum plots, in a publication-ready format

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
        """

        HyperfineFitter.__init__(**locals())

        # Define global variables for potentially huge arrays
        global glob_data, glob_error, glob_vel
        glob_data = self.data
        glob_error = self.error
        glob_vel = self.vel

        # Define a global configuration dictionary that we'll use in multiprocessing
        global glob_config

        keys_to_glob = [
            "config_defaults",
            "config",
            "props",
            "labels",
            "strength_lines",
            "v_lines",
            "fit_type",
            "burn_in",
            "thin",
        ]
        for k in keys_to_glob:
            glob_config[k] = self.__dict__[k]

        # Keep track if we've already loaded in the big
        # mcfine output
        self.mcfine_output_loaded = False

    def plot_step(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        plot_name=None,
        grid=None,
    ):
        """Create step plot for either a spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            mcfine_output_filename (str): Name for the mcfine output.
                Defaults to None, which will pull from config.toml
            plot_name (str): Output plot name. Defaults to None, which will pull from
                config.toml
            grid (np.ndarray): If fitting a cube, can pass a ndarray of 1 (plot) and 0
                (do not plot) to avoid making loads of plots. Defaults to None, which
                will then use the fitting mask and plot everything
        """

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
            )

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if plot_name is None:
            plot_name = get_dict_val(
                self.config,
                self.config_defaults,
                table="step_plot",
                key="plot_name",
            )

        if self.data_type == "spectrum":
            fit_dict = load_pkl(f"{fit_dict_filename}.pkl")

            n_comp = fit_dict["n_comp"]

            if "sampler" not in fit_dict:
                self.logger.warning(
                    "Can only produce step plots when emcee sampler is present"
                )
                return False
            sampler = fit_dict["sampler"]

            if n_comp == 0:
                return True

            plot_step(
                sampler,
                plot_name,
                n_comp=n_comp,
            )

        elif self.data_type == "cube":

            # Load in the mcfine output as a global variable, if we haven't already
            if not self.mcfine_output_loaded:
                global glob_mcfine_output
                glob_mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")
                self.mcfine_output_loaded = True

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

            if grid is None:
                grid = self.mask

            ij_list = [
                (i, j)
                for i in range(grid.shape[0])
                for j in range(grid.shape[1])
                if grid[i, j] != 0
            ]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_step,
                                plot_name=plot_name,
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

        return True

    def plot_corner(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        plot_name=None,
        grid=None,
    ):
        """Create corner plot for either a spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            mcfine_output_filename (str): Name for the mcfine output.
                Defaults to None, which will pull from config.toml
            plot_name (str): Output plot name. Defaults to None, which will pull from
                config.toml
            grid (np.ndarray): If fitting a cube, can pass a ndarray of 1 (plot) and 0
                (do not plot) to avoid making loads of plots. Defaults to None, which
                will then use the fitting mask and plot everything
        """

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
            )

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if plot_name is None:
            plot_name = get_dict_val(
                self.config,
                self.config_defaults,
                table="plot_corner",
                key="plot_name",
            )

        if self.data_type == "spectrum":
            fit_dict = load_pkl(f"{fit_dict_filename}.pkl")

            n_comp = fit_dict["n_comp"]

            if n_comp == 0:
                return

            flat_samples = get_samples_from_fit_dict(
                fit_dict,
                burn_in_frac=self.burn_in,
                thin_frac=self.thin,
            )

            plot_corner(
                flat_samples=flat_samples,
                plot_name=plot_name,
                n_comp=n_comp,
            )

        elif self.data_type == "cube":

            # Load in the mcfine output as a global variable, if we haven't already
            if not self.mcfine_output_loaded:
                global glob_mcfine_output
                glob_mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")
                self.mcfine_output_loaded = True

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

            if grid is None:
                grid = self.mask

            ij_list = [
                (i, j)
                for i in range(grid.shape[0])
                for j in range(grid.shape[1])
                if grid[i, j] != 0
            ]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_corner,
                                plot_name=plot_name,
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

    def plot_fit(
        self,
        fit_dict_filename=None,
        mcfine_output_filename=None,
        plot_name=None,
        grid=None,
        loc_image=None,
    ):
        """Create fit spectrum plot for either a single spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            mcfine_output_filename (str): Name for the mcfine output.
                Defaults to None, which will pull from config.toml
            plot_name (str): Output plot name. Defaults to None, which will pull from
                config.toml
            grid (np.ndarray): If fitting a cube, can pass a ndarray of 1 (plot) and 0
                (do not plot) to avoid making loads of plots. Defaults to None, which
                will then use the fitting mask and plot everything
            loc_image (np.ndarray): If fitting a cube, we can pass an image to here
                so the fit will also have a subplot showing the location of the
                pixel in question. Defaults to None, which will not plot this
        """

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="fit_dict_filename",
            )

        if mcfine_output_filename is None:
            mcfine_output_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="mcfine_output_filename",
                logger=self.logger,
            )

        if plot_name is None:
            plot_name = get_dict_val(
                self.config,
                self.config_defaults,
                table="plot_fit",
                key="plot_name",
            )

        show_individual_components = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="show_individual_components",
        )

        show_hyperfine_components = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="show_hyperfine_components",
        )

        n_points = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="n_points",
        )

        n_draws = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="n_draws",
        )

        figsize = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="figsize",
        )

        x_label = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="x_label",
        )

        y_label = get_dict_val(
            self.config,
            self.config_defaults,
            table="plot_fit",
            key="y_label",
        )

        if self.data_type == "spectrum":
            fit_dict = load_pkl(f"{fit_dict_filename}.pkl")
            n_comp = fit_dict["n_comp"]

            if n_comp == 0:
                flat_samples = None
            else:
                flat_samples = get_samples_from_fit_dict(
                    fit_dict,
                    burn_in_frac=self.burn_in,
                    thin_frac=self.thin,
                )

            plot_fit(
                flat_samples=flat_samples,
                data=self.data,
                error=self.error,
                n_comp=n_comp,
                plot_name=plot_name,
                show_individual_components=show_individual_components,
                show_hyperfine_components=show_hyperfine_components,
                n_points=n_points,
                n_draws=n_draws,
                figsize=figsize,
                x_label=x_label,
                y_label=y_label,
            )

        elif self.data_type == "cube":

            # Load in the mcfine output as a global variable, if we haven't already
            if not self.mcfine_output_loaded:
                global glob_mcfine_output
                glob_mcfine_output = load_pkl(f"{mcfine_output_filename}.pkl")
                self.mcfine_output_loaded = True

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

            if grid is None:
                grid = self.mask

            ij_list = [
                (i, j)
                for i in range(grid.shape[0])
                for j in range(grid.shape[1])
                if grid[i, j] != 0
            ]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(
                                parallel_plot_fit,
                                fit_dict_filename=fit_dict_filename,
                                consolidate_fit_dict=self.consolidate_fit_dict,
                                plot_name=plot_name,
                                loc_image=loc_image,
                                show_individual_components=show_individual_components,
                                show_hyperfine_components=show_hyperfine_components,
                                n_points=n_points,
                                n_draws=n_draws,
                                figsize=figsize,
                                x_label=x_label,
                                y_label=y_label,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                        dynamic_ncols=True,
                    )
                )
