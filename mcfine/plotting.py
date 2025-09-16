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

from .fitting import HyperfineFitter, chi_square
from .utils import load_fit_dict, get_dict_val

ALLOWED_FILE_EXTS = [
    "png",
    "pdf",
    "ps",
    "eps",
    "svg",
]


class HyperfinePlotter(HyperfineFitter):

    def __init__(
        self,
        data=None,
        vel=None,
        error=None,
        mask=None,
        config_file=None,
        local_file=None,
    ):
        """Plotting support for McFine

        Allows for plotting step/corner/fit spectrum plots, in a publication-ready format

        Args:
            data (np.ndarray): Either a 1D array of intensity (spectrum) or a 3D array of intensities (cube).
                Intensities should be in K.
            vel (np.ndarray): Array of velocity values that correspond to data, in km/s.
            error (np.ndarray): Array of errors in intensity. Should have the same shape as `data`. Defaults to None.
            mask (np.ndarray): 1/0 mask to indicate significant emission in the cube (i.e. the pixels to fit). Should
                have shape of `data.shape[1:]`. Defaults to None, which will fit all pixels in a cube.
            config_file (str): Path to config.toml file. Defaults to None, which will use the default settings
            local_file (str): Path to local.toml file. Defaults to None, which will use the default settings
        """

        HyperfineFitter.__init__(**locals())

    def plot_step(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        plot_name=None,
        grid=None,
    ):
        """Create step plot for either a spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            n_comp_filename (str): Name for the file containing the n_comp map. Only
                used for cubes. Defaults to None, which will pull from config.toml
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

        if plot_name is None:
            plot_name = get_dict_val(
                self.config,
                self.config_defaults,
                table="step_plot",
                key="plot_name",
            )

        if self.data_type == "spectrum":
            fit_dict = load_fit_dict(f"{fit_dict_filename}.pkl")

            n_comp = fit_dict["n_comp"]

            if "sampler" not in fit_dict:
                self.logger.warning(
                    "Can only produce step plots when emcee sampler is present"
                )
                return False
            sampler = fit_dict["sampler"]

            if n_comp == 0:
                return
            samples = sampler.get_chain()
            self.step(
                samples,
                plot_name,
                n_comp=n_comp,
            )

        elif self.data_type == "cube":

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="n_comp_filename",
                )

            n_comp = np.load(f"{n_comp_filename}.npy")
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
                                self.parallel_step,
                                plot_name=plot_name,
                                fit_dict_filename=fit_dict_filename,
                                n_comp=n_comp,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                    )
                )

    def parallel_step(
        self,
        ij,
        plot_name="step",
        fit_dict_filename="fit_dict",
        n_comp=None,
    ):
        """Wrapper to parallelise step plotting"""

        if n_comp is None:
            raise Warning("n_comp should be defined!")

        i, j = ij[0], ij[1]

        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"

        n_comp_pix = int(n_comp[i, j])
        if n_comp_pix == 0:
            return
        fit_dict = load_fit_dict(cube_fit_dict_filename)

        if "sampler" not in fit_dict:
            self.logger.warning(
                "Can only produce step plots when emcee sampler is present"
            )
            return False
        sampler = fit_dict["sampler"]

        samples = sampler.get_chain()
        cube_plot_name = f"{plot_name}_{i}_{j}"
        self.step(
            samples,
            plot_name=cube_plot_name,
            n_comp=n_comp_pix,
        )

    def step(
        self,
        samples,
        plot_name="step_plot",
        n_comp=1,
    ):
        """Make a step plot"""

        file_exts = get_dict_val(
            self.config,
            self.config_defaults,
            table="plotting",
            key="file_exts",
        )

        # Load up the labels
        labels = []
        for i in range(n_comp):
            for label in self.labels:
                labels.append(label % i)

        fig, axes = plt.subplots(
            nrows=len(self.labels),
            ncols=n_comp,
            squeeze=False,
            figsize=(6 * n_comp, 2 * len(self.labels)),
            sharex="all",
        )
        plt.subplots_adjust(hspace=0.1)

        axes = axes.T.flatten()

        colour = cycle(iter(plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))))

        ax_i = 0

        for j in range(n_comp):

            for i in range(len(self.labels)):

                c = next(colour)

                # sample_no = i * n_comp + j

                ax = axes[ax_i]
                ax.plot(samples[:, :, ax_i], c=c, alpha=0.3)
                ax.set_xlim(0, len(samples))

                plt.text(
                    0.05,
                    0.9,
                    self.labels[i] % j,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="white", edgecolor="black", alpha=1),
                    transform=ax.transAxes,
                )

                if i == len(self.labels) - 1:
                    ax.set_xlabel("Step Number")

                ax_i += 1

        for file_ext in file_exts:
            if file_ext not in ALLOWED_FILE_EXTS:
                self.logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
                sys.exit()

            plt.savefig(
                f"{plot_name}.{file_ext}",
                bbox_inches="tight",
            )

        plt.close()

    def plot_corner(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        plot_name=None,
        grid=None,
    ):
        """Create corner plot for either a spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            n_comp_filename (str): Name for the file containing the n_comp map. Only
                used for cubes. Defaults to None, which will pull from config.toml
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

        if plot_name is None:
            plot_name = get_dict_val(
                self.config,
                self.config_defaults,
                table="plot_corner",
                key="plot_name",
            )

        if self.data_type == "spectrum":
            fit_dict = load_fit_dict(f"{fit_dict_filename}.pkl")

            n_comp = fit_dict["n_comp"]

            if n_comp == 0:
                return

            # If we have the full emcee sampler, prefer that
            if "sampler" in fit_dict:

                sampler = fit_dict["sampler"]
                flat_samples = self.get_samples(sampler)

            # Otherwise, pull from the covariance matrix
            else:

                cov_matrix = fit_dict["cov"]["matrix"]
                cov_med = fit_dict["cov"]["med"]
                flat_samples = np.random.multivariate_normal(
                    cov_med, cov_matrix, size=10000
                )

            self.corner(
                flat_samples=flat_samples,
                plot_name=plot_name,
                n_comp=n_comp,
            )

        elif self.data_type == "cube":

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(
                    self.config,
                    self.config_defaults,
                    table="multicomponent_fitter",
                    key="n_comp_filename",
                )

            n_comp = np.load(f"{n_comp_filename}.npy")
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
                                self.parallel_corner,
                                plot_name=plot_name,
                                fit_dict_filename=fit_dict_filename,
                                n_comp=n_comp,
                            ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list),
                    )
                )

    def parallel_corner(
        self,
        ij,
        plot_name="corner",
        fit_dict_filename=None,
        n_comp=None,
    ):
        """Wrapper to parallelise corner plotting"""

        if fit_dict_filename is None:
            self.logger.warning("sampler_filename should be defined!")
            sys.exit()
        if n_comp is None:
            self.logger.warning("n_comp should be defined!")
            sys.exit()

        i, j = ij[0], ij[1]
        n_comp_pix = int(n_comp[i, j])

        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        if n_comp_pix == 0:
            return
        fit_dict = load_fit_dict(cube_fit_dict_filename)

        # If we have the full emcee sampler, prefer that here
        if "sampler" in fit_dict:

            sampler = fit_dict["sampler"]
            flat_samples = self.get_samples(sampler)

        # Otherwise, pull from the covariance matrix
        else:

            cov_matrix = fit_dict["cov"]["matrix"]
            cov_med = fit_dict["cov"]["med"]
            flat_samples = np.random.multivariate_normal(
                cov_med, cov_matrix, size=10000
            )

        cube_plot_name = f"{plot_name}_{i}_{j}"
        self.corner(flat_samples, plot_name=cube_plot_name, n_comp=n_comp_pix)

    def corner(
        self,
        flat_samples,
        plot_name,
        n_comp=1,
    ):
        """Make a corner plot"""

        file_exts = get_dict_val(
            self.config,
            self.config_defaults,
            table="plotting",
            key="file_exts",
        )

        # Load up the labels
        labels = []
        for i in range(n_comp):
            for label in self.labels:
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
                self.logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
                sys.exit()

            plt.savefig(
                f"{plot_name}.{file_ext}",
                bbox_inches="tight",
            )

        plt.close()

        return True

    def plot_fit(
        self,
        fit_dict_filename=None,
        n_comp_filename=None,
        plot_name=None,
        grid=None,
        loc_image=None,
    ):
        """Create fit spectrum plot for either a single spectrum or a (subset of) a cube

        Args:
            fit_dict_filename (str): Name for the file containing the fit parameters.
                Defaults to None, which will pull from config.toml
            n_comp_filename (str): Name for the file containing the n_comp map. Only
                used for cubes. Defaults to None, which will pull from config.toml
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
        if n_comp_filename is None:
            n_comp_filename = get_dict_val(
                self.config,
                self.config_defaults,
                table="multicomponent_fitter",
                key="n_comp_filename",
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
            fit_dict = load_fit_dict(f"{fit_dict_filename}.pkl")
            n_comp = fit_dict["n_comp"]

            if n_comp == 0:
                flat_samples = None
            else:

                # If we have the sampler, prefer that
                if "sampler" in fit_dict:

                    sampler = fit_dict["sampler"]
                    flat_samples = self.get_samples(sampler)

                else:

                    cov_matrix = fit_dict["cov"]["matrix"]
                    cov_med = fit_dict["cov"]["med"]
                    flat_samples = np.random.multivariate_normal(
                        cov_med, cov_matrix, size=10000
                    )

            self.fit(
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

            n_comp = np.load(f"{n_comp_filename}.npy")
            if grid is None:
                grid = self.mask

            chunksize = get_dict_val(
                self.config,
                self.config_defaults,
                table="plotting",
                key="chunksize",
            )

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
                                self.parallel_plot_fit,
                                fit_dict_filename=fit_dict_filename,
                                plot_name=plot_name,
                                loc_image=loc_image,
                                n_comp=n_comp,
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
                    )
                )

    def parallel_plot_fit(
        self,
        ij,
        fit_dict_filename=None,
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

        if fit_dict_filename is None:
            self.logger.warning("fit_dict_filename should be defined!")
            sys.exit()

        if n_comp is None:
            self.logger.warning("n_comp should be defined!")
            sys.exit()

        i, j = ij[0], ij[1]

        cube_fit_dict_filename = f"{fit_dict_filename}_{i}_{j}.pkl"
        n_comp_pix = int(n_comp[i, j])
        data = self.data[:, i, j]
        error = self.error[:, i, j]
        if n_comp_pix > 0:
            fit_dict = load_fit_dict(cube_fit_dict_filename)

            # If we have the full emcee sampler, prefer that here
            if "sampler" in fit_dict:
                sampler = fit_dict["sampler"]
                flat_samples = self.get_samples(sampler)

            # Otherwise fall back to the covariance matrix
            else:

                cov_matrix = fit_dict["cov"]["matrix"]
                cov_med = fit_dict["cov"]["med"]
                flat_samples = np.random.multivariate_normal(
                    cov_med, cov_matrix, size=10000
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

        self.fit(
            flat_samples=flat_samples,
            data=data,
            error=error,
            loc_image=loc_image,
            i=i,
            j=j,
            n_comp=n_comp_pix,
            plot_name=cube_plot_name,
            show_individual_components=show_individual_components,
            show_hyperfine_components=show_hyperfine_components,
            n_points=n_points,
            n_draws=n_draws,
            figsize=figsize,
            x_label=x_label,
            y_label=y_label,
        )

    def fit(
        self,
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
            self.config,
            self.config_defaults,
            table="plotting",
            key="file_exts",
        )

        vel_min, vel_max = np.nanmin(self.vel), np.nanmax(self.vel)

        vel_plot_mcmc = np.linspace(vel_min, vel_max, n_points)

        fit_percentiles_components = None
        if flat_samples is not None:
            fit_mcmc = super(HyperfinePlotter, self).get_fits_from_samples(
                samples=flat_samples,
                vel=vel_plot_mcmc,
                n_draws=n_draws,
                n_comp=n_comp,
            )
            fit_chisq = super(HyperfinePlotter, self).get_fits_from_samples(
                samples=flat_samples,
                vel=self.vel,
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
            model = np.zeros_like(self.vel)

        # Calculate reduced chisq
        chisq = chi_square(data, model, observed_error=error)
        deg_freedom = len(data[~np.isnan(data)]) - (n_comp * len(self.props))
        chisq_red = chisq / deg_freedom

        fig = plt.figure(figsize=figsize)

        if loc_image is not None:
            gs = GridSpec(1, 3, figure=fig)
            ax = fig.add_subplot(gs[0, :-1])
        else:
            gs = GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0])

        plt.step(
            self.vel,
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
                self.logger.warning(f"file_ext should be one of {ALLOWED_FILE_EXTS}")
                sys.exit()

            plt.savefig(
                f"{plot_name}.{file_ext}",
                bbox_inches="tight",
            )

        plt.close()
