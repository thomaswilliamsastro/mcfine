import multiprocessing as mp
import sys
import warnings
from functools import partial

import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from tqdm import tqdm

from .fitting import HyperfineFitter, chi_square
from .utils import load_fit_dict, get_dict_val

ALLOWED_FILE_EXTS = [
    'png',
    'pdf',
    'ps',
    'eps',
    'svg',
]


class HyperfinePlotter(HyperfineFitter):

    def __init__(self,
                 data=None,
                 vel=None,
                 error=None,
                 mask=None,
                 config_file=None,
                 local_file=None,
                 ):
        """TODO: Docstring"""

        HyperfineFitter.__init__(**locals())

    def plot_step(self,
                  fit_dict_filename=None,
                  n_comp_filename=None,
                  plot_name=None,
                  grid=None,
                  ):
        """TODO"""

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(self.config,
                                             self.config_defaults,
                                             table='multicomponent_fitter',
                                             key='fit_dict_filename',
                                             )

        if plot_name is None:
            plot_name = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='step_plot',
                                     key='plot_name',
                                     )

        if self.data_type == 'spectrum':
            fit_dict = load_fit_dict(fit_dict_filename + '.pkl')

            n_comp = fit_dict['n_comp']
            sampler = fit_dict['sampler']

            if n_comp == 0:
                return
            samples = sampler.get_chain()
            self.step(samples,
                      plot_name,
                      n_comp=n_comp,
                      )

        elif self.data_type == 'cube':

            chunksize = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='plotting',
                                     key='chunksize',
                                     )

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(self.config,
                                               self.config_defaults,
                                               table='multicomponent_fitter',
                                               key='n_comp_filename',
                                               )

            n_comp = np.load(n_comp_filename + '.npy')
            if grid is None:
                grid = self.mask

            ij_list = [(i, j)
                       for i in range(grid.shape[0])
                       for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(self.parallel_step,
                                    plot_name=plot_name,
                                    fit_dict_filename=fit_dict_filename,
                                    n_comp=n_comp,
                                    ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list)
                    )
                )

    def parallel_step(self,
                      ij,
                      plot_name='step',
                      fit_dict_filename='fit_dict',
                      n_comp=None,
                      ):
        """TODO"""

        if n_comp is None:
            raise Warning('n_comp should be defined!')

        i, j = ij[0], ij[1]

        cube_fit_dict_filename = fit_dict_filename + '_%s_%s.pkl' % (i, j)

        n_comp_pix = int(n_comp[i, j])
        if n_comp_pix == 0:
            return
        fit_dict = load_fit_dict(cube_fit_dict_filename)

        sampler = fit_dict['sampler']

        samples = sampler.get_chain()
        cube_plot_name = plot_name + '_%s_%s' % (i, j)
        self.step(samples,
                  plot_name=cube_plot_name,
                  n_comp=n_comp_pix,
                  )

    def step(self,
             samples,
             plot_name='step_plot',
             n_comp=1,
             ):
        """TODO"""

        file_exts = get_dict_val(self.config,
                                 self.config_defaults,
                                 table='plotting',
                                 key='file_exts',
                                 )

        n_dims = n_comp * len(self.labels)

        # Load up the labels
        labels = []
        for i in range(n_comp):
            for label in self.labels:
                labels.append(label % i)

        fig, axes = plt.subplots(n_dims, figsize=(8, 6 * n_comp), sharex='all')
        for i in range(n_dims):

            if n_dims == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(samples[:, :, i], 'k', alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel('Step Number')

        for file_ext in file_exts:
            if file_ext not in ALLOWED_FILE_EXTS:
                self.logger.warning('file_ext should be one of %s' % ALLOWED_FILE_EXTS)
                sys.exit()

            plt.savefig('%s.%s' % (plot_name, file_ext), bbox_inches='tight')

        plt.close()

    def plot_corner(self,
                    fit_dict_filename=None,
                    n_comp_filename=None,
                    plot_name=None,
                    grid=None,
                    ):
        """TODO"""

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(self.config,
                                             self.config_defaults,
                                             table='multicomponent_fitter',
                                             key='fit_dict_filename',
                                             )

        if plot_name is None:
            plot_name = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='plot_corner',
                                     key='plot_name',
                                     )

        if self.data_type == 'spectrum':
            fit_dict = load_fit_dict('%s.pkl' % fit_dict_filename)

            n_comp = fit_dict['n_comp']

            if n_comp == 0:
                return

            sampler = fit_dict['sampler']

            flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                             flat=True,
                                             )
            self.corner(flat_samples=flat_samples,
                        plot_name=plot_name,
                        n_comp=n_comp,
                        )

        elif self.data_type == 'cube':

            chunksize = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='plotting',
                                     key='chunksize',
                                     )

            if n_comp_filename is None:
                n_comp_filename = get_dict_val(self.config,
                                               self.config_defaults,
                                               table='multicomponent_fitter',
                                               key='n_comp_filename',
                                               )

            n_comp = np.load('%s.npy' % n_comp_filename)
            if grid is None:
                grid = self.mask

            ij_list = [(i, j)
                       for i in range(grid.shape[0])
                       for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(self.parallel_corner,
                                    plot_name=plot_name,
                                    fit_dict_filename=fit_dict_filename,
                                    n_comp=n_comp,
                                    ),
                            ij_list,
                            chunksize=chunksize,
                        ),
                        total=len(ij_list)
                    )
                )

    def parallel_corner(self,
                        ij,
                        plot_name='corner',
                        fit_dict_filename=None,
                        n_comp=None,
                        ):
        """TODO"""

        if fit_dict_filename is None:
            self.logger.warning('sampler_filename should be defined!')
            sys.exit()
        if n_comp is None:
            self.logger.warning('n_comp should be defined!')
            sys.exit()

        i, j = ij[0], ij[1]
        n_comp_pix = int(n_comp[i, j])

        cube_fit_dict_filename = '%s_%s_%s.pkl' % (fit_dict_filename, i, j)
        if n_comp_pix == 0:
            return
        fit_dict = load_fit_dict(cube_fit_dict_filename)
        sampler = fit_dict['sampler']
        flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
        cube_plot_name = '%s_%s_%s' % (plot_name, i, j)
        self.corner(flat_samples, plot_name=cube_plot_name, n_comp=n_comp_pix)

    def corner(self,
               flat_samples,
               plot_name,
               n_comp=1,
               ):
        """TODO"""

        file_exts = get_dict_val(self.config,
                                 self.config_defaults,
                                 table='plotting',
                                 key='file_exts',
                                 )

        # Load up the labels
        labels = []
        for i in range(n_comp):
            for label in self.labels:
                labels.append(label % i)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corner.corner(flat_samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])

        for file_ext in file_exts:
            if file_ext not in ALLOWED_FILE_EXTS:
                self.logger.warning('file_ext should be one of %s' % ALLOWED_FILE_EXTS)
                sys.exit()

            plt.savefig('%s.%s' % (plot_name, file_ext), bbox_inches='tight')

        plt.close()

    def plot_fit(self,
                 fit_dict_filename='fit_dict',
                 n_comp_filename='n_comp',
                 plot_name='fit',
                 grid=None,
                 ):
        """TODO"""

        if fit_dict_filename is None:
            fit_dict_filename = get_dict_val(self.config,
                                             self.config_defaults,
                                             table='multicomponent_fitter',
                                             key='fit_dict_filename',
                                             )
        if n_comp_filename is None:
            n_comp_filename = get_dict_val(self.config,
                                           self.config_defaults,
                                           table='multicomponent_fitter',
                                           key='n_comp_filename',
                                           )
        if plot_name is None:
            plot_name = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='plot_fit',
                                     key='plot_name',
                                     )

        show_individual_components = get_dict_val(self.config,
                                                  self.config_defaults,
                                                  table='plot_fit',
                                                  key='show_individual_components',
                                                  )

        show_hyperfine_components = get_dict_val(self.config,
                                                 self.config_defaults,
                                                 table='plot_fit',
                                                 key='show_hyperfine_components',
                                                 )

        n_points = get_dict_val(self.config,
                                self.config_defaults,
                                table='plot_fit',
                                key='n_points',
                                )

        n_draws = get_dict_val(self.config,
                               self.config_defaults,
                               table='plot_fit',
                               key='n_draws',
                               )

        figsize = get_dict_val(self.config,
                               self.config_defaults,
                               table='plot_fit',
                               key='figsize',
                               )

        x_label = get_dict_val(self.config,
                               self.config_defaults,
                               table='plot_fit',
                               key='x_label',
                               )

        y_label = get_dict_val(self.config,
                               self.config_defaults,
                               table='plot_fit',
                               key='y_label',
                               )

        if self.data_type == 'spectrum':
            fit_dict = load_fit_dict(fit_dict_filename + '.pkl')
            sampler = fit_dict['sampler']
            n_comp = fit_dict['n_comp']

            if n_comp == 0:
                flat_samples = None
            else:
                flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                                 flat=True,
                                                 )

            self.fit(flat_samples=flat_samples,
                     data=self.data,
                     show_individual_components=show_individual_components,
                     show_hyperfine_components=show_hyperfine_components,
                     n_comp=n_comp,
                     n_points=n_points,
                     n_draws=n_draws,
                     figsize=figsize,
                     x_label=x_label,
                     y_label=y_label,
                     plot_name=plot_name,
                     )

        elif self.data_type == 'cube':

            n_comp = np.load('%s.npy' % n_comp_filename)
            if grid is None:
                grid = self.mask

            chunksize = get_dict_val(self.config,
                                     self.config_defaults,
                                     table='plotting',
                                     key='chunksize',
                                     )

            ij_list = [(i, j)
                       for i in range(grid.shape[0])
                       for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(
                    tqdm(
                        pool.imap(
                            partial(self.parallel_plot_fit,
                                    fit_dict_filename=fit_dict_filename,
                                    plot_name=plot_name,
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
                            chunksize=chunksize),
                        total=len(ij_list)
                    )
                )

    def parallel_plot_fit(self,
                          ij,
                          fit_dict_filename=None,
                          n_comp=None,
                          plot_name='fit',
                          show_individual_components=True,
                          show_hyperfine_components=False,
                          n_points=1000,
                          n_draws=100,
                          figsize=(10, 4),
                          x_label=r'Velocity (km s$^{-1}$)',
                          y_label='Intensity (K)',
                          ):
        """TODO"""

        if fit_dict_filename is None:
            self.logger.warning('fit_dict_filename should be defined!')
            sys.exit()

        if n_comp is None:
            self.logger.warning('n_comp should be defined!')
            sys.exit()

        i, j = ij[0], ij[1]

        cube_fit_dict_filename = '%s_%s_%s.pkl' % (fit_dict_filename, i, j)
        n_comp_pix = int(n_comp[i, j])
        data = self.data[:, i, j]
        error = self.error[:, i, j]
        if n_comp_pix > 0:
            fit_dict = load_fit_dict(cube_fit_dict_filename)
            sampler = fit_dict['sampler']
            flat_samples = sampler.get_chain(discard=self.n_steps // 2,
                                             flat=True,
                                             )
        else:
            flat_samples = None
        cube_plot_name = plot_name + '_%s_%s' % (i, j)
        self.fit(flat_samples=flat_samples,
                 data=data,
                 error=error,
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

    def fit(self,
            flat_samples,
            data,
            error,
            n_comp=1,
            plot_name='fit',
            show_individual_components=True,
            show_hyperfine_components=False,
            n_points=1000,
            n_draws=100,
            figsize=(10, 4),
            x_label=r'Velocity (km s$^{-1}$)',
            y_label='Intensity (K)',
            ):
        """TODO"""

        file_exts = get_dict_val(self.config,
                                 self.config_defaults,
                                 table='plotting',
                                 key='file_exts',
                                 )

        vel_min, vel_max = np.nanmin(self.vel), np.nanmax(self.vel)

        vel_plot_mcmc = np.linspace(vel_min, vel_max, n_points)

        if flat_samples is not None:
            fit_mcmc = super(HyperfinePlotter, self).get_fits_from_samples(samples=flat_samples,
                                                                           vel=vel_plot_mcmc,
                                                                           n_draws=n_draws,
                                                                           n_comp=n_comp,
                                                                           )
            fit_chisq = super(HyperfinePlotter, self).get_fits_from_samples(samples=flat_samples,
                                                                            vel=self.vel,
                                                                            n_draws=n_draws,
                                                                            n_comp=n_comp,
                                                                            )
            model = np.nanmedian(np.nansum(fit_chisq, axis=-1), axis=1)


            if show_individual_components:
                fit_percentiles_components = np.nanpercentile(fit_mcmc, [50, 16, 84], axis=1)
            fit_percentiles = np.nanpercentile(np.nansum(fit_mcmc, axis=-1), [50, 16, 84], axis=1)
        else:
            fit_percentiles = np.zeros([3, n_points])
            model = np.zeros_like(self.vel)

        # Calculate reduced chisq
        chisq = chi_square(data, model, observed_error=error)
        deg_freedom = len(data[~np.isnan(data)]) - (n_comp * len(self.props))
        chisq_red = chisq / deg_freedom

        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        plt.step(self.vel, data, where='mid', c='k')

        y_lim = plt.ylim()
        y_min, y_max = y_lim[0], 1.2 * np.nanmax(data)

        plt.plot(vel_plot_mcmc, fit_percentiles[0, :], c='r', zorder=99)
        if flat_samples is not None:
            plt.fill_between(vel_plot_mcmc, fit_percentiles[1, :], fit_percentiles[2, :],
                             color='r', alpha=0.75, zorder=99)

            if show_individual_components:
                plt.plot(vel_plot_mcmc, fit_percentiles_components[0, :, :], c='k')
                for i in range(fit_percentiles_components.shape[-1]):
                    plt.fill_between(vel_plot_mcmc,
                                     fit_percentiles_components[1, :, i], fit_percentiles_components[2, :, i],
                                     color='k', alpha=0.75)

        plt.xlim([vel_min, vel_max])
        plt.ylim([y_min, y_max])

        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        plt.grid()

        plt.text(0.95, 0.95, r'$\chi_\nu^2=%.2f$' % chisq_red,
                 ha='right',
                 va='top',
                 bbox=dict(facecolor='white', edgecolor='black', alpha=1),
                 transform=ax.transAxes,
                 )

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.tight_layout()

        for file_ext in file_exts:
            if file_ext not in ALLOWED_FILE_EXTS:
                self.logger.warning('file_ext should be one of %s' % ALLOWED_FILE_EXTS)
                sys.exit()

            plt.savefig('%s.%s' % (plot_name, file_ext), bbox_inches='tight')

        plt.close()
