import multiprocessing as mp
import warnings
from functools import partial

import corner
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import tqdm

from .fitting import *


class HyperfinePlotter(HyperfineFitter):

    def __init__(self, data, vel, error=None, mask=None, fit_type='lte', fit_method='mcmc',
                 line='n2hp10', rest_freq=None, radex_datafile=None,
                 p0=None, delta_bic_cutoff=10, n_walkers=100, n_steps=1000, n_cores=None):
        """TODO: Docstring"""

        HyperfineFitter.__init__(self, data, vel, error, mask, fit_type, fit_method,
                                 line, rest_freq, radex_datafile,
                                 p0, delta_bic_cutoff, n_walkers, n_steps, n_cores)

    def plot_step(self, plot_name='step_plot', grid=None, n_comp_filename='n_comp', sampler_filename='sampler',
                  chunksize=1):

        if self.data_type == 'spectrum':
            sampler = load_sampler(sampler_filename + '.pkl')
            n_comp = int(np.loadtxt(n_comp_filename + '.txt'))
            if n_comp == 0:
                return
            samples = sampler.get_chain()
            self.step(samples, plot_name, n_comp=n_comp)
        elif self.data_type == 'cube':

            n_comp = np.load(n_comp_filename + '.npy')
            if grid is None:
                grid = self.mask

            ij_list = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(tqdm(pool.imap(partial(self.parallel_step,
                                            plot_name=plot_name,
                                            sampler_filename=sampler_filename,
                                            n_comp=n_comp),
                                    ij_list, chunksize=chunksize), total=len(ij_list)))

    def parallel_step(self, ij, plot_name='step_plot', sampler_filename='sampler', n_comp=None):

        if n_comp is None:
            raise Warning('n_comp should be defined!')

        i, j = ij[0], ij[1]

        cube_sampler_filename = sampler_filename + '_%s_%s.pkl' % (i, j)
        n_comp_pix = int(n_comp[i, j])
        if n_comp_pix == 0:
            return
        sampler = load_sampler(cube_sampler_filename)
        samples = sampler.get_chain()
        cube_plot_name = plot_name + '_%s_%s' % (i, j)
        self.step(samples, plot_name=cube_plot_name, n_comp=n_comp_pix)

    def step(self, samples, plot_name='step_plot', n_comp=1):

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
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step Number")

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    def plot_corner(self, plot_name='corner_plot', grid=None, n_comp_filename='n_comp', sampler_filename='sampler',
                    chunksize=1):

        if self.data_type == 'spectrum':
            sampler = load_sampler(sampler_filename + '.pkl')
            n_comp = int(np.loadtxt(n_comp_filename + '.txt'))
            if n_comp == 0:
                return
            flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
            self.corner(flat_samples, plot_name, n_comp=n_comp)

        elif self.data_type == 'cube':

            n_comp = np.load(n_comp_filename + '.npy')
            if grid is None:
                grid = self.mask

            ij_list = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(tqdm(pool.imap(partial(self.parallel_corner,
                                            plot_name=plot_name,
                                            sampler_filename=sampler_filename,
                                            n_comp=n_comp),
                                    ij_list, chunksize=chunksize), total=len(ij_list)))

    def parallel_corner(self, ij, plot_name='corner_plot', sampler_filename=None, n_comp=None):

        if sampler_filename is None or n_comp is None:
            raise Warning('sampler_filename and n_comp should be defined!')

        i, j = ij[0], ij[1]
        n_comp_pix = int(n_comp[i, j])

        cube_sampler_filename = sampler_filename + '_%s_%s.pkl' % (i, j)
        if n_comp_pix == 0:
            return
        sampler = load_sampler(cube_sampler_filename)
        flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
        cube_plot_name = plot_name + '_%s_%s' % (i, j)
        self.corner(flat_samples, plot_name=cube_plot_name, n_comp=n_comp_pix)

    def corner(self, flat_samples, plot_name, n_comp=1):

        # Load up the labels
        labels = []
        for i in range(n_comp):
            for label in self.labels:
                labels.append(label % i)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corner.corner(flat_samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    def plot_fit(self, grid=None, show_individual_components=True, show_hyperfine_components=False,
                 plot_name='fit_plot', n_comp_filename='n_comp', sampler_filename='sampler',
                 n_points=1000, n_draws=100, figsize=(10, 4),
                 x_label=r'Velocity (km s$^{-1}$)', y_label='Intensity (K)',
                 chunksize=1):

        if self.data_type == 'spectrum':
            sampler = load_sampler(sampler_filename + '.pkl')
            n_comp = int(np.loadtxt(n_comp_filename + '.txt'))
            if n_comp == 0:
                flat_samples = None
            else:
                flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
            self.fit(flat_samples, self.data, show_individual_components, show_hyperfine_components,
                     int(n_comp), n_points, n_draws,
                     figsize, x_label, y_label, plot_name)
        elif self.data_type == 'cube':

            n_comp = np.load(n_comp_filename + '.npy')
            if grid is None:
                grid = self.mask

            ij_list = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
                       if grid[i, j] != 0]

            with mp.Pool(self.n_cores) as pool:
                list(tqdm(pool.imap(partial(self.parallel_plot_fit,
                                            sampler_filename=sampler_filename,
                                            plot_name=plot_name,
                                            show_individual_components=show_individual_components,
                                            show_hyperfine_components=show_hyperfine_components,
                                            n_comp=n_comp, n_points=n_points, n_draws=n_draws, figsize=figsize,
                                            x_label=x_label, y_label=y_label),
                                    ij_list, chunksize=chunksize), total=len(ij_list)))

    def parallel_plot_fit(self, ij, plot_name='fit_plot',
                          show_individual_components=True, show_hyperfine_components=False, sampler_filename=None,
                          n_comp=None, n_points=1000, n_draws=100,
                          figsize=(10, 4), x_label=r'Velocity (km s$^{-1}$)', y_label='Intensity (K)'):

        if not sampler_filename and n_comp is None:
            raise Warning('sampler_filename and n_comp should be defined!')

        i, j = ij[0], ij[1]

        cube_sampler_filename = sampler_filename + '_%s_%s.pkl' % (i, j)
        n_comp_pix = int(n_comp[i, j])
        data = self.data[:, i, j]
        if n_comp_pix > 0:
            sampler = load_sampler(cube_sampler_filename)
            flat_samples = sampler.get_chain(discard=self.n_steps // 2, flat=True)
        else:
            flat_samples = None
        cube_plot_name = plot_name + '_%s_%s' % (i, j)
        self.fit(flat_samples, data, show_individual_components=show_individual_components,
                 show_hyperfine_components=show_hyperfine_components,
                 n_comp=n_comp_pix, n_points=n_points, n_draws=n_draws,
                 figsize=figsize, x_label=x_label, y_label=y_label, plot_name=cube_plot_name)

    def fit(self, flat_samples, data, show_individual_components=True, show_hyperfine_components=False,
            n_comp=1, n_points=1000, n_draws=100,
            figsize=(10, 4), x_label=r'Velocity (km s$^{-1}$)', y_label='Intensity (K)', plot_name='fit_plot'):

        vel_min, vel_max = np.nanmin(self.vel), np.nanmax(self.vel)

        vel_plot_mcmc = np.linspace(vel_min, vel_max, n_points)

        if flat_samples is not None:
            fit_mcmc = super(HyperfinePlotter, self).get_fits_from_samples(flat_samples, vel_plot_mcmc, n_draws, n_comp)

            if show_individual_components:
                fit_percentiles_components = np.nanpercentile(fit_mcmc, [50, 16, 84], axis=1)
            fit_percentiles = np.nanpercentile(np.nansum(fit_mcmc, axis=-1), [50, 16, 84], axis=1)
        else:
            fit_percentiles = np.zeros([3, n_points])

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

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.tight_layout()

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()
