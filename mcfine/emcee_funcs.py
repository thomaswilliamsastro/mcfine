import logging
import warnings

import numpy as np

logger = logging.getLogger("mcfine")


def get_samples(
    sampler,
    burn_in_frac,
    thin_frac,
):
    """Get samples from an emcee sampler

    Args:
        sampler: emcee sampler
        burn_in_frac: burn-in fraction as a function of autocorrelation time
        thin_frac: thin fraction as a function of autocorrelation time
    """

    # Get the burn-in and thin parameters from
    # the sampler
    burn_in, thin = get_burn_in_thin(
        sampler,
        burn_in_frac=burn_in_frac,
        thin_frac=thin_frac,
    )

    samples = sampler.get_chain(
        discard=burn_in,
        thin=thin,
        flat=True,
    )

    return samples


def get_burn_in_thin(
    sampler,
    burn_in_frac,
    thin_frac,
):
    """Get burn-in and thin from a sampler

    Args:
        sampler: emcee sampler
        burn_in_frac: burn-in fraction as a function of autocorrelation time
        thin_frac: thin fraction as a function of autocorrelation time
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau = sampler.get_autocorr_time(tol=0)

    # Fallback if the autocorrelation time is
    # all NaN
    if np.all(np.isnan(tau)):
        logger.debug("Autocorrelation time is NaN, will use all samples")
        burn_in = 0
        thin = 1
    else:
        burn_in = int(burn_in_frac * np.nanmax(tau))
        thin = int(thin_frac * np.nanmin(tau))

        # Ensure we don't have a slice step of 0
        if thin == 0:
            logger.debug("Thin is 0, will round up to 1")
            thin = 1

    return burn_in, thin


def get_samples_from_fit_dict(
    fit_dict,
    burn_in_frac,
    thin_frac,
):
    """Pull out a bunch of samples from a fit dictionary

    Args:
        fit_dict (dict): Fit dictionary
        burn_in_frac: burn-in fraction as a function of autocorrelation time
        thin_frac: thin fraction as a function of autocorrelation time
    """

    # If we have the full emcee sampler, prefer that here
    if "sampler" in fit_dict:

        sampler = fit_dict["sampler"]
        flat_samples = get_samples(
            sampler,
            burn_in_frac=burn_in_frac,
            thin_frac=thin_frac,
        )

    # Otherwise, sample from the covariance matrix
    else:

        cov_matrix = fit_dict["cov_matrix"]
        cov_med = fit_dict["cov_med"]
        flat_samples = np.random.multivariate_normal(cov_med, cov_matrix, size=10000)

    return flat_samples
