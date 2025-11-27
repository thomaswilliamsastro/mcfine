import numpy as np


def residual(
    observed,
    model,
    observed_error=None,
):
    """Calculate standard residual.

    If errors are provided, then this is the sum of :math:`({\\rm obs}-{\\rm model})/{\\rm error}`. Else the sum of
    :math:`({\\rm obs}-{\\rm model})/{\\rm model}`.

    Args:
        observed (np.ndarray): Observed values.
        model (np.ndarray): Model values.
        observed_error (float or np.ndarray): The error in the observed values. Defaults to None.

    Returns:
        float: The residual value.

    """

    if observed_error is not None:
        res = (observed - model) / observed_error
    else:
        res = (observed - model) / model

    return res


def chi_square(
    observed,
    model,
    observed_error=None,
):
    """Calculate standard chi-square.

    If errors are provided, then this is the sum of :math:`({\\rm obs}-{\\rm model})^2/{\\rm error}^2`. Else the sum of
    :math:`({\\rm obs}-{\\rm model})^2/{\\rm model}^2`.

    Args:
        observed (np.ndarray): Observed values.
        model (np.ndarray): Model values.
        observed_error (float or np.ndarray): The error in the observed values. Defaults to None.

    Returns:
        float: The chi-square value.

    """

    res = residual(observed, model, observed_error)
    chisq = np.nansum(res**2)

    return chisq


def ln_prior(
    theta,
    vel,
    props,
    bounds,
    n_comp=1,
):
    """Apply flat priors to the emcee fitting

    This also ensures any velocity components are strictly increasing,
    to avoid degeneracies there

    Args:
        theta: Parameters for the fit
        vel: Observed velocity
        props: List of properties being fit
        bounds: Bounds on parameters
        n_comp: Number of components to fit. Defaults to 1

    Returns:
        0 if within bounds, -infinity otherwise

    """

    prop_len = len(props)

    for prop in range(prop_len):

        values = np.array([theta[prop_len * i + prop] for i in range(n_comp)])

        if not np.logical_and(
            bounds[prop][0] <= values, values <= bounds[prop][1]
        ).all():
            return -np.inf

    if n_comp > 1:

        dv = np.abs(np.nanmedian(np.diff(vel)))

        # Insist on monotonically increasing velocity components
        v_idx = np.where(np.asarray(props) == "v")[0][0]

        vels = np.array([theta[prop_len * i + v_idx] for i in range(n_comp)])
        vel_diffs = np.diff(vels)

        # Make sure components are separated by at least one channel
        if np.any(vel_diffs < dv):
            return -np.inf

    return 0
