import numpy as np

from .vars import T_BACKGROUND


def gaussian(
    x,
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

    y = amp * np.exp(-((x - centre) ** 2) / (2 * width**2))
    return y


def hyperfine_structure_lte(
    t_ex,
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
        v_centre (float): Central velocity of the strongest component (km/s).
        strength_lines (np.ndarray): Array of relative line strengths
        v_lines (np.ndarray): Array of relative velocity shifts for the lines (km/s)
        vel (np.ndarray): Velocity array (km/s)
        line_width (float): Width of components (assumed to be the same for each hyperfine component; km/s).
        return_hyperfine_components (bool): Return the intensity for each hyperfine component. Defaults to False.
        log_tau (bool): If True, will assume tau is in a logarithmic scale. Else is linear. Defaults to True.

    Returns:
        Intensity for each individual hyperfine component (if `return_hyperfine_components` is True), and the total
            intensity for all components
    """

    if log_tau:
        strength = np.exp(tau) * strength_lines
    else:
        strength = tau * strength_lines

    tau_components = gaussian(
        vel[:, np.newaxis], strength, v_lines + v_centre, line_width
    )

    total_tau = np.nansum(tau_components, axis=-1)
    intensity_total = (1 - np.exp(-total_tau)) * (t_ex - T_BACKGROUND)

    if not return_hyperfine_components:
        return intensity_total
    else:
        intensity_components = (1 - np.exp(-tau_components)) * (t_ex - T_BACKGROUND)
        return intensity_components, intensity_total


def hyperfine_structure_pure_gauss(
    t,
    v_centre,
    line_width,
    strength_lines,
    v_lines,
    vel,
    return_hyperfine_components=False,
):
    """Create hyperfine intensity profile for a pure Gaussian profile.

    Takes line strengths and relative velocity centres, along with peak temperature to
    produce a hyperfine intensity profile.

    Args:
        t (float): Line temperature (K).
        v_centre (float): Central velocity of the strongest component (km/s).
        strength_lines (np.ndarray): Array of relative line strengths
        v_lines (np.ndarray): Array of relative velocity shifts for the lines (km/s)
        vel (np.ndarray): Velocity array (km/s)
        line_width (float): Width of components (assumed to be the same for each hyperfine component; km/s).
        return_hyperfine_components (bool): Return the intensity for each hyperfine component. Defaults to False.

    Returns:
        Intensity for each individual hyperfine component (if `return_hyperfine_components` is True), and the total
            intensity for all components
    """

    intensity_components = gaussian(
        vel[:, np.newaxis], t * strength_lines, v_lines + v_centre, line_width
    )

    intensity_total = np.nansum(intensity_components, axis=-1)

    if not return_hyperfine_components:
        return intensity_total
    else:
        return intensity_components, intensity_total


def hyperfine_structure_radex(
    t_ex,
    tau,
    v_centre,
    line_width,
    v_lines,
    vel,
    return_hyperfine_components=False,
):
    """Create hyperfine intensity profile using RADEX.

    Takes line strengths and relative velocity centres calculated with RADEX, and produces a spectrum.

    Args:
        t_ex (float): Excitation temperature (K).
        tau (float): Total optical depth of the line.
        v_centre (float): Central velocity of the strongest component (km/s).
        line_width (float): Width of components (assumed to be the same for each hyperfine component; km/s).
        v_lines (float): Velocity of the various components (km/s).
        vel (np.ndarray): Velocity array (km/s)
        return_hyperfine_components (bool): Return the intensity for each hyperfine component. Defaults to False.

    Returns:
        Intensity for each individual hyperfine component (if `return_hyperfine_components` is True), and the total
            intensity for all components
    """

    tau_components = gaussian(vel[:, np.newaxis], tau, v_lines + v_centre, line_width)

    intensity_components = (1 - np.exp(-tau_components)) * (t_ex - T_BACKGROUND)

    intensity_total = np.nansum(intensity_components, axis=-1)

    if not return_hyperfine_components:
        return intensity_total
    else:
        return intensity_components, intensity_total
