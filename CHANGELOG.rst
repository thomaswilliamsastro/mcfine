#########
Changelog
#########

================
0.3 (Unreleased)
================

* Add dependabot.yml
* Use exact version pins in pyproject.toml
* Update tutorials since some functions have changed
* Explicitly make ndradexhyperfine optional
* Added pure Gaussian option, which just models purely Gaussian line profiles with Tpeak/v/sigma
* Fix bug in map making if ``fit_type`` is not ``lte``
* Increased default sigma bound to 500, to better capture velocity dispersion on larger
  scales
* Step plot is now more colourful, and fixed potential overlap between axis and tick labels
* Added ``loc_image`` to ``plot_fit``, to locate the fit within the image

================
0.2 (01/07/2024)
================

* Updated GitHub actions
* Simplified installation files
* Move to basinhopping for lmfit initial guess, with adaptive Nelder-Mead for the local minimizer
* Refactor to use toml files
* Simplify file structure to a single output per fit
* Much improved documentation

================
0.1 (17/11/2021)
================

* Initial Release