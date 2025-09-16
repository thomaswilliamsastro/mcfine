#########
Changelog
#########

================
0.3 (2025-09-16)
================

* General script tidying up
* Add ``adaptive`` options for the emcee, to automatically choose walkers/steps
* Add option to reduce the emcee sampler down to a covariance matrix, to minimize space requirement
* Ensure likelihood is correctly updated when encouraging spatial coherence
* Use hardlinks where possible in coherence to minimize space requirements
* Fix crash if data has values but error does not
* Fix crash if NaNs in spectrum when doing derivative spectroscopy
* Add 21cm HI line to line list
* Add in AIC criterion alongside BIC, by default this is also 10 like BIC
* Add option to get initial velocities for LMFIT via iterative derivative spectroscopy
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
0.2 (2024-07-01)
================

* Updated GitHub actions
* Simplified installation files
* Move to basinhopping for lmfit initial guess, with adaptive Nelder-Mead for the local minimizer
* Refactor to use toml files
* Simplify file structure to a single output per fit
* Much improved documentation

================
0.1 (2021-11-17)
================

* Initial Release