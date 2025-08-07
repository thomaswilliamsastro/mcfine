##################################################
McFine: A tool for hyperfine spectral line fitting
##################################################

.. image:: https://img.shields.io/pypi/v/mcfine.svg?label=PyPI&style=flat-square
   :target: https://pypi.org/pypi/mcfine/
.. image:: https://img.shields.io/pypi/pyversions/mcfine.svg?label=Python&color=yellow&style=flat-square
   :target: https://pypi.org/pypi/mcfine/
.. image:: https://img.shields.io/github/actions/workflow/status/thomaswilliamsastro/mcfine/build.yml?branch=main&style=flat-square
   :target: https://github.com/thomaswilliamsastro/mcfine/actions
.. image:: https://readthedocs.org/projects/mcfine/badge/?version=latest&style=flat-square
   :target: https://mcfine.readthedocs.io/en/latest/
.. image:: https://img.shields.io/badge/license-GNUv3-blue.svg?label=License&style=flat-square

Much of radio astronomy is using magic to turn line intensities into gas conditions. Magic because many molecules are
complicated, and there are often multiple distinct components down the line of sight. McFine attempts to do this
wizardry in a fully automated, Bayesian way so you can turn your spectra into science without too much hassle.

McFine uses an iterative approach, fitting and comparing increasingly complex models until it deems them sufficiently
complicated. It does this through the Bayesian and Akaike Information Criterion, and specifically the change in these
metrics between models. If given a data cube, it can also use the neighbouring information to attempt a better fit.
For more details about the philosophy and maths of McFine, have a read of Williams & Watkins (2024). It's quite
short!