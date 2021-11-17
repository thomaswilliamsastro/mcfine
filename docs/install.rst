#################
Installing McFine
#################

============
Requirements
============

McFine requires (the brackets indicate the version tested on, earlier versions may work but not explicitly supported):

* python (3.9)
* astropy (4.3.1)
* emcee (3.1.1)
* lmfit (1.0.3)
* ndradexhyperfine (0.2.4, this is a forked version of ndradex with some bug fixes specific to hyperfine lines)
* numpy (1.21.4)
* scipy (1.7.2)
* tqdm (4.50.0)

The plotting requires:

* matplotlib (3.4.3)
* corner (2.2.1)

Although not a requirement, each fit takes around 1 minute. So, if you're fitting a big cube, it is highly recommended
to run on a cluster. McFine uses the python multiprocessing library to speed up the fitting as much as possible.

============
Installation
============


* McFine is pip installable:

.. code-block:: shell

    pip install mcfine

* Alternatively, you can install from the Github repository:

.. code-block:: shell

    cd install_dir
    git clone https://github.com/thomaswilliamsastro/mcfine
    pip install .