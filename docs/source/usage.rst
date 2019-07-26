=====
Usage
=====

Zagros can be run as follows, after modifying the likelihood and prior functions according to the user's needs in the file zagros.py::

    $ python zagros.py <input-ms> <ms-column> --hypo <hypo-id> --npsrc <no-ptsrc> --ngsrc <no-gausrc> --npar <no-pars> --basedir <output-dir> --fileroot <outfile-prefix>

* ``<input-ms>`` is a CASA measurement set (MS) with the observed visibilities
* ``<ms-column>`` is the MS column containing the observed visibilities
* ``<hypo-id>`` is the integer id used to denote the hypothesis to be tested, defined inside zagros.py
* ``<no-ptsrc>`` is the number of point sources in the source model
* ``<no-gausrc>`` is the number of Gaussian sources in the source model
* ``<no-pars>`` is the number of free parameters (source *and* instrumental) in the model to be evaluated
* ``<output-dir>`` is the output directory in which dyPolyChord creates the output files
* ``<outfile-prefix>`` is the prefix that dyPolyChord adds to the output files

Currently, zagros can handle only point sources and Gaussian morphologies for modelling astronomical sources.

.. todo:: Update Gaussian example

.. todo:: Update example with antenna gain parameters
