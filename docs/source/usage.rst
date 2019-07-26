=====
Usage
=====

Zagros can be run as follows, after modifying the likelihood and prior functions according to the user's needs in the file zagros.py::

    $ python zagros.py ``<input-ms> <ms-column> --hypo <hypo-id> --npsrc <no-ptsrc> --ngsrc <no-gausrc> --npar <no-pars> --basedir <output-dir> --fileroot <outfile-prefix>``

Currently, zagros can handle only point and Gaussian profiles for modelling astronomical sources.
