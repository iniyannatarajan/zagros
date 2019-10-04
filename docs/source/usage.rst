=====
Usage
=====

Zagros can be run as follows, after modifying the likelihood and prior functions according to the user's needs in the file zagros.py::

    python zagros.py <input-ms> <ms-column> --hypo <hypo-id> --npar <no-pars> --basedir <output-dir> --fileroot <outfile-prefix>

* ``<input-ms>`` is a CASA measurement set (MS) with the observed visibilities
* ``<ms-column>`` is the MS column containing the observed visibilities
* ``<hypo-id>`` is the integer id used to denote the hypothesis to be tested, defined inside zagros.py
* ``<no-pars>`` is the number of free parameters (source *and* instrumental) in the model to be evaluated
* ``<output-dir>`` is the output directory in which dyPolyChord creates the output files
* ``<outfile-prefix>`` is the prefix that dyPolyChord adds to the output files

The aim is to perform statistical analyses of radio interferometric visibilities, by constructing parameteric models based on physical assumptions and comparing them against the observed data. 
The user must define the model parameters and construct their priors and the likelihood computation.
The model construction (during each iteration of the likelihood computation) is facilitated by **codex-africanus** and the numerical sampling of the model parameteres is done by **dyPolyChord**.

To distribute computation between multiple cores, use MPI::

    mpirun -np <no-cores> python zagros.py <input-ms> <ms-column> --hypo <hypo-id> --npar <no-pars> --basedir <output-dir> --fileroot <outfile-prefix>

Setting up the RIME:
--------------------

The Radio Interferometry Measurement Equation (RIME) is given by

.. math::

    V_{pq} = G_{p} \left(
        \sum_{s} E_{ps} L_{p} K_{ps}
        B_{s}
        K_{qs}^H L_{q}^H E_{qs}^H
        \right) G_{q}^H

where for antenna :math:`p` and :math:`q`, and source :math:`s`:

- :math:`G_{p}` represents direction-independent effects.
- :math:`E_{ps}` represents direction-dependent effects.
- :math:`L_{p}` represents the feed rotation.
- :math:`K_{ps}` represents the phase delay term.
- :math:`B_{s}` represents the brightness matrix.

Important Notes:
----------------

.. note:: The DIE and DDE Jones matrices kwargs to predict_vis() corresponding to both ant1 and ant2 must be the same; the Hermitian conjugate will be performed inside africanus;
          no need for the user to do this externally.

.. note:: The zagros.py script can be used as a template to analyse more complex source and instrumental effects.
