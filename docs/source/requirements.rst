===========================
Requirements & Installation
===========================

Zagros requires that the following python packages are installed:

   * `codex-africanus <https://github.com/ska-sa/codex-africanus>`_
   * `PolyChordLite <https://github.com/PolyChord/PolyChordLite>`_
   * `dyPolyChord <https://github.com/ejhigson/dyPolyChord>`_

.. note:: dyPolyChord requires the installation of the C++ library PolyChord and the corresponding python wrapper pypolychord. Both are included in PolyChordLite.

Quick installation instructions:
--------------------------------

The following instructions should set up the environment required for running Zagros. Detailed installation instructions for each dependency can be found on the links above.

.. note:: Zagros must be run on a machine with a CUDA-enabled GPU device. It is possible to modify zagros to not use CUDA, but the model prediction step will be slow.

1) Codex-africans can be installed in one of two ways.

Using pip::

    pip install codex-africanus[complete]

.. note:: Without the **[complete]** option above, codex-africanus will not install dependencies such as cupy, dask, scipy, astropy, and python-casacore (pyrap).

Building from source::

    git clone git://github.com/ska-sa/codex-africanus
    cd codex-africanus
    python setup.py install

.. note:: Codex-africanus has many dependencies, the most important of which are cupy, dask, scipy, astropy, and python-casacore (pyrap). If building from source, make sure these are installed.

2) PolyChord can be installed by downloading PolyChordLite from github::

    git clone https://github.com/PolyChord/PolyChordLite.git

And running the following commands from within the PolyChordLite directory::

    make pypolychord
    python setup.py install --user

It is recommended that the following environment variables be set up::

    export PYTHONPATH=/path/to/PolyChordLite/pypolychord:$PYTHONPATH
    export LD_LIBRARY_PATH=/path/to/PolyChordLite/lib:$LD_LIBRARY_PATH

3) dyPolyChord can be installed by::

    pip install dyPolyChord

.. note:: dyPolyChord will install successfully without PolyChordLite, but for successful sampling, PolyChordLite is necessary.
