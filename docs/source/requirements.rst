===========================
Requirements & Installation
===========================

Requirements:
-------------

.. note:: Zagros must be run on a machine with a CUDA-enabled GPU device. It is possible to modify zagros to not use CUDA, but the model prediction step will be slow.

Zagros requires that the following python packages are installed:

   * `codex-africanus <https://github.com/ska-sa/codex-africanus>`_
   * `PolyChordLite <https://github.com/PolyChord/PolyChordLite>`_
   * `dyPolyChord <https://github.com/ejhigson/dyPolyChord>`_
   * `mpi4py <https://pypi.org/project/mpi4py>`_

.. note:: dyPolyChord requires the installation of the C++ library PolyChord and the corresponding python wrapper pypolychord. Both are included in PolyChordLite.

Installation:
-------------

The following instructions should set up the environment required for running Zagros. Detailed installation instructions for each dependency can be found on the links above.

.. note:: Use pip3 for python3.

1) Codex-africans can be installed in one of two ways.

The first is by using pip (recommended)::

    pip install codex-africanus[complete-cuda]

.. note:: Without the **[complete-cuda]** option above, codex-africanus will not install dependencies such as cupy, dask, scipy, astropy, and python-casacore (pyrap). If not using cuda, use **[complete]**.

The second way is to build from source::

    git clone git://github.com/ska-sa/codex-africanus
    cd codex-africanus
    python setup.py install

.. note:: Codex-africanus has many dependencies, the most important of which are cupy, dask, scipy, astropy, and python-casacore (pyrap). If building from source, make sure these are installed.

2) PyPolyChord for Linux can be installed using pip (recommended)::

    pip install pypolychord

An alternative is to download the latest PolyChordLite from github::

    git clone https://github.com/PolyChord/PolyChordLite.git

and running the following commands from within the PolyChordLite directory::

    make pypolychord
    python setup.py install --user

In this case, it is recommended that the following environment variables be set up::

    export PYTHONPATH=/path/to/PolyChordLite/pypolychord:$PYTHONPATH
    export LD_LIBRARY_PATH=/path/to/PolyChordLite/lib:$LD_LIBRARY_PATH

3) dyPolyChord can be installed by::

    pip install dyPolyChord

.. note:: dyPolyChord will install successfully without PolyChordLite, but for successful running, PolyChordLite is necessary.

4) mpi4py (optional, for exploiting the MPI capabilities of Zagros/PolyChord)::

    pip install mpi4py
