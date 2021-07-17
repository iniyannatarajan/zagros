#!/usr/bin/env python

# INI: Adapted from casacore.tables.msutil.msregularize:

import sys
import argparse
from numpy import arange
import pyrap.tables as pt


def regularize_ms(msname):
    """ Regularize an MS

    The output MS will have the same number of baselines for each time stamp.
    All new rows are fully flagged. 

    First, missing rows are written into a separate MS <msname>_missing.MS,
    which is concatenated with the original MS and sorted in order of TIME,
    DATADESC_ID, ANTENNA1, ANTENNA2 to form a new regular MS. This MS is
    a 'deep' copy copy of the original MS.

    If no rows were missing, no new MS is created.
    """

    msprefix = msname.rsplit('.', maxsplit=1)[0]

    # Get all baselines
    tab = pt.table(msname)
    t1 = tab.sort('unique ANTENNA1,ANTENNA2')
    nadded = 0

    # Iterate in time and band over the MS
    for tsub in tab.iter(['TIME','DATA_DESC_ID']):
        nmissing = t1.nrows() - tsub.nrows()

        if nmissing < 0:
            raise ValueError("A time/band chunk has too many rows")
        elif nmissing > 0:
            ant1 = list(tsub.getcol('ANTENNA1'))
            ant2 = list(tsub.getcol('ANTENNA2'))

            # select baseline permutations that are missing from the current 'tsub'
            t2 = pt.taql('select from $t1 where !any(ANTENNA1 == $ant1 && ANTENNA2 == $ant2)')
            if t2.nrows() != nmissing:
                raise ValueError("A time/band chunk behaves strangely")

            # for the first iteration, create a new table and open for writing
            if nadded == 0:
                tnew = t2.copy(msprefix+"_missing.MS", deep=True)
                tnew = pt.table(msprefix+"_missing.MS", readonly=False)
            else:
                t2.copyrows(tnew)

            # set the correct time and band in the new rows.
            tnew.putcell('TIME', arange(nadded, nadded+nmissing), tsub.getcell('TIME',0))
            tnew.putcell('DATA_DESC_ID', arange(nadded, nadded+nmissing), tsub.getcell('DATA_DESC_ID',0))
            nadded += nmissing # update nadded

    # close tables
    t1.close()

    # combine the new table with the existing one
    if nadded > 0:
        # initialize DATA with zeros and flag the added rows
        pt.taql('update $tnew set DATA=0+0i')
        pt.taql('update $tnew set FLAG=True')
        pt.taql('update $tnew set FLAG_ROW=True')

        tcombs = pt.table([tab, tnew]).sort('TIME,DATA_DESC_ID,ANTENNA1,ANTENNA2')
        tcombs.copy(msprefix+"_regularized.MS", deep=True)

        # close and/or delete temporary tables
        tnew.close()
        pt.tabledelete(msprefix+"_missing.MS") # tnew
        t2.close()
        tcombs.close()

        print(msprefix+"_regularized.MS", 'contains the regularized MS.')

    else:
        print(f"{msname} is already regularized. No changes made.")

    # close tables before exiting
    t1.close()
    tab.close()


def main():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description="Regularize an MS so that all timestamps have entries for all baselines")
    parser.add_argument('msname', type=str, help="Name of the input MS to be regularized")
    args = parser.parse_args()

    regularize_ms(args.msname)

if __name__ == '__main__':

    ret=main()
    sys.exit(ret)

