import argparse
from collections import defaultdict
from operator import getitem
from pprint import pprint

import dask
import dask.array as da
import numpy as np

from daskms import xds_from_ms


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000)
    return p


def _per_bl_exp(ant1, ant2, exp):
    baselines = np.stack([ant1, ant2], axis=1)
    ubl, inv = np.unique(baselines, return_inverse=True, axis=0)

    return {(a1, a2): np.unique(exp[bl_idx == inv])
            for bl_idx, (a1, a2) in enumerate(ubl)}


def _noop(chunk, axis, keepdims):
    return chunk


def _merge_bl_exp(per_bl_exp, axis, keepdims):
    # List of unique exposures per baseline
    if isinstance(per_bl_exp, list):
        result = {}

        for d in per_bl_exp:
            for ant_pair, uexp in d.items():
                try:
                    existing_uexp = result[ant_pair]
                except KeyError:
                    # Assign
                    result[ant_pair] = uexp
                else:
                    # Merge exposure arrays
                    result[ant_pair] = np.unique([uexp, existing_uexp])

        return result
    # Easy singleton case
    elif isinstance(per_bl_exp, dict):
        return per_bl_exp
    else:
        raise TypeError("Unhandled per_bl_exp type(%s)" % type(per_bl_exp))


args = create_parser().parse_args()

ubl_exps = []

for ds in xds_from_ms(args.ms, chunks={"row": args.row_chunks}):
    per_bl_exp = da.blockwise(_per_bl_exp, ("row",),
                              ds.ANTENNA1.data, ("row",),
                              ds.ANTENNA2.data, ("row",),
                              ds.EXPOSURE.data, ("row",),
                              meta=np.empty((0,), dtype=np.object),
                              dtype=np.object)

    reduction = da.reduction(per_bl_exp,
                             chunk=_noop,
                             combine=_merge_bl_exp,
                             aggregate=_merge_bl_exp,
                             concatenate=False,
                             split_every=16,
                             meta=np.empty((0,), dtype=np.object),
                             dtype=np.object)

    ubl_exps.append(reduction)


for ubl_exps in dask.compute(ubl_exps):
    pprint(ubl_exps)
