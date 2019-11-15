import argparse
from collections import defaultdict
from operator import getitem
from pprint import pprint

import dask
import dask.array as da
import numpy as np

from daskms import xds_from_ms, xds_from_table


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
        result = defaultdict(list)

        for d in per_bl_exp:
            for ant_pair, uexp in d.items():
                result[ant_pair].append(uexp)

        return {k: np.unique(v) for k, v in result.items()}

    # Easy singleton case
    elif isinstance(per_bl_exp, dict):
        return per_bl_exp
    else:
        raise TypeError("Unhandled per_bl_exp type(%s)" % type(per_bl_exp))


args = create_parser().parse_args()

ubl_exps = []


ddid_ds = xds_from_table("::".join((args.ms, "DATA_DESCRIPTION")))[0].compute()
spw_ds = [ds.compute() for ds in
          xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                         group_cols="__row__")]
pol_ds = [ds.compute() for ds in
          xds_from_table("::".join((args.ms, "POLARIZATION")),
                         group_cols="__row__")]


for ds in xds_from_ms(args.ms, chunks={"row": args.row_chunks}):
    spw_id = ddid_ds.SPECTRAL_WINDOW_ID.values[ds.DATA_DESC_ID]
    pol_id = ddid_ds.POLARIZATION_ID.values[ds.DATA_DESC_ID]
    spw = spw_ds[spw_id]
    pol = pol_ds[pol_id]

    assert tuple(map(len, ds.FLAG.data.chunks[1:])) == (1, 1)

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
