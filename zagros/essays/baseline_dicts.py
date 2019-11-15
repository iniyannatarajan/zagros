import argparse
from collections import defaultdict
from itertools import product
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


def _baseline_meta(ant1, ant2):
    """ Create a {(a1, a2): rows} dictionary """
    baselines = np.stack([ant1, ant2], axis=1)
    ubl, inv = np.unique(baselines, return_inverse=True, axis=0)

    return {(a1, a2): bl == inv for bl, (a1, a2) in enumerate(ubl)}


def baseline_meta(ant1, ant2):
    """
    Create an array of baseline metadata dictionaries. Each chunk contains
    one dictionary
    """
    return da.blockwise(_baseline_meta, ("row",),
                        ant1, ("row",),
                        ant2, ("row",),
                        meta=np.empty((0,), dtype=np.object),
                        dtype=np.object)


def _per_bl_exp(bl_meta, exp):
    """
    Create {(a1, a2): unique_exp} dictionary describing
    unique exposures for a baseline
    """
    return {(a1, a2): np.unique(exp[r]) for (a1, a2), r in bl_meta.items()}


def _noop(chunk, axis, keepdims):
    """ Does nothing to the input chunk """
    return chunk


def _merge_bl_exp(per_bl_exp, axis, keepdims):
    """ Merge lists of {(a1, a2): uexp} dictionaries """

    # List of unique exposures per baseline
    if isinstance(per_bl_exp, list):
        # List of dictionaries, merge into a final result dictionary
        result = defaultdict(list)

        for d in per_bl_exp:
            for ant_pair, uexp in d.items():
                result[ant_pair].append(uexp)

        return {k: np.unique(np.concatenate(v)) for k, v in result.items()}

    # Easy singleton case
    elif isinstance(per_bl_exp, dict):
        return per_bl_exp
    else:
        raise TypeError("Unhandled per_bl_exp type(%s)" % type(per_bl_exp))


def baseline_exposure(baseline_meta, exposure):
    """
    Performs a reduction creating unique exposures per baseline
    """

    # Find per-bl unique exposures per array chunk
    per_bl_exp = da.blockwise(_per_bl_exp, ("row",),
                              baseline_meta, ("row",),
                              exposure, ("row",),
                              dtype=np.object)

    # Then merge dictionaries together in a parallel reduction
    # operation to produce a final dask array
    # holding a single object
    reduction = da.reduction(per_bl_exp,
                             chunk=_noop,
                             combine=_merge_bl_exp,
                             aggregate=_merge_bl_exp,
                             concatenate=False,
                             split_every=16,
                             meta=np.empty((), dtype=np.object),
                             dtype=np.object)

    return reduction


def chunk_shapes(array):
    """
    Create a dask shape array where each chunk corresponds to the
    shape of that chunk of the array input
    """
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph
    name = "chunk-shapes-" + tokenize(array)
    chunks = array.chunks

    shapes = list(product(*chunks))
    ids = list(product((name,), *map(range, [len(c) for c in chunks])))
    layers = {i: s for i, s in zip(ids, shapes)}
    graph = HighLevelGraph.from_collections(name, layers, [array])

    return da.Array(graph, name, chunks, dtype=np.object)


def _baseline_sigma(sefd, chan_width, bl_exp):
    """
    Create {(a1, a2): sigma} dictionary for each baseline
    """
    assert isinstance(sefd, list)
    assert isinstance(chan_width, list)
    sefd = sefd[0]
    chan_width = chan_width[0]

    return {(a1, a2): np.sqrt(sefd[a1]*sefd[a2]/(2*chan_width[0]*exp[0]))
            for (a1, a2), exp in bl_exp.items()}


def baseline_sigma(baseline_exposure, sefd, chan_width):
    """
    Create dask array of a single chunk represented by a dictionary
    holding the sigmas for each unique baseline
    """
    assert sefd.ndim == 1 and len(sefd.chunks[0]) == 1
    assert chan_width.ndim == 1 and len(chan_width.chunks[0]) == 1
    return da.blockwise(_baseline_sigma, (),
                        sefd, ("ant",),
                        chan_width, ("chan",),
                        baseline_exposure, (),
                        meta=np.empty((), dtype=np.object),
                        dtype=np.object)


def _create_weight_vector(shape, bl_sigma, bl_meta, dtype_):
    """
    Create the weight vector from the sigmas for each baseline
    """
    weight = np.zeros(shape, dtype_)

    for (a1, a2), sigma in bl_sigma.items():
        rows = bl_meta[(a1, a2)]
        weight[rows] = 1.0 / (sigma**2)

    return weight


def create_weight_vector(vis, baseline_sigma, baseline_meta):
    """
    Create a dask array weight vector from the sigmas for
    each baseline
    """
    return da.blockwise(_create_weight_vector, ("row", "chan", "corr"),
                        chunk_shapes(vis), ("row", "chan", "corr"),
                        baseline_sigma, (),
                        baseline_meta, ("row",),
                        dtype_=vis.real.dtype,
                        dtype=vis.real.dtype)


if __name__ == "__main__":
    args = create_parser().parse_args()

    ddid_ds = xds_from_table("::".join((args.ms, "DATA_DESCRIPTION")))[0].compute()
    ant_ds = xds_from_table("::".join((args.ms, "ANTENNA")))[0]
    spw_ds = xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                            group_cols="__row__")
    pol_ds = xds_from_table("::".join((args.ms, "POLARIZATION")),
                            group_cols="__row__")

    results = []

    for ds in xds_from_ms(args.ms, chunks={"row": args.row_chunks}):
        spw_id = ddid_ds.SPECTRAL_WINDOW_ID.values[ds.DATA_DESC_ID]
        pol_id = ddid_ds.POLARIZATION_ID.values[ds.DATA_DESC_ID]
        spw = spw_ds[spw_id]
        pol = pol_ds[pol_id]

        ants = ant_ds.POSITION.shape[0]
        sefd = da.random.random(ants, chunks=ants)

        bl_meta = baseline_meta(ds.ANTENNA1.data, ds.ANTENNA2.data)
        bl_exp = baseline_exposure(bl_meta, ds.EXPOSURE.data)
        bl_sigma = baseline_sigma(bl_exp, sefd, spw.CHAN_WIDTH.data[0])

        weight = create_weight_vector(ds.DATA.data, bl_sigma, bl_meta)

        results.append(weight)

    for result in dask.compute(results):
        pprint(result)
