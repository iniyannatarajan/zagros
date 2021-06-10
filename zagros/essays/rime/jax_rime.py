from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from jax import lax

minus_two_pi_over_c = -2*jnp.pi/3e8


@jit
def phase_delay(lm, uvw, frequency):
    out_dtype = jnp.result_type(lm, uvw, frequency, jnp.complex64)

    one = lm.dtype.type(1.0)
    neg_two_pi_over_c = lm.dtype.type(minus_two_pi_over_c)
    complex_one = out_dtype.type(1j)

    l = lm[:, 0, None, None]  # noqa
    m = lm[:, 1, None, None]

    u = uvw[None, :, 0, None]
    v = uvw[None, :, 1, None]
    w = uvw[None, :, 2, None]

    n = jnp.sqrt(one - l**2 - m**2) - one

    real_phase = (neg_two_pi_over_c *
                  (l * u + m * v + n * w) *
                  frequency[None, None, :])

    return jnp.exp(complex_one*real_phase)


@jit
def brightness(stokes):
    return jnp.stack([
        stokes[:, 0] + stokes[:, 1]*0j,
        stokes[:, 2] + stokes[:, 3]*1j,
        stokes[:, 2] - stokes[:, 3]*1j,
        stokes[:, 0] - stokes[:, 1]*0j],
        axis=1)


@jit
def coherency(nsrc, lm, uvw, frequency, stokes):
    return jnp.einsum("srf,si->srfi",
                      phase_delay(lm, uvw, frequency),
                      brightness(stokes))


@jit
def fused_rime(lm, uvw, frequency, stokes):

    # Full expansion over source axis -- very memory hungry

    # return jnp.einsum("srf,si->rfi",
    #                   phase_delay(lm, uvw, frequency),
    #                   brightness(stokes))



    source = lm.shape[0]
    row = uvw.shape[0]
    chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, stokes.dtype,
                            jnp.complex64)
    vis = np.empty((row, chan, corr), dtype)

    def body(s, vis):
        coh = jnp.einsum("srf,si->rfi",
                         phase_delay(lm[None, s], uvw, frequency),
                         brightness(stokes[None, s]))

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)