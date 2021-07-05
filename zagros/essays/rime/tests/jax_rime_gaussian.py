from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
from scipy.constants import c as lightspeed

minus_two_pi_over_c = -2*jnp.pi/lightspeed

@jit
def gaussian(uvw, frequency, shape_params):
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum

    two = uvw.dtype.type(2.0)
    one = uvw.dtype.type(1.0)
    zero = uvw.dtype.type(0.0)

    fwhm = two * jnp.sqrt(two * jnp.log(two))
    fwhminv = one / fwhm
    gauss_scale = fwhminv * jnp.sqrt(two) * jnp.pi / lightspeed

    dtype = jnp.result_type(*(jnp.dtype(a.dtype.name) for
                             a in (uvw, frequency, shape_params)))

    nsrc = shape_params.shape[0]
    nrow = uvw.shape[0]
    nchan = frequency.shape[0]
 
    shape = jnp.empty((nsrc, nrow, nchan), dtype=dtype)
    scaled_freq = jnp.empty_like(frequency)
 
    # Scale each frequency
    for f in range(frequency.shape[0]):
        #scaled_freq[f] = frequency[f] * gauss_scale
        scaled_freq = scaled_freq.at[f].set(frequency[f] * gauss_scale)
 
    for s in range(shape_params.shape[0]):
        emaj, emin, angle = shape_params[s]
 
        # Convert to l-projection, m-projection, ratio
        el = emaj * jnp.sin(angle)
        em = emaj * jnp.cos(angle)
        er = emin / emaj
        #er = emin / (one if emaj == zero else emaj)
 
        for r in range(uvw.shape[0]):
            u, v, w = uvw[r]
 
            u1 = (u*em - v*el)*er
            v1 = u*el + v*em
 
            for f in range(scaled_freq.shape[0]):
                fu1 = u1*scaled_freq[f]
                fv1 = v1*scaled_freq[f]
 
                #shape[s, r, f] = jnp.exp(-(fu1*fu1 + fv1*fv1))
                shape = shape.at[s, r, f].set(jnp.exp(-(fu1*fu1 + fv1*fv1)))
 
    return shape


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
def fused_rime(lm, uvw, frequency, shape_params, stokes):

    # Full expansion over source axis -- very memory hungry

    # return jnp.einsum("srf,si->rfi",
    #                   phase_delay(lm, uvw, frequency),
    #                   brightness(stokes))



    source = lm.shape[0]
    row = uvw.shape[0]
    chan = frequency.shape[0]
    corr = stokes.shape[1]

    dtype = jnp.result_type(lm.dtype, uvw.dtype,
                            frequency.dtype, shape_params.dtype, stokes.dtype,
                            jnp.complex64)
    vis = jnp.empty((row, chan, corr), dtype)

    def body(s, vis):
        phdelay = phase_delay(lm[None, s], uvw, frequency)
        brness = brightness(stokes[None, s])
        '''coh = jnp.einsum("srf,si->rfi",
                         phdelay,
                         brness)'''
        gauss_shape = gaussian(uvw, frequency, shape_params[None, s])
        coh = jnp.einsum("srf,srf,si->rfi",
                         phdelay,
                         gauss_shape,
                         brness)

        return vis + coh.astype(dtype)

    return lax.fori_loop(0, source, body, vis)
