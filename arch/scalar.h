/* tapmatrix~: the portable float kernels, in t_sample (float, or double with
 * PD_FLOATSIZE=64). The SIMD headers (neon.h, ...) give the same results up to
 * rounding and fall back to these.
 *   coeffs()   4-point coefficients of delread4~, times a gain
 *   fir4()     a static tap: a 4-tap FIR added into the output
 *   moving()   a moving tap: delay and gain ramp per sample
 */
#ifndef TAPMATRIX_SCALAR_H
#define TAPMATRIX_SCALAR_H

#include <math.h>
#include <stdint.h>

/* a constant in the sample type (no double promotion in float builds) */
#define TMK(v) ((t_sample)(v))

#if PD_FLOATSIZE == 64
#define tm_floor floor
#define tm_fabs fabs
#else
#define tm_floor floorf
#define tm_fabs fabsf
#endif

/* 4-point coefficients of delread4~, for a (newest), b, c, d (oldest);
   frac goes from b (0) to c (1) */
static inline void coeffs(t_sample frac, t_sample g, t_sample h[4])
{
    /* Pd's formula is linear in a, b, c, d: evaluate it for one input at a time */
    const t_sample f1 = 1 - frac;
    for (int k = 0; k < 4; k++) {
        const t_sample a = k == 0, b = k == 1, c = k == 2, d = k == 3, cb = c - b;
        h[k] = g * (b + frac * (cb - TMK(1.0 / 6) * f1 * ((d - a - 3 * cb) * frac + (d + 2 * a - 3 * b))));
    }
}

/* one static tap: out[i] += h0 x[i] + h1 x[i-1] + h2 x[i-2] + h3 x[i-3] */
static inline void fir4(const t_sample *x, const t_sample h[4], t_sample *out, int n)
{
    const t_sample h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3];
    for (int i = 0; i < n; i++)
        out[i] += h0 * x[i] + h1 * x[i - 1] + h2 * x[i - 2] + h3 * x[i - 3];
}

/* x samples -> 32.32 fixed point; a float splits exactly into two halves */
static inline int64_t fix32(t_sample x)
{
    const t_sample hi = tm_floor(x);
    const t_sample lo = (x - hi) * 65536;           /* exact, [0, 65536) */
    const t_sample lohi = tm_floor(lo);
    return (int64_t)(int32_t)hi * 4294967296ll + (int64_t)(int32_t)lohi * 65536 +
        (int64_t)(int32_t)((lo - lohi) * 65536);
}

/* the fraction of a 32.32 position, [0, 1): in float only 24 bits (rounding
   the full 32 could give 1) */
static inline t_sample fracpos(uint64_t T)
{
#if PD_FLOATSIZE == 64
    return (t_sample)(uint32_t)T * (1.0 / 4294967296.0);
#else
    return (t_sample)((uint32_t)T >> 8) * TMK(1.0 / 16777216);
#endif
}

/* one moving tap, per sample: delay d0 -> d0 + dd*n, gain g0 -> g0 + gd*n.
   The read position t = w + i - d is a 32.32 integer: between x[j] (older)
   and x[j+1] */
static void moving(const t_sample *ring, int mask, int w, t_sample d0, t_sample dd, t_sample g0,
                   t_sample gd, t_sample *out, int n)
{
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd);
    t_sample g = g0;
    for (int i = 0; i < n; i++, T += inc, g += gd) {
        const int j = (int)(uint32_t)(T >> 32);
        const t_sample fp = fracpos(T), frac = 1 - fp;
        const t_sample c = ring[j & mask], b = ring[(j + 1) & mask],
            a = ring[(j + 2) & mask], d = ring[(j - 1) & mask], cb = c - b;
        out[i] += g * (b + frac * (cb - TMK(1.0 / 6) * fp * ((d - a - 3 * cb) * frac + (d + 2 * a - 3 * b))));
    }
}

#endif
