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

/* ---- 2x oversampling (-oversample) ----------------------------------------
   The delay line is read as a signal at twice the sample rate: u[2k] = x[k]
   (the input), u[2k+1] = o[k + K] (the half-sample values, from a halfband
   interpolator of 2K taps). The 4-point interpolation then works on u, where
   the signal is at most a quarter of u's rate: at 20 kHz (48 kHz) and the
   worst fraction 0.5 it loses 0.5 dB instead of 8.4 dB. o[k] is computed when
   x[k] arrives, so it lags by K samples: delays must be at least K + 1. */
#ifndef TAPMATRIX_OS_K
#define TAPMATRIX_OS_K 24           /* 48 taps: passband and images < -91 dB up to 20 kHz at 48 kHz */
#endif
#define TAPMATRIX_OS_BETA 9.0       /* Kaiser window */

/* the halfband's taps c[j] for the offsets +-(j + 1/2): a Kaiser-windowed sinc,
   normalised to DC gain 1 */
static void halfband_coeffs(t_sample *c, int K, double beta)
{
    double sum = 0, tmp[64];
    for (int j = 0; j < K && j < 64; j++) {
        const double t = j + 0.5, r = 1 - (t / K) * (t / K);
        double i0a = 1, i0b = 1, ta = 1, tb = 1;            /* I0 by its series */
        for (int m = 1; m < 40; m++) {
            ta *= (beta * sqrt(r > 0 ? r : 0) / 2) / m;
            tb *= (beta / 2) / m;
            i0a += ta * ta, i0b += tb * tb;
        }
        tmp[j] = sin(3.141592653589793 * t) / (3.141592653589793 * t) * i0a / i0b;
        sum += 2 * tmp[j];
    }
    for (int j = 0; j < K; j++) c[j] = (t_sample)(tmp[j] / sum);
}

/* o[i] = the half-sample value between x[i-K] and x[i-K+1], i = 0 .. n-1;
   reads x[i-2K+1 .. i]. Loops in this order vectorise without reordering sums. */
static void halfband(const t_sample *x, const t_sample *c, int K, t_sample *o, int n)
{
    for (int i = 0; i < n; i++) o[i] = 0;
    for (int j = 0; j < K; j++) {
        const t_sample cj = c[j];
        const t_sample *lo = x - K - j, *hi = x - K + 1 + j;
        for (int i = 0; i < n; i++) o[i] += cj * (lo[i] + hi[i]);
    }
}

/* a static tap on u: its 4 points alternate between the two rings, so it is
   two 2-tap FIRs on contiguous memory: p holds points a (newest) and c,
   q points b and d: out[i] += h0 p[i] + h1 q[i] + h2 p[i-1] + h3 q[i-1] */
static inline void fir2x2(const t_sample *p, const t_sample *q, const t_sample h[4], t_sample *out, int n)
{
    const t_sample h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3];
    for (int i = 0; i < n; i++)
        out[i] += h0 * p[i] + h1 * q[i] + h2 * p[i - 1] + h3 * q[i - 1];
}

/* a moving tap on u: as moving(), with the read position in u's samples
   (twice the input's); x and o: logical 0 of the two rings */
static void moving_os(const t_sample *x, const t_sample *o, int mask, int K, int w, t_sample d0,
                      t_sample dd, t_sample g0, t_sample gd, t_sample *out, int n)
{
    uint64_t T = ((uint64_t)(uint32_t)(2 * w) << 32) - (uint64_t)fix32(2 * d0);
    const uint64_t inc = (2ull << 32) - (uint64_t)fix32(2 * dd);
    t_sample g = g0;
#define TM_U(m) (((m) & 1) ? o[(((m) >> 1) + K) & mask] : x[((m) >> 1) & mask])
    for (int i = 0; i < n; i++, T += inc, g += gd) {
        const uint32_t j = (uint32_t)(T >> 32);
        const t_sample fp = fracpos(T), frac = 1 - fp;
        const t_sample c = TM_U(j), b = TM_U(j + 1), a = TM_U(j + 2), d = TM_U(j - 1), cb = c - b;
        out[i] += g * (b + frac * (cb - TMK(1.0 / 6) * fp * ((d - a - 3 * cb) * frac + (d + 2 * a - 3 * b))));
    }
#undef TM_U
}

#endif
