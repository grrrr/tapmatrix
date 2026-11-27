/* tapmatrix~: the fixed engine (16-bit fixed point). On the ESP32-P4 the kernels
 * are the PIE SIMD routines in arch/fixed_pie.S; on hosts the same integer
 * arithmetic in C below, to test the engine; other ESP32s have no fixed engine.
 *   TAPMATRIX_FIXED, TAPMATRIX_PIE        what is there
 *   tapmatrix_fixed_sum(), tapmatrix_fixed_moving()   the kernels
 *   t_fixedtap, fixed_ramp_start(), fixed_ramp_block()  a moving tap's preparation
 *   to_fixed(), fixed_from()                conversions
 *
 * The number format: a sample, gain or coefficient v in [-1, 1) is the int16
 * n = round(v * 32768), i.e. 1 sign bit and 15 fraction bits ("Q15"). A
 * product of two such numbers has 30 fraction bits; shifting it right by 15
 * gives the format back. The kernels add the products of all taps of an
 * output in a wide accumulator (PIE: QACC, 8 lanes of 40 bits; here int64)
 * and shift once at the end, so only the final result rounds and saturates.
 * The shift is 15 for this format; `fixedshift` changes it for tests only
 * (e.g. 14 doubles the output to look at headroom).
 * Gains: |g| < 1 (a larger one saturates); a moving tap's gain also >= 0.
 */
#ifndef TAPMATRIX_FIXED_H
#define TAPMATRIX_FIXED_H

#include <stdint.h>
#include <string.h>

/* One moving tap for one block, prepared for the moving kernel (112 bytes,
   16-aligned). Chunk c = samples 8c .. 8c+7. Lane i reads a 5-point window
   base[c][i .. i+4]. With t = a[c] + v[i] (16-bit, never overflows): the
   fraction of the read position past x[j] is t & 0x7fff (15 bits), and lanes with
   t < 0 use window points 0..3 (oldest first: d c b a), the others points 1..4
   (the read position moved by one sample within the chunk). mode[c] != 0: no
   such step in the chunk, all lanes use points 0..3 (1) or 1..4 (2). Gain of
   sample 8c+i: g0[c] + rg[i]. */
typedef struct {
    const int16_t *base[8];
    int16_t g0[8];
    int16_t a[8];
    int16_t mode[8];
    int16_t v[8] __attribute__((aligned(16)));
    int16_t rg[8] __attribute__((aligned(16)));
} __attribute__((aligned(16))) t_mvtap;     /* offsets used by fixed_pie.S */
_Static_assert(sizeof(void *) != 4 || sizeof(t_mvtap) == 112, "t_mvtap: fixed_pie.S expects 112 bytes");

#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TAPMATRIX_PIE 1
#define TAPMATRIX_FIXED 1
/* fixed_pie.S: out[i] = sum over taps t, k=0..3 of coef[4t+k] * src[t][i-k],
   i = 0 .. 8*nchunks-1, the accumulator shifted by `shift` */
void tapmatrix_fixed_sum(const int16_t *const *src, const int16_t *coef, int ntaps,
                       int16_t *out, int nchunks, int shift);
/* moving taps, n = 8*nchunks samples (<= 64) */
void tapmatrix_fixed_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks);

#elif defined(ESP_PLATFORM)
#define TAPMATRIX_PIE 0
#define TAPMATRIX_FIXED 0             /* other ESP32s: no fixed engine */

#else
#define TAPMATRIX_PIE 0
#define TAPMATRIX_FIXED 1             /* host: fixed with the C kernels below, for tests */
/* The PIE arithmetic in C: vmul = (a*b) >> 15, saturated; add/sub saturated;
   the accumulator 64-bit. */
static inline int16_t sat16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : v; }
static inline int16_t vmul(int16_t a, int16_t b) { return sat16(((int32_t)a * b) >> 15); }
static void tapmatrix_fixed_sum(const int16_t *const *src, const int16_t *coef, int ntaps,
                              int16_t *out, int nchunks, int shift)
{
    for (int i = 0; i < 8 * nchunks; i++) {
        int64_t acc = 0;
        for (int t = 0; t < ntaps; t++)
            for (int k = 0; k < 4; k++) acc += (int32_t)coef[4 * t + k] * src[t][i - k];
        out[i] = sat16((int32_t)(acc >> shift));
    }
}
static void tapmatrix_fixed_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks)
{
    for (int c = 0; c < nchunks; c++)
        for (int i = 0; i < 8; i++) {
            int64_t acc = 0;
            for (int t = 0; t < ntaps; t++) {
                const t_mvtap *m = &taps[t];
                const int16_t tt = sat16(m->a[c] + m->v[i]);
                const int16_t q = tt & 0x7fff, u = sat16(32767 - q);
                const int16_t P = vmul(q, u), Q = vmul(P, q);
                const int16_t Qh = vmul(Q, 16384), Ph = vmul(P, 16384);
                int16_t hb = sat16(q + Qh), hc = sat16(sat16(u + Ph) - Qh);
                int16_t ha = vmul(sat16(P + Q), -5461), hd = vmul(sat16(sat16(P + P) - Q), -5461);
                const int16_t g = sat16(m->g0[c] + m->rg[i]);
                ha = vmul(ha, g); hb = vmul(hb, g); hc = vmul(hc, g); hd = vmul(hd, g);
                const int16_t L = m->mode[c] ? (m->mode[c] == 1 ? -1 : 0) : (tt < 0 ? -1 : 0);
                const int16_t cs[5] = { hd & L, (hc & L) | (hd & ~L), (hb & L) | (hc & ~L),
                                        (ha & L) | (hb & ~L), ha & ~L };
                for (int k = 0; k < 5; k++) acc += (int32_t)cs[k] * m->base[c][i + k];
            }
            out[8 * c + i] = sat16((int32_t)(acc >> 15));
        }
}
#endif

/* a sample -> fixed, rounded, saturated (no lrintf: a library call) */
static inline int16_t to_fixed(t_sample v)
{
    v *= 32768;
    if (v >= 32767) return 32767;
    if (v <= -32768) return -32768;
    return (int16_t)(int32_t)(v + (v >= 0 ? TMK(0.5) : TMK(-0.5)));
}

/* x >= 0 samples -> 32.32, without floorf (a library call) */
static inline uint64_t fixpos(t_sample x)
{
    const int32_t hi = (int32_t)x;
    const t_sample r = (x - (t_sample)hi) * 65536;
    const int32_t mid = (int32_t)r;
    const int32_t lo = (int32_t)((r - (t_sample)mid) * 65536);
    return ((uint64_t)(uint32_t)hi << 32) | ((uint32_t)mid << 16) | (uint32_t)lo;
}

/* a moving tap's state in the fixed engine: the ramp's fixed parts, set in its
   first block */
typedef struct {
    int mv;                         /* set */
    int up;                         /* the read position runs faster than the input */
    uint64_t T, inc;                /* read position (32.32) of the next block, step */
    int32_t G, G8;                  /* gain (30 fraction bits) of the next block, step per 8 samples */
    int16_t v[8], rg[8];
} t_fixedtap;

/* A moving tap's first block: what stays the same for the ramp (delay and gain
   change by the same amount every block). |dd| < 1/7. */
static void fixed_ramp_start(t_fixedtap *t, int w, t_sample d0, t_sample dd, t_sample g0, t_sample gd)
{
    const int32_t ddf = (int32_t)(dd * TMK(2147483648.0));   /* the bits below 2^-31 don't matter */
    t->inc = (1ull << 32) - (uint64_t)((int64_t)ddf * 2);
    t->T = ((uint64_t)(uint32_t)w << 32) - fixpos(d0);
    t->up = ddf < 0;
    /* lane offsets in 1/32768 of a sample: the fraction moves by |dd| per sample
       (up) or by -dd; t = a + v stays within int16 */
    const uint32_t D = (uint32_t)(t->up ? -(int64_t)ddf : ddf) * 2;   /* |dd| * 2^32 */
    for (int i = 0; i < 8; i++) {
        const int16_t o = (int16_t)(((uint32_t)i * D) >> 17);
        t->v[i] = t->up ? o : -o;
    }
    const t_sample g8 = 8 * gd * TMK(1073741824.0);
    t->G = (int32_t)(g0 * TMK(1073741824.0) + TMK(0.5));
    t->G8 = (int32_t)(g8 + (g8 < 0 ? TMK(-0.5) : TMK(0.5)));
    for (int i = 0; i < 8; i++) t->rg[i] = (int16_t)(((int64_t)i * t->G8 / 8) >> 15);
    t->mv = 1;
}

/* every block of the ramp: the chunks' windows, fractions and gains, then on.
   qr: the input's fixed ring with its mirror (len, mask, mir as the ring's) */
static void fixed_ramp_block(t_fixedtap *t, t_mvtap *m, const int16_t *qr, int len, int mask, int mir, int n)
{
    memcpy(m->v, t->v, sizeof(m->v));
    memcpy(m->rg, t->rg, sizeof(m->rg));
    const int up = t->up;
    const uint32_t off = up ? 1 : 2;
    const int16_t v7 = t->v[7];
    const uint64_t inc8 = t->inc * 8;
    uint64_t T = t->T;
    int32_t G = t->G;
    for (int c = 0; c < (n >> 3); c++, T += inc8, G += t->G8) {
        const uint32_t J = (uint32_t)(T >> 32);
        /* window point 0 of lane 0; a chunk reads at most 12 samples on: near the
           ring's end from the mirror in front, which runs on into the start */
        const int X = (int)((J - off) & mask);
        m->base[c] = qr + mir + (X + 12 > len ? X - len : X);
        const int32_t g = G >> 15;
        m->g0[c] = g > 32767 ? 32767 : g < 0 ? 0 : g;
        const int32_t F = (int32_t)((uint32_t)T >> 17);
        const int32_t a = up ? F - 32768 : F;      /* lane 0: t < 0 (up) / t >= 0 */
        m->a[c] = (int16_t)a;
        /* the lanes are monotonic: a step in the chunk iff lane 7 is past it */
        m->mode[c] = up ? (a + v7 >= 0 ? 0 : 1) : (a + v7 < 0 ? 0 : 2);
    }
    t->T = T;
    t->G = G;
}

/* a silent block of the ramp (gain 0 throughout: G == G8 == 0), not computed: only
   on to the next block, exactly as fixed_ramp_block() would go (n/8 chunks) */
static inline void fixed_ramp_skip(t_fixedtap *t, int n)
{
    t->T += t->inc * 8 * (uint64_t)(n >> 3);
    t->G += t->G8 * (n >> 3);
}

/* n samples -> fixed, rounded, saturated */
static inline void fixed_from(const t_sample *r, int16_t *q, int n)
{
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)(r[i] * 32768 + (r[i] >= 0 ? TMK(0.5) : TMK(-0.5)));
        v = v > 32767 ? 32767 : v;
        q[i] = (int16_t)(v < -32768 ? -32768 : v);
    }
}

#endif
