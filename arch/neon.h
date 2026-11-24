/* tapmatrix~: the float kernels with NEON (armv7 with -mfpu=neon, arm64),
 * 4 output samples per vector. The same results as fir4() and moving() in
 * scalar.h up to float rounding. Single precision only: with PD_FLOATSIZE=64
 * TAPMATRIX_NEON is 0 and the scalar kernels run.
 *   fir4_neon(), fir2x2_neon(), moving_neon()   n a multiple of 4; moving_neon: |dd| < 1/3
 *   moving_os_neon()                            -oversample; |dd| < 1/6
 *
 * How the moving kernel works (arch/avx2.h does the same with 8 lanes, the
 * fixed engine's PIE kernel too): lane i computes output sample i. Its read
 * position is t_i = t_0 + i * (1 - dd) samples (32.32 fixed point, as in
 * moving()). Within 4 samples the integer part of t_i - i changes by at most
 * one when 3 |dd| < 1, so every lane finds its 4 interpolation points in a
 * window of 5 consecutive samples starting at ring[X + i]: points 0..3 (lanes
 * "L") or points 1..4 (the others). Five loads at X, X+1, .., X+4 give all
 * four windows at once (lane i of load k is ring[X + i + k]); a per-lane mask
 * picks points 0..3 or 1..4. Which lanes are which: the lane's fraction
 * f_i = f_0 + i * step (32-bit, wrapping) wrapped around past f_0 exactly
 * when the lane crossed a sample boundary. The limit on |dd| follows from the
 * window: a faster ramp would need more than 5 points per lane; such a block
 * takes the scalar kernel.
 *
 * Reading the intrinsics (all work on the 4 lanes at once, lane by lane):
 *   float32x4_t, uint32x4_t   4 floats, 4 unsigned int32 (one 128-bit register)
 *   vld1q_f32(p)        load p[0..3] (any alignment); vst1q_f32(p, v) store
 *   vld1q_u32(p)        load 4 unsigned ints
 *   vdupq_n_f32(v), vdupq_n_u32(v)   v in all 4 lanes
 *   vaddq_f32 / vsubq_f32 / vmulq_f32   a + b, a - b, a * b; vnegq_f32(a): -a
 *   vmlaq_f32(a, b, c)  a + b*c;  vmlsq_f32(a, b, c): a - b*c
 *   vaddq_u32(a, b)     integer add (wrapping)
 *   vcgtq_u32(a, b), vcltq_u32(a, b)   all ones where a > b (a < b), unsigned
 *   vmvnq_u32(a)        bitwise not
 *   vbslq_f32(m, x, y)  bitwise select: x where m's bits are set, else y
 *   vshrq_n_u32(v, 8)   v >> 8;  vcvtq_f32_u32(v): unsigned int -> float
 */
#ifndef TAPMATRIX_NEON_H
#define TAPMATRIX_NEON_H

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && PD_FLOATSIZE != 64
#include <arm_neon.h>
#define TAPMATRIX_NEON 1

/* one static tap: out[i] += h0 x[i] + h1 x[i-1] + h2 x[i-2] + h3 x[i-3], 4 at a time.
   x points at the tap's newest input sample for output 0 (in the ring); the
   loads at x+i-1 .. x+i-3 are the same 4 samples, one, two, three samples older. */
static void fir4_neon(const float *x, const float h[4], float *out, int n)
{
    /* the 4 coefficients (interpolation and gain), each copied to all lanes */
    const float32x4_t h0 = vdupq_n_f32(h[0]), h1 = vdupq_n_f32(h[1]),
        h2 = vdupq_n_f32(h[2]), h3 = vdupq_n_f32(h[3]);
    for (int i = 0; i < n; i += 4) {
        float32x4_t acc = vld1q_f32(out + i);               /* what the other taps added */
        acc = vmlaq_f32(acc, h0, vld1q_f32(x + i));         /* += h0 * x[i .. i+3] */
        acc = vmlaq_f32(acc, h1, vld1q_f32(x + i - 1));     /* += h1 * x[i-1 .. i+2] */
        acc = vmlaq_f32(acc, h2, vld1q_f32(x + i - 2));     /* += h2 * x[i-2 .. i+1] */
        acc = vmlaq_f32(acc, h3, vld1q_f32(x + i - 3));     /* += h3 * x[i-3 .. i] */
        vst1q_f32(out + i, acc);                            /* back into the output */
    }
}

/* a static tap on the 2x-oversampled line (scalar.h: fir2x2), 4 at a time:
   out[i] += h0 p[i] + h1 q[i] + h2 p[i-1] + h3 q[i-1]. The tap's 4 points
   alternate between the two rings: p has the newest (a) and the third (c),
   q the second (b) and the oldest (d), each as 4 consecutive values. */
static void fir2x2_neon(const float *p, const float *q, const float h[4], float *out, int n)
{
    const float32x4_t h0 = vdupq_n_f32(h[0]), h1 = vdupq_n_f32(h[1]),
        h2 = vdupq_n_f32(h[2]), h3 = vdupq_n_f32(h[3]);
    for (int i = 0; i < n; i += 4) {
        float32x4_t acc = vld1q_f32(out + i);
        acc = vmlaq_f32(acc, h0, vld1q_f32(p + i));         /* point a */
        acc = vmlaq_f32(acc, h1, vld1q_f32(q + i));         /* point b */
        acc = vmlaq_f32(acc, h2, vld1q_f32(p + i - 1));     /* point c */
        acc = vmlaq_f32(acc, h3, vld1q_f32(q + i - 1));     /* point d */
        vst1q_f32(out + i, acc);
    }
}

/* A moving tap, 4 samples per vector. Lane i of a chunk reads a 5-point
   window ring[X + i .. X + i + 4]; lanes in L use points 0..3 (d c b a), the
   others 1..4: within 4 samples the read position steps by at most one
   sample (|dd| < 1/3). ring: logical 0 of the ring (the mirror in front of it),
   len its length, w the write position of the block. */
static void moving_neon(const float *ring, int mask, int len, int w, float d0, float dd,
                        float g0, float gd, float *out, int n)
{
    /* --- once per block ---
       The read position of output sample 0, t = w - d0, as a 32.32 number
       (integer part: ring index of the point c; fraction: the way to b), and
       its step per output sample, 1 - dd (one sample on, minus the delay's
       change). In 32.32 the steps add up exactly. */
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd), inc4 = inc * 4;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    const int up = dd < 0;          /* the read position runs faster than the input */
    const uint32_t off = up ? 1 : 2;        /* window start X = J - off */
    /* per lane i: the fraction's offset i * step from lane 0 (wraps mod 2^32,
       which is the point), and the gain's offset i * gd */
    const uint32_t lo[4] = { 0, step, 2 * step, 3 * step };
    const uint32x4_t offs = vld1q_u32(lo);
    const float gl[4] = { 0, gd, 2 * gd, 3 * gd };
    const float32x4_t rg = vld1q_f32(gl), sixth = vdupq_n_f32(1.f / 6.f),
        one = vdupq_n_f32(1.f), two = vdupq_n_f32(2.f), three = vdupq_n_f32(3.f),
        tofrac = vdupq_n_f32(1.f / 16777216.f);
    float g = g0;
    /* --- per chunk of 4 output samples --- */
    for (int i = 0; i < n; i += 4, T += inc4, g += 4 * gd) {
        /* lane 0: integer part J (ring index of its point c), fraction F0 */
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T;
        /* the window starts at X = J - off: lane i's 5 candidates are ring[X+i .. X+i+4] */
        int X = (int)((J - off) & mask);
        if (X + 8 > len) X -= len;          /* near the end: the mirror in front */
        const float *p = ring + X;
        /* each lane's fraction f_i = F0 + i*step (32 bits, wrapping) */
        const uint32x4_t f = vaddq_u32(vdupq_n_u32(F0), offs);
        /* lanes past a step: the fraction wrapped (falling: it grew; rising: it shrank);
           a wrap means the lane's read position crossed one more (or one fewer)
           sample boundary than its lane index */
        const uint32x4_t wrapped = up ? vcltq_u32(f, vdupq_n_u32(F0)) : vcgtq_u32(f, vdupq_n_u32(F0));
        /* L: the lanes that use points 0..3 (behind: wrapped; ahead: not wrapped) */
        const uint32x4_t L = up ? vmvnq_u32(wrapped) : wrapped;
        /* the 5 window points of all lanes: load k, lane i = ring[X + i + k] */
        const float32x4_t w0 = vld1q_f32(p), w1 = vld1q_f32(p + 1), w2 = vld1q_f32(p + 2),
            w3 = vld1q_f32(p + 3), w4 = vld1q_f32(p + 4);
        /* d (oldest) .. a (newest) of each lane: points 0..3 where L is set, else 1..4 */
        const float32x4_t d = vbslq_f32(L, w0, w1), c = vbslq_f32(L, w1, w2),
            b = vbslq_f32(L, w2, w3), a = vbslq_f32(L, w3, w4);
        /* fp: the fraction past c (24 bits, exact in float), frac = 1 - fp: past b */
        const float32x4_t fp = vmulq_f32(vcvtq_f32_u32(vshrq_n_u32(f, 8)), tofrac);
        const float32x4_t frac = vsubq_f32(one, fp), cb = vsubq_f32(c, b);
        /* y = b + frac*(cb - (1/6) fp ((d - a - 3cb) frac + (d + 2a - 3b))), as delread4~,
           evaluated from the inside out: */
        float32x4_t t1 = vmlaq_f32(vsubq_f32(d, a), vnegq_f32(three), cb);  /* d - a - 3cb */
        float32x4_t t2 = vmlsq_f32(vmlaq_f32(d, two, a), three, b);         /* d + 2a - 3b */
        float32x4_t t = vmlaq_f32(t2, t1, frac);                            /* t2 + t1 frac */
        t = vmlsq_f32(cb, vmulq_f32(sixth, fp), t);                         /* cb - fp/6 t */
        const float32x4_t y = vmlaq_f32(b, frac, t);                        /* b + frac t */
        const float32x4_t gv = vaddq_f32(vdupq_n_f32(g), rg);               /* the gain ramp per lane */
        vst1q_f32(out + i, vmlaq_f32(vld1q_f32(out + i), y, gv));           /* out += y g */
    }
}
/* a moving tap on the 2x line (scalar.h: moving_os), 4 samples per vector: as
   moving_neon() with the read position in u's samples; window point k of all
   lanes is 4 consecutive values of one ring (x or the half-samples o).
   Needs |2 dd| < 1/3. x, o: logical 0 of the two rings. */
static void moving_os_neon(const float *x, const float *o, int mask, int len, int K, int w,
                           float d0, float dd, float g0, float gd, float *out, int n)
{
    /* as moving_neon(), in u's samples: position 2(w - d0), step 2(1 - dd) per output sample */
    uint64_t T = ((uint64_t)(uint32_t)(2 * w) << 32) - (uint64_t)fix32(2 * d0);
    const uint64_t inc = (2ull << 32) - (uint64_t)fix32(2 * dd), inc4 = inc * 4;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    const int up = dd < 0;                  /* lanes that wrapped are one u sample on (else back) */
    const uint32_t off = up ? 1 : 2;        /* window start X = J - off (in u) */
    const uint32_t lo[4] = { 0, step, 2 * step, 3 * step };
    const uint32x4_t offs = vld1q_u32(lo);
    const float gl[4] = { 0, gd, 2 * gd, 3 * gd };
    const float32x4_t rg = vld1q_f32(gl), sixth = vdupq_n_f32(1.f / 6.f),
        one = vdupq_n_f32(1.f), two = vdupq_n_f32(2.f), three = vdupq_n_f32(3.f),
        tofrac = vdupq_n_f32(1.f / 16777216.f);
    float g = g0;
    for (int i = 0; i < n; i += 4, T += inc4, g += 4 * gd) {
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T, X = J - off;
        float32x4_t wk[5];
        for (int k = 0; k < 5; k++) {           /* window point k of all lanes: one ring, 4 in a row */
            const uint32_t m = X + k;           /* lane 0's u index; lane i's is m + 2i */
            /* u[m + 2i]: even m -> x[m/2 + i]; odd m -> o[(m-1)/2 + K + i] (o lags by K) */
            const float *r = (m & 1) ? o : x;
            int s = (int)(((m >> 1) + ((m & 1) ? (uint32_t)K : 0)) & (uint32_t)mask);
            if (s + 4 > len) s -= len;          /* near the end: the mirror in front */
            wk[k] = vld1q_f32(r + s);
        }
        /* from here on as in moving_neon(): which lanes use points 0..3, then
           the interpolation and the gain */
        const uint32x4_t f = vaddq_u32(vdupq_n_u32(F0), offs);              /* each lane's fraction */
        const uint32x4_t wrapped = up ? vcltq_u32(f, vdupq_n_u32(F0)) : vcgtq_u32(f, vdupq_n_u32(F0));
        const uint32x4_t L = up ? vmvnq_u32(wrapped) : wrapped;
        const float32x4_t d = vbslq_f32(L, wk[0], wk[1]), c = vbslq_f32(L, wk[1], wk[2]),
            b = vbslq_f32(L, wk[2], wk[3]), a = vbslq_f32(L, wk[3], wk[4]);
        const float32x4_t fp = vmulq_f32(vcvtq_f32_u32(vshrq_n_u32(f, 8)), tofrac);
        const float32x4_t frac = vsubq_f32(one, fp), cb = vsubq_f32(c, b);
        float32x4_t t1 = vmlaq_f32(vsubq_f32(d, a), vnegq_f32(three), cb);  /* d - a - 3cb */
        float32x4_t t2 = vmlsq_f32(vmlaq_f32(d, two, a), three, b);         /* d + 2a - 3b */
        float32x4_t t = vmlaq_f32(t2, t1, frac);
        t = vmlsq_f32(cb, vmulq_f32(sixth, fp), t);
        const float32x4_t y = vmlaq_f32(b, frac, t);                        /* the interpolated value */
        const float32x4_t gv = vaddq_f32(vdupq_n_f32(g), rg);
        vst1q_f32(out + i, vmlaq_f32(vld1q_f32(out + i), y, gv));           /* out += y g */
    }
}
#else
#define TAPMATRIX_NEON 0
#endif

#endif
