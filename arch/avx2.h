/* tapmatrix~: the float kernels with AVX2 + FMA (x86_64, i386), 8 output
 * samples per vector. The same results as fir4() and moving() in scalar.h up
 * to float rounding (FMA rounds once where the scalar code rounds twice).
 * Single precision only: with PD_FLOATSIZE=64 TAPMATRIX_AVX2 is 0.
 *   fir4_avx2(), fir2x2_avx2(), moving_avx2()   n a multiple of 8; moving_avx2: |dd| < 1/7
 *   halfband_avx2(), moving_os_avx2()           -oversample; moving_os_avx2: |dd| < 1/14
 *   tm_avx2                      1 if the CPU has AVX2 and FMA (tm_avx2_init())
 *
 * The binaries are built for a baseline x86 (pd-lib-builder: no -mavx2), so
 * these functions are compiled for AVX2 one by one (target attribute) and
 * chosen at run time: the same binary runs on CPUs without AVX2.
 *
 * How the moving kernel works (moving_neon() in neon.h does the same with 4
 * lanes): lane i of a chunk computes output sample i. Its read position is
 * t_i = t_0 + i * (1 - dd) in samples (32.32 fixed point, as in moving()).
 * Within 8 samples the integer part of t_i - i changes by at most one when
 * 7 |dd| < 1, so every lane finds its 4 interpolation points in a window of
 * 5 consecutive samples starting at ring[X + i]: points 0..3 (lanes "L") or
 * points 1..4 (the others). Five unaligned loads at X, X+1, .., X+4 give all
 * eight windows at once (lane i of load k is ring[X + i + k]); a per-lane
 * mask picks points 0..3 or 1..4. Which lanes are which: the lane's fraction
 * f_i = f_0 + i * step (32-bit, wrapping) wrapped around past f_0 exactly
 * when the lane crossed a sample boundary. The limit on |dd| follows from the
 * window: a faster ramp would need more than 5 points per lane; such a block
 * takes the scalar kernel.
 *
 * Reading the intrinsics (all work on the 8 lanes at once, lane by lane):
 *   __m256              8 floats; __m256i: 8 int32 (the same 256-bit register)
 *   _mm256_loadu_ps(p)  load p[0..7] (any alignment); _mm256_storeu_ps(p, v) store
 *   _mm256_set1_ps(v)   v in all 8 lanes; _mm256_setr_epi32(a, .., h): lanes 0..7
 *   _mm256_setzero_ps() all lanes 0
 *   _mm256_add_ps / sub_ps / mul_ps          a + b, a - b, a * b
 *   _mm256_fmadd_ps(a, b, c)   a*b + c, rounded once ("fused multiply-add")
 *   _mm256_fnmadd_ps(a, b, c)  c - a*b, rounded once
 *   _mm256_add_epi32 / mullo_epi32 / xor_si256   integer add, multiply (low 32
 *                       bits), bitwise xor
 *   _mm256_srli_epi32(v, 8)    v >> 8 (unsigned)
 *   _mm256_cvtepi32_ps(v)      int32 -> float
 *   _mm256_cmpgt_epi32(a, b)   all ones where a > b (signed), else 0
 *   _mm256_castsi256_ps(v)     the same bits seen as floats (no conversion)
 *   _mm256_blendv_ps(x, y, m)  y where the sign bit of m is set, else x
 */
#ifndef TAPMATRIX_AVX2_H
#define TAPMATRIX_AVX2_H

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    PD_FLOATSIZE != 64 && !defined(TAPMATRIX_NO_AVX2)
#include <immintrin.h>
#define TAPMATRIX_AVX2 1
/* compile one function for AVX2 + FMA although the file is built for baseline x86 */
#define TM_AVX2 __attribute__((target("avx2,fma")))

static int tm_avx2;                 /* this CPU can run the functions below */

/* at class setup: ask the CPU (CPUID) once */
static void tm_avx2_init(void)
{
    __builtin_cpu_init();
    tm_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

/* one static tap: out[i] += h0 x[i] + h1 x[i-1] + h2 x[i-2] + h3 x[i-3], 8 at a time.
   x points at the tap's newest input sample for output 0 (in the ring); the
   loads at x+i-1 .. x+i-3 are the same 8 samples, one, two, three samples older. */
TM_AVX2 static void fir4_avx2(const float *x, const float h[4], float *out, int n)
{
    /* the 4 coefficients (interpolation and gain), each copied to all lanes */
    const __m256 h0 = _mm256_set1_ps(h[0]), h1 = _mm256_set1_ps(h[1]),
        h2 = _mm256_set1_ps(h[2]), h3 = _mm256_set1_ps(h[3]);
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(out + i);                      /* what the other taps added */
        acc = _mm256_fmadd_ps(h0, _mm256_loadu_ps(x + i), acc);     /* += h0 * x[i .. i+7] */
        acc = _mm256_fmadd_ps(h1, _mm256_loadu_ps(x + i - 1), acc); /* += h1 * x[i-1 .. i+6] */
        acc = _mm256_fmadd_ps(h2, _mm256_loadu_ps(x + i - 2), acc); /* += h2 * x[i-2 .. i+5] */
        acc = _mm256_fmadd_ps(h3, _mm256_loadu_ps(x + i - 3), acc); /* += h3 * x[i-3 .. i+4] */
        _mm256_storeu_ps(out + i, acc);                             /* back into the output */
    }
}

/* a static tap on the 2x-oversampled line (scalar.h: fir2x2), 8 at a time:
   out[i] += h0 p[i] + h1 q[i] + h2 p[i-1] + h3 q[i-1]. The tap's 4 points
   alternate between the two rings: p has the newest (a) and the third (c),
   q the second (b) and the oldest (d), each as 8 consecutive values. */
TM_AVX2 static void fir2x2_avx2(const float *p, const float *q, const float h[4], float *out, int n)
{
    const __m256 h0 = _mm256_set1_ps(h[0]), h1 = _mm256_set1_ps(h[1]),
        h2 = _mm256_set1_ps(h[2]), h3 = _mm256_set1_ps(h[3]);
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(out + i);
        acc = _mm256_fmadd_ps(h0, _mm256_loadu_ps(p + i), acc);     /* point a */
        acc = _mm256_fmadd_ps(h1, _mm256_loadu_ps(q + i), acc);     /* point b */
        acc = _mm256_fmadd_ps(h2, _mm256_loadu_ps(p + i - 1), acc); /* point c */
        acc = _mm256_fmadd_ps(h3, _mm256_loadu_ps(q + i - 1), acc); /* point d */
        _mm256_storeu_ps(out + i, acc);
    }
}

/* unsigned 32-bit a > b per lane. AVX2 compares signed only: flipping the
   sign bit of both maps 0 .. 2^32-1 in order onto -2^31 .. 2^31-1. */
TM_AVX2 static inline __m256i tm_cmpgt_u32(__m256i a, __m256i b)
{
    const __m256i s = _mm256_set1_epi32((int)0x80000000u);
    return _mm256_cmpgt_epi32(_mm256_xor_si256(a, s), _mm256_xor_si256(b, s));
}

/* a moving tap: delay d0 -> d0 + dd*n, gain g0 -> g0 + gd*n, 8 samples per
   vector (see the top of this file). ring: logical 0 of the ring (the mirror
   in front of it), len its length, w the write position of the block. */
TM_AVX2 static void moving_avx2(const float *ring, int mask, int len, int w, float d0, float dd,
                                float g0, float gd, float *out, int n)
{
    /* --- once per block ---
       The read position of output sample 0, t = w - d0, as a 32.32 number
       (integer part: ring index of the point c; fraction: the way to b), and
       its step per output sample, 1 - dd (one sample on, minus the delay's
       change). In 32.32 the steps add up exactly. */
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd), inc8 = inc * 8;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    /* dd < 0: the read position runs faster than the input, lanes that wrapped
       are one sample further on; dd >= 0: one sample behind */
    const int up = dd < 0;
    const uint32_t off = up ? 1 : 2;        /* window start X = j_0 - off */
    /* per lane i: the fraction's offset i * step from lane 0 (wraps mod 2^32,
       which is the point), and the gain's offset i * gd */
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i offs = _mm256_mullo_epi32(lane, _mm256_set1_epi32((int)step));
    const __m256 rg = _mm256_mul_ps(_mm256_cvtepi32_ps(lane), _mm256_set1_ps(gd)),
        sixth = _mm256_set1_ps(1.f / 6.f), one = _mm256_set1_ps(1.f), two = _mm256_set1_ps(2.f),
        three = _mm256_set1_ps(3.f), tofrac = _mm256_set1_ps(1.f / 16777216.f);
    float g = g0;
    /* --- per chunk of 8 output samples --- */
    for (int i = 0; i < n; i += 8, T += inc8, g += 8 * gd) {
        /* lane 0: integer part J (ring index of its point c), fraction F0 */
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T;
        /* the window starts at X = J - off: lane i's 5 candidates are ring[X+i .. X+i+4] */
        int X = (int)((J - off) & mask);
        if (X + 12 > len) X -= len;         /* the loads reach X + 11: near the end, the mirror */
        const float *p = ring + X;
        /* each lane's fraction f_i = F0 + i*step (32 bits, wrapping) */
        const __m256i f0 = _mm256_set1_epi32((int)F0);
        const __m256i f = _mm256_add_epi32(f0, offs);     /* each lane's fraction, 0.32 */
        /* wrapped: rising fractions (up) wrap to below f_0, falling ones to above;
           a wrap means the lane's read position crossed one more (or one fewer)
           sample boundary than its lane index */
        const __m256i wrapped = up ? tm_cmpgt_u32(f0, f) : tm_cmpgt_u32(f, f0);
        /* L: the lanes that use points 0..3 (behind: wrapped; ahead: not wrapped) */
        const __m256 L = _mm256_castsi256_ps(up ? _mm256_xor_si256(wrapped, _mm256_set1_epi32(-1)) : wrapped);
        /* the 5 window points of all lanes: load k, lane i = ring[X + i + k] */
        const __m256 w0 = _mm256_loadu_ps(p), w1 = _mm256_loadu_ps(p + 1), w2 = _mm256_loadu_ps(p + 2),
            w3 = _mm256_loadu_ps(p + 3), w4 = _mm256_loadu_ps(p + 4);
        /* d (oldest) .. a (newest) of each lane: blendv takes its second operand where L is set */
        const __m256 d = _mm256_blendv_ps(w1, w0, L), c = _mm256_blendv_ps(w2, w1, L),
            b = _mm256_blendv_ps(w3, w2, L), a = _mm256_blendv_ps(w4, w3, L);
        /* fp: the fraction past c (24 bits, exact in float), frac = 1 - fp: past b */
        const __m256 fp = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(f, 8)), tofrac);
        const __m256 frac = _mm256_sub_ps(one, fp), cb = _mm256_sub_ps(c, b);
        /* y = b + frac*(cb - (1/6) fp ((d - a - 3cb) frac + (d + 2a - 3b))), as delread4~,
           evaluated from the inside out: */
        const __m256 t1 = _mm256_fnmadd_ps(three, cb, _mm256_sub_ps(d, a));  /* d - a - 3cb */
        const __m256 t2 = _mm256_fnmadd_ps(three, b, _mm256_fmadd_ps(two, a, d)); /* d + 2a - 3b */
        __m256 t = _mm256_fmadd_ps(t1, frac, t2);                            /* t1 frac + t2 */
        t = _mm256_fnmadd_ps(_mm256_mul_ps(sixth, fp), t, cb);               /* cb - fp/6 t */
        const __m256 y = _mm256_fmadd_ps(frac, t, b);                        /* b + frac t */
        const __m256 gv = _mm256_add_ps(_mm256_set1_ps(g), rg);  /* the gain ramp per lane */
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(y, gv, _mm256_loadu_ps(out + i)));  /* out += y g */
    }
}

/* the halfband of -oversample (scalar.h: halfband), 8 outputs at a time:
   o[i] = sum over j of c[j] * (x[i-K-j] + x[i-K+1+j]), the half-sample value
   between x[i-K] and x[i-K+1]. The filter is symmetric: each coefficient
   multiplies the sum of its two inputs (K multiplies instead of 2K). */
TM_AVX2 static void halfband_avx2(const float *x, const float *c, int K, float *o, int n)
{
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < K; j++)         /* acc += c[j] * (x[i-K-j ..] + x[i-K+1+j ..]) */
            acc = _mm256_fmadd_ps(_mm256_set1_ps(c[j]),
                _mm256_add_ps(_mm256_loadu_ps(x + i - K - j), _mm256_loadu_ps(x + i - K + 1 + j)), acc);
        _mm256_storeu_ps(o + i, acc);       /* 8 half-sample values */
    }
}

/* a moving tap on the 2x line (scalar.h: moving_os), 8 samples per vector:
   as moving_avx2() with the read position in u's samples. Lane i reads its
   window u[X + 2i + k], k = 0..4; for one k all lanes read the same ring (x
   for even X + k, the half-samples o for odd), 8 consecutive values. Needs
   |2 dd| < 1/7. x, o: logical 0 of the two rings. */
TM_AVX2 static void moving_os_avx2(const float *x, const float *o, int mask, int len, int K, int w,
                                   float d0, float dd, float g0, float gd, float *out, int n)
{
    /* as moving_avx2(), in u's samples: position 2(w - d0), step 2(1 - dd) per output sample */
    uint64_t T = ((uint64_t)(uint32_t)(2 * w) << 32) - (uint64_t)fix32(2 * d0);
    const uint64_t inc = (2ull << 32) - (uint64_t)fix32(2 * dd), inc8 = inc * 8;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    const int up = dd < 0;                  /* lanes that wrapped are one u sample on (else back) */
    const uint32_t off = up ? 1 : 2;        /* window start X = J - off (in u) */
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i offs = _mm256_mullo_epi32(lane, _mm256_set1_epi32((int)step));
    const __m256 rg = _mm256_mul_ps(_mm256_cvtepi32_ps(lane), _mm256_set1_ps(gd)),
        sixth = _mm256_set1_ps(1.f / 6.f), one = _mm256_set1_ps(1.f), two = _mm256_set1_ps(2.f),
        three = _mm256_set1_ps(3.f), tofrac = _mm256_set1_ps(1.f / 16777216.f);
    float g = g0;
    for (int i = 0; i < n; i += 8, T += inc8, g += 8 * gd) {
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T, X = J - off;
        __m256 wk[5];
        for (int k = 0; k < 5; k++) {           /* window point k of all lanes: one ring, 8 in a row */
            const uint32_t m = X + k;           /* lane 0's u index; lane i's is m + 2i */
            /* u[m + 2i]: even m -> x[m/2 + i]; odd m -> o[(m-1)/2 + K + i] (o lags by K) */
            const float *r = (m & 1) ? o : x;
            int s = (int)(((m >> 1) + ((m & 1) ? (uint32_t)K : 0)) & (uint32_t)mask);
            if (s + 8 > len) s -= len;          /* near the end: the mirror in front */
            wk[k] = _mm256_loadu_ps(r + s);
        }
        /* from here on as in moving_avx2(): which lanes use points 0..3, then
           the interpolation and the gain */
        const __m256i f0 = _mm256_set1_epi32((int)F0);
        const __m256i f = _mm256_add_epi32(f0, offs);                        /* each lane's fraction */
        const __m256i wrapped = up ? tm_cmpgt_u32(f0, f) : tm_cmpgt_u32(f, f0);
        const __m256 L = _mm256_castsi256_ps(up ? _mm256_xor_si256(wrapped, _mm256_set1_epi32(-1)) : wrapped);
        const __m256 d = _mm256_blendv_ps(wk[1], wk[0], L), c = _mm256_blendv_ps(wk[2], wk[1], L),
            b = _mm256_blendv_ps(wk[3], wk[2], L), a = _mm256_blendv_ps(wk[4], wk[3], L);
        const __m256 fp = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(f, 8)), tofrac);
        const __m256 frac = _mm256_sub_ps(one, fp), cb = _mm256_sub_ps(c, b);
        const __m256 t1 = _mm256_fnmadd_ps(three, cb, _mm256_sub_ps(d, a));  /* d - a - 3cb */
        const __m256 t2 = _mm256_fnmadd_ps(three, b, _mm256_fmadd_ps(two, a, d)); /* d + 2a - 3b */
        __m256 t = _mm256_fmadd_ps(t1, frac, t2);
        t = _mm256_fnmadd_ps(_mm256_mul_ps(sixth, fp), t, cb);
        const __m256 y = _mm256_fmadd_ps(frac, t, b);                        /* the interpolated value */
        const __m256 gv = _mm256_add_ps(_mm256_set1_ps(g), rg);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(y, gv, _mm256_loadu_ps(out + i)));  /* out += y g */
    }
}
#else
#define TAPMATRIX_AVX2 0
#define tm_avx2 0
static inline void tm_avx2_init(void) {}
#endif

#endif
