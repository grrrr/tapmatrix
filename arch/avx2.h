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
 * Reading the intrinsics: __m256 holds 8 floats, __m256i 8 int32. _mm256_
 * loadu_ps / storeu_ps load / store 8 floats (any alignment), set1 copies a
 * value to all lanes; fmadd(a, b, c) = a*b + c, fnmadd(a, b, c) = c - a*b
 * (one rounding); blendv_ps(x, y, m): y where m's sign bit is set, else x;
 * cmpgt_epi32: signed compare, all ones where true. */
#ifndef TAPMATRIX_AVX2_H
#define TAPMATRIX_AVX2_H

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    PD_FLOATSIZE != 64 && !defined(TAPMATRIX_NO_AVX2)
#include <immintrin.h>
#define TAPMATRIX_AVX2 1
#define TM_AVX2 __attribute__((target("avx2,fma")))

static int tm_avx2;

static void tm_avx2_init(void)
{
    __builtin_cpu_init();
    tm_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

/* one static tap: out[i] += h0 x[i] + h1 x[i-1] + h2 x[i-2] + h3 x[i-3], 8 at a time */
TM_AVX2 static void fir4_avx2(const float *x, const float h[4], float *out, int n)
{
    const __m256 h0 = _mm256_set1_ps(h[0]), h1 = _mm256_set1_ps(h[1]),
        h2 = _mm256_set1_ps(h[2]), h3 = _mm256_set1_ps(h[3]);
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(out + i);
        acc = _mm256_fmadd_ps(h0, _mm256_loadu_ps(x + i), acc);
        acc = _mm256_fmadd_ps(h1, _mm256_loadu_ps(x + i - 1), acc);
        acc = _mm256_fmadd_ps(h2, _mm256_loadu_ps(x + i - 2), acc);
        acc = _mm256_fmadd_ps(h3, _mm256_loadu_ps(x + i - 3), acc);
        _mm256_storeu_ps(out + i, acc);
    }
}

/* a static tap on the 2x-oversampled line (scalar.h: fir2x2), 8 at a time */
TM_AVX2 static void fir2x2_avx2(const float *p, const float *q, const float h[4], float *out, int n)
{
    const __m256 h0 = _mm256_set1_ps(h[0]), h1 = _mm256_set1_ps(h[1]),
        h2 = _mm256_set1_ps(h[2]), h3 = _mm256_set1_ps(h[3]);
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_loadu_ps(out + i);
        acc = _mm256_fmadd_ps(h0, _mm256_loadu_ps(p + i), acc);
        acc = _mm256_fmadd_ps(h1, _mm256_loadu_ps(q + i), acc);
        acc = _mm256_fmadd_ps(h2, _mm256_loadu_ps(p + i - 1), acc);
        acc = _mm256_fmadd_ps(h3, _mm256_loadu_ps(q + i - 1), acc);
        _mm256_storeu_ps(out + i, acc);
    }
}

/* unsigned 32-bit a > b per lane (AVX2 compares signed only: flip the sign bits) */
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
    /* the read position of sample 0 and its step per sample, 32.32 */
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd), inc8 = inc * 8;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    /* dd < 0: the read position runs faster than the input, lanes that wrapped
       are one sample further on; dd >= 0: one sample behind */
    const int up = dd < 0;
    const uint32_t off = up ? 1 : 2;        /* window start X = j_0 - off */
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i offs = _mm256_mullo_epi32(lane, _mm256_set1_epi32((int)step));
    const __m256 rg = _mm256_mul_ps(_mm256_cvtepi32_ps(lane), _mm256_set1_ps(gd)),
        sixth = _mm256_set1_ps(1.f / 6.f), one = _mm256_set1_ps(1.f), two = _mm256_set1_ps(2.f),
        three = _mm256_set1_ps(3.f), tofrac = _mm256_set1_ps(1.f / 16777216.f);
    float g = g0;
    for (int i = 0; i < n; i += 8, T += inc8, g += 8 * gd) {
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T;
        int X = (int)((J - off) & mask);
        if (X + 12 > len) X -= len;         /* the loads reach X + 11: near the end, the mirror */
        const float *p = ring + X;
        const __m256i f0 = _mm256_set1_epi32((int)F0);
        const __m256i f = _mm256_add_epi32(f0, offs);     /* each lane's fraction, 0.32 */
        /* wrapped: rising fractions (up) wrap to below f_0, falling ones to above */
        const __m256i wrapped = up ? tm_cmpgt_u32(f0, f) : tm_cmpgt_u32(f, f0);
        /* L: the lanes that use points 0..3 (behind: wrapped; ahead: not wrapped) */
        const __m256 L = _mm256_castsi256_ps(up ? _mm256_xor_si256(wrapped, _mm256_set1_epi32(-1)) : wrapped);
        const __m256 w0 = _mm256_loadu_ps(p), w1 = _mm256_loadu_ps(p + 1), w2 = _mm256_loadu_ps(p + 2),
            w3 = _mm256_loadu_ps(p + 3), w4 = _mm256_loadu_ps(p + 4);
        /* d (oldest) .. a (newest) of each lane: blendv takes its second operand where L is set */
        const __m256 d = _mm256_blendv_ps(w1, w0, L), c = _mm256_blendv_ps(w2, w1, L),
            b = _mm256_blendv_ps(w3, w2, L), a = _mm256_blendv_ps(w4, w3, L);
        /* fp: the fraction past c (24 bits, exact in float), frac = 1 - fp: past b */
        const __m256 fp = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(f, 8)), tofrac);
        const __m256 frac = _mm256_sub_ps(one, fp), cb = _mm256_sub_ps(c, b);
        /* y = b + frac*(cb - (1/6) fp ((d - a - 3cb) frac + (d + 2a - 3b))), as delread4~ */
        const __m256 t1 = _mm256_fnmadd_ps(three, cb, _mm256_sub_ps(d, a));
        const __m256 t2 = _mm256_fnmadd_ps(three, b, _mm256_fmadd_ps(two, a, d));
        __m256 t = _mm256_fmadd_ps(t1, frac, t2);
        t = _mm256_fnmadd_ps(_mm256_mul_ps(sixth, fp), t, cb);
        const __m256 y = _mm256_fmadd_ps(frac, t, b);
        const __m256 gv = _mm256_add_ps(_mm256_set1_ps(g), rg);  /* the gain ramp per lane */
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(y, gv, _mm256_loadu_ps(out + i)));
    }
}
/* the halfband of -oversample (scalar.h: halfband), 8 outputs at a time */
TM_AVX2 static void halfband_avx2(const float *x, const float *c, int K, float *o, int n)
{
    for (int i = 0; i < n; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < K; j++)
            acc = _mm256_fmadd_ps(_mm256_set1_ps(c[j]),
                _mm256_add_ps(_mm256_loadu_ps(x + i - K - j), _mm256_loadu_ps(x + i - K + 1 + j)), acc);
        _mm256_storeu_ps(o + i, acc);
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
    uint64_t T = ((uint64_t)(uint32_t)(2 * w) << 32) - (uint64_t)fix32(2 * d0);
    const uint64_t inc = (2ull << 32) - (uint64_t)fix32(2 * dd), inc8 = inc * 8;
    const uint32_t step = (uint32_t)inc;
    const int up = dd < 0;
    const uint32_t off = up ? 1 : 2;
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
            const uint32_t m = X + k;
            const float *r = (m & 1) ? o : x;
            int s = (int)(((m >> 1) + ((m & 1) ? (uint32_t)K : 0)) & (uint32_t)mask);
            if (s + 8 > len) s -= len;          /* near the end: the mirror in front */
            wk[k] = _mm256_loadu_ps(r + s);
        }
        const __m256i f0 = _mm256_set1_epi32((int)F0);
        const __m256i f = _mm256_add_epi32(f0, offs);
        const __m256i wrapped = up ? tm_cmpgt_u32(f0, f) : tm_cmpgt_u32(f, f0);
        const __m256 L = _mm256_castsi256_ps(up ? _mm256_xor_si256(wrapped, _mm256_set1_epi32(-1)) : wrapped);
        const __m256 d = _mm256_blendv_ps(wk[1], wk[0], L), c = _mm256_blendv_ps(wk[2], wk[1], L),
            b = _mm256_blendv_ps(wk[3], wk[2], L), a = _mm256_blendv_ps(wk[4], wk[3], L);
        const __m256 fp = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(f, 8)), tofrac);
        const __m256 frac = _mm256_sub_ps(one, fp), cb = _mm256_sub_ps(c, b);
        const __m256 t1 = _mm256_fnmadd_ps(three, cb, _mm256_sub_ps(d, a));
        const __m256 t2 = _mm256_fnmadd_ps(three, b, _mm256_fmadd_ps(two, a, d));
        __m256 t = _mm256_fmadd_ps(t1, frac, t2);
        t = _mm256_fnmadd_ps(_mm256_mul_ps(sixth, fp), t, cb);
        const __m256 y = _mm256_fmadd_ps(frac, t, b);
        const __m256 gv = _mm256_add_ps(_mm256_set1_ps(g), rg);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(y, gv, _mm256_loadu_ps(out + i)));
    }
}
#else
#define TAPMATRIX_AVX2 0
#define tm_avx2 0
static inline void tm_avx2_init(void) {}
#endif

#endif
