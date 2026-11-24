/* The SIMD float kernels (arch/neon.h, arch/avx2.h) against the scalar ones
 * (arch/scalar.h): static taps (fir4) and moving taps (moving) with random
 * delays, gains and ramps, also near the ends of the ring (the mirror) and
 * near the kernels' limits on |dd|. No Pd needed.
 *   cc -O2 -I<pd>/src -o kerneltest kerneltest.c -lm && ./kerneltest
 * Prints the largest difference relative to the output's peak, per kernel. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "m_pd.h"
#include "../arch/scalar.h"
#include "../arch/neon.h"
#include "../arch/avx2.h"

#define LEN 1024                    /* ring length (power of 2) */
#define N 64                        /* block */
#define MIR (N + 3)                 /* mirror in front of the ring, as tapmatrix~ */

static t_sample buf[MIR + LEN];

static double urand(double lo, double hi) { return lo + (hi - lo) * (rand() / (double)RAND_MAX); }

/* the mirror: ring[-MIR .. -1] = ring[LEN-MIR .. LEN-1] */
static void fill(void)
{
    t_sample *ring = buf + MIR;
    for (int i = 0; i < LEN; i++) ring[i] = (t_sample)urand(-1, 1);
    memcpy(buf, ring + LEN - MIR, MIR * sizeof(t_sample));
}

static double maxdiff(const t_sample *a, const t_sample *b, int n, double *peak)
{
    double m = 0;
    for (int i = 0; i < n; i++) {
        m = fmax(m, fabs((double)a[i] - b[i]));
        *peak = fmax(*peak, fabs((double)a[i]));
    }
    return m;
}

int main(void)
{
    const char *name = TAPMATRIX_AVX2 ? "avx2" : TAPMATRIX_NEON ? "neon" : "none";
    tm_avx2_init();
#if TAPMATRIX_AVX2
    if (!tm_avx2) { printf("SKIP avx2: this CPU has no AVX2/FMA\n"); return 0; }
#endif
    if (!TAPMATRIX_AVX2 && !TAPMATRIX_NEON) { printf("SKIP: no SIMD kernels in this build\n"); return 0; }
    srand(1);
    fill();
    t_sample *ring = buf + MIR;
    double es = 0, em = 0, ps = 0, pm = 0;
    int nm = 0;
    for (int trial = 0; trial < 20000; trial++) {
        t_sample ref[N], simd[N];
        /* static: a block's reads may cross the ring's end (the mirror) */
        const int a0 = rand() % LEN, a1 = a0 + N > LEN ? a0 - LEN : a0;
        t_sample h[4];
        coeffs((t_sample)urand(0, 1), (t_sample)urand(-2, 2), h);
        for (int i = 0; i < N; i++) ref[i] = simd[i] = (t_sample)urand(-1, 1);
        fir4(ring + a1, h, ref, N);
#if TAPMATRIX_AVX2
        fir4_avx2(ring + a1, h, simd, N);
#elif TAPMATRIX_NEON
        fir4_neon(ring + a1, h, simd, N);
#endif
        es = fmax(es, maxdiff(ref, simd, N, &ps));
        /* moving: the block's write position w, a delay that keeps the reads
           inside the ring's history, ramps up to the kernel's limit */
        const int w = (rand() % (LEN / N)) * N;
        const double lim = TAPMATRIX_AVX2 ? 1.0 / 7 : 1.0 / 3;
        const double dd = trial % 10 == 0 ? 0 : trial % 10 == 1 ? lim * 0.999 : trial % 10 == 2 ? -lim * 0.999
            : urand(-lim, lim) * 0.999;
        const double d0 = urand(1 + fmax(0, -dd * N), LEN - 2 * N - 8);
        const t_sample g0 = (t_sample)urand(-1, 1), gd = (t_sample)urand(-0.02, 0.02);
        for (int i = 0; i < N; i++) ref[i] = simd[i] = 0;
        moving(ring, LEN - 1, w, (t_sample)d0, (t_sample)dd, g0, gd, ref, N);
#if TAPMATRIX_AVX2
        moving_avx2(ring, LEN - 1, LEN, w, (t_sample)d0, (t_sample)dd, g0, gd, simd, N);
#elif TAPMATRIX_NEON
        moving_neon(ring, LEN - 1, LEN, w, (t_sample)d0, (t_sample)dd, g0, gd, simd, N);
#endif
        em = fmax(em, maxdiff(ref, simd, N, &pm));
        nm++;
    }
    const int ok = es / ps < 1e-6 && em / pm < 1e-5;
    printf("%s %s: static max |simd - scalar| %.2e of peak %.2f, moving %.2e of peak %.2f (%d ramps)\n",
           ok ? "PASS" : "FAIL", name, es, ps, em, pm, nm);
    return !ok;
}
