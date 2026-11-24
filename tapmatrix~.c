/*
 * [tapmatrix~ [-float|-fixed] [-oversample] <ins> <outs> <maxdelay-ms> [ramp-ms]]
 *
 * A matrix of fractional delays: every input -> output cell is a tap with its
 * own delay and gain. Input: one multichannel signal (<ins> channels, missing
 * ones are silent). Output: one multichannel signal of <outs> channels, each
 * the sum of its taps. For WFS (sources x speakers), multi-tap delays, delay
 * panning and matrix mixing. Delay read with Pd's 4-point interpolation
 * (as delread4~), optionally on the delay line oversampled 2x (-oversample:
 * flat up to 20 kHz, float engine only, delays from TAPMATRIX_OS_K + 1 samples).
 * Design, measurements and decisions: README.md here.
 *
 * Messages (left inlet):
 *   tap <in> <out> <delay-ms> <gain> [ramp-ms]   set a cell; delay and gain ramp
 *                                               there over ramp-ms (default:
 *                                               the creation argument, 20 ms)
 *   row <in> <ramp-ms> <delay-array> <gain-array> [first-out]
 *                                               a whole row (input -> outputs first-out ..)
 *                                               from two arrays (delays in ms, gains)
 *   rowl <in> <ramp-ms> <delay> <gain> ...      the same from a list (outputs 0, 1, ...)
 *   jump <in> <xfade-ms> <delay-array> <gain-array> [first-out]
 *                                               dip-and-jump of the row: fade out, new
 *                                               delays at the turning point, fade in;
 *                                               rows/jumps during the fade-out only
 *                                               change the targets after the dip
 *   get <in> <delay-array> <gain-array>         the row's current delays (ms), gains
 *   ramp <ms>                                   default ramp time
 *   clear [in]                                  all gains (or one input's) to 0 at once
 *   engine float|fixed                            switch the engine (also the flag
 *                                               at creation; float is the default).
 *                                               fixed: 16-bit fixed point (ESP32-P4:
 *                                               PIE SIMD; host: the same in C, for tests)
 *   check <blocks>                              accuracy test, fixed only: for that many
 *                                               blocks also compute everything in
 *                                               float and compare; then the right
 *                                               outlet sends "check <snr-db> <peak-error>"
 *   stats <blocks>                              time measurement: averaged over that
 *                                               many blocks, the right outlet sends
 *                                               "stats <moving-fixed> <moving-float>
 *                                               <us-total> <us-inputs> <us-moving-prep>
 *                                               <us-moving-kernel> <us-static-kernel>"
 *   fixedshift <n>                                accumulator shift for PIE (test)
 *
 * How it computes (per block of n samples): the inputs go into ring buffers
 * (PSRAM, via getbytes; the last n+3 samples mirrored in front, so a block's
 * reads are contiguous, also across the ring's end). A tap that doesn't move
 * in this block is a 4-tap FIR with the gain folded into its coefficients,
 * added straight into the output. A moving tap ramps its delay and gain per
 * sample; its read position is a 32.32 integer. Ramps are counted in whole
 * blocks, each block's values computed from the ramp's ends.
 */
/* 32-bit Linux (armhf) with a 64-bit time_t default (Debian trixie and later):
   clock_gettime() would become __clock_gettime64@GLIBC_2.34 and the binary would
   not load on older systems. The timer for "stats" (arch/platform.h) is the only
   use of time types here, so take the old 32-bit ABI. Before any system header. */
#if defined(__linux__) && defined(__arm__)
#undef _TIME_BITS
#undef _FILE_OFFSET_BITS
#endif
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "m_pd.h"

/* the platform and the kernels (README.md, "Source layout") */
#include "arch/platform.h"
#include "arch/scalar.h"
#include "arch/neon.h"
#include "arch/avx2.h"
#include "arch/fixed.h"

static t_class *tapmatrix_class;
static t_sample tm_hb[TAPMATRIX_OS_K];      /* the halfband's taps (-oversample) */

typedef struct {
    t_sample d, g;                  /* now: delay in samples, gain */
    t_sample dfrom, dto, gfrom, gto;    /* ramp: from, to */
    int nb, left;                   /* ramp: blocks in all, left */
    t_sample inb;                   /* 1 / nb */
    t_fixedtap q;                     /* fixed engine, moving */
    int hvalid;                     /* h, qh hold the coefficients of d, g */
    t_sample h[4];
    int16_t qh[4];
    t_sample pd, pg;                /* jump: delay (samples) and gain after the dip */
} t_tap;

typedef struct _tapmatrix {
    t_object x_obj;
    t_float x_f;
    int x_ins, x_outs;
    t_float x_maxms, x_rampms;
    t_float x_sr;
    int x_n;                        /* block length the buffers are made for */
    int x_len, x_mask, x_mir;       /* ring: length (power of 2), mirror n+3 */
    int x_w;                        /* logical write position */
    t_sample *x_ring;               /* [ins][mir + len] */
    int x_os;                       /* -oversample */
    t_sample *x_oring;              /* [ins][mir + len], -oversample: the half-sample values */
    t_tap *x_tap;                   /* [outs][ins] */
    int *x_jleft;                   /* [ins] jump: blocks to the dip's turning point + 1, 0 = none */
    int *x_jxb;                     /* [ins] jump: fade length in blocks */
    int x_fixed;
    int16_t *x_qring;               /* [ins][mir + len], Q15 copy */
    int16_t *x_qcoef;               /* [ins * 4] scratch per output */
    const int16_t **x_qsrc;         /* [ins] */
    int16_t *x_qout, *x_qout2;      /* [n] each, 16-byte aligned: static, moving taps */
    void *x_qmem;
    t_mvtap *x_mv;                  /* [ins], moving taps of one output */
    void *x_mvmem;
    int x_qshift;
    int x_check;                    /* blocks still to compare */
    int x_stats;                    /* blocks still to count ("stats") */
    uint32_t x_st[8];               /* blocks, fixed moving, float moving, cycles:
                                       inputs, prep, moving kernel, static kernel, all */
    double x_es, x_ee;              /* signal and error energy */
    t_sample x_emax;
    t_sample *x_fcheck;                /* [n] float reference of the fixed part */
    t_outlet *x_info;
    t_clock *x_clock;               /* reports the check outside the DSP tick */
    t_clock *x_sclock;              /* and the stats */
} t_tapmatrix;

static void tapmatrix_freebuf(t_tapmatrix *x)
{
    if (x->x_ring) freebytes(x->x_ring, x->x_ins * (x->x_mir + x->x_len) * sizeof(t_sample));
    if (x->x_oring) freebytes(x->x_oring, x->x_ins * (x->x_mir + x->x_len) * sizeof(t_sample));
    x->x_oring = 0;
    if (x->x_qring) freebytes(x->x_qring, x->x_ins * (x->x_mir + x->x_len) * sizeof(int16_t));
    if (x->x_qmem) fastfree(x->x_qmem, 2 * x->x_n * sizeof(int16_t) + 32);
    if (x->x_mvmem) fastfree(x->x_mvmem, x->x_ins * sizeof(t_mvtap) + 16);
    x->x_mvmem = 0; x->x_mv = 0;
    if (x->x_fcheck) freebytes(x->x_fcheck, x->x_n * sizeof(t_sample));
    x->x_ring = 0; x->x_qring = 0; x->x_qmem = 0; x->x_qout = 0; x->x_fcheck = 0;
}

static void tapmatrix_allocq(t_tapmatrix *x)
{
    if (!x->x_fixed || x->x_qring || !x->x_n) return;
    x->x_qring = (int16_t *)getbytes(x->x_ins * (x->x_mir + x->x_len) * sizeof(int16_t));
    x->x_qmem = fastmem(2 * x->x_n * sizeof(int16_t) + 32);
    x->x_qout = (int16_t *)(((uintptr_t)x->x_qmem + 15) & ~(uintptr_t)15);
    x->x_qout2 = x->x_qout + ((x->x_n + 7) & ~7);
    x->x_mvmem = fastmem(x->x_ins * sizeof(t_mvtap) + 16);
    x->x_mv = (t_mvtap *)(((uintptr_t)x->x_mvmem + 15) & ~(uintptr_t)15);
    x->x_fcheck = (t_sample *)getbytes(x->x_n * sizeof(t_sample));
}

static t_sample clampd(t_tapmatrix *x, t_sample d)
{
    const t_sample lo = x->x_os ? TAPMATRIX_OS_K + 1 : 1, hi = x->x_len - x->x_n - 4;
    return (d < lo ? lo : d > hi ? hi : d);
}

/* (re)make the buffers for block length n */
static void tapmatrix_alloc(t_tapmatrix *x, int n, t_float sr)
{
    int need = (int)(x->x_maxms * 0.001f * sr) + n + 8 + (x->x_os ? 2 * TAPMATRIX_OS_K : 0), len = 64;
    while (len < need) len <<= 1;
    if (x->x_ring && x->x_n == n && x->x_len == len) return;
    tapmatrix_freebuf(x);
    x->x_n = n;
    x->x_len = len;
    x->x_mask = len - 1;
    x->x_mir = n + 3 + (x->x_os ? 2 * TAPMATRIX_OS_K : 0);    /* + the halfband's history */
    x->x_w = 0;
    for (int i = 0; i < x->x_ins * x->x_outs; i++) x->x_tap[i].q.mv = 0;
    x->x_ring = (t_sample *)getbytes(x->x_ins * (x->x_mir + len) * sizeof(t_sample));
    if (x->x_os) x->x_oring = (t_sample *)getbytes(x->x_ins * (x->x_mir + len) * sizeof(t_sample));
    tapmatrix_allocq(x);
    /* delays set before the first DSP run were not clamped (no block length yet) */
    for (int i = 0; i < x->x_ins * x->x_outs; i++) {
        t_tap *t = &x->x_tap[i];
        t->d = clampd(x, t->d), t->dto = clampd(x, t->dto), t->dfrom = clampd(x, t->dfrom);
        t->pd = clampd(x, t->pd);
        t->hvalid = 0;
    }
}

/* delay ms -> samples, as delread4~ computes it */
static t_sample ms2d(t_tapmatrix *x, t_float ms)
{
    t_sample d = ms * (t_sample)(x->x_sr * 0.001);
    return x->x_n ? clampd(x, d) : d;
}

/* ramp ms -> whole blocks (0 before the first DSP run: set at once) */
static int ms2blocks(t_tapmatrix *x, t_float ms)
{
    return (x->x_n && x->x_sr > 0) ? (int)(ms * 0.001f * x->x_sr / x->x_n + 0.5f) : 0;
}

/* one cell to delay d (samples) and gain g over `blocks` blocks; a silent tap
   (gain 0 now) jumps to its new delay and fades in */
static void tap_set(t_tap *t, t_sample d, t_sample g, int blocks)
{
    t->hvalid = 0;
    t->q.mv = 0;
    if (t->g == 0) t->d = d;
    t->dfrom = t->d; t->gfrom = t->g;
    t->dto = d; t->gto = g;
    if (blocks < 1) {
        t->d = d; t->g = g; t->left = 0;
    } else {
        t->nb = t->left = blocks;
        t->inb = TMK(1) / blocks;
    }
}

/* a cell of input `in` from a row message: while the row dips (jump), only
   the targets after the dip change */
static void row_set(t_tapmatrix *x, int in, int out, t_float ms, t_float g, int blocks)
{
    t_tap *t = &x->x_tap[out * x->x_ins + in];
    if (x->x_jleft[in]) t->pd = ms2d(x, ms), t->pg = g;
    else tap_set(t, ms2d(x, ms), g, blocks);
}

static int bad_in(t_tapmatrix *x, int in)
{
    if (in >= 0 && in < x->x_ins) return 0;
    pd_error(x, "tapmatrix~: no input %d", in);
    return 1;
}

/* a float array by name -> its values and size (0: error reported) */
static int getarray(t_tapmatrix *x, t_symbol *s, t_word **vec, int *n)
{
    t_garray *a = (t_garray *)pd_findbyclass(s, garray_class);
    if (!a) {
        pd_error(x, "tapmatrix~: %s: no such array", s->s_name);
        return 0;
    }
    if (!garray_getfloatwords(a, n, vec)) {
        pd_error(x, "tapmatrix~: %s: bad template", s->s_name);
        return 0;
    }
    return 1;
}

static void tapmatrix_tap(t_tapmatrix *x, t_symbol *s, int argc, t_atom *argv)
{
    int in = atom_getfloatarg(0, argc, argv), out = atom_getfloatarg(1, argc, argv);
    t_float ms = atom_getfloatarg(2, argc, argv), g = atom_getfloatarg(3, argc, argv);
    t_float ramp = argc > 4 ? atom_getfloatarg(4, argc, argv) : x->x_rampms;
    (void)s;
    if (in < 0 || in >= x->x_ins || out < 0 || out >= x->x_outs) {
        pd_error(x, "tapmatrix~: no tap %d %d", in, out);
        return;
    }
    tap_set(&x->x_tap[out * x->x_ins + in], ms2d(x, ms), g, ms2blocks(x, ramp));
}

/* row <in> <ramp-ms> <delay-array> <gain-array> [first-out]: the cells
   in -> first-out .. outs-1 from two arrays (delays in ms, gains) */
static void tapmatrix_row(t_tapmatrix *x, t_symbol *s, int argc, t_atom *argv)
{
    const int in = atom_getfloatarg(0, argc, argv), first = atom_getfloatarg(4, argc, argv);
    const int blocks = ms2blocks(x, atom_getfloatarg(1, argc, argv));
    t_word *dv, *gv;
    int nd, ng;
    (void)s;
    if (bad_in(x, in) || first < 0 || argc < 4 || !getarray(x, atom_getsymbolarg(2, argc, argv), &dv, &nd) ||
        !getarray(x, atom_getsymbolarg(3, argc, argv), &gv, &ng)) return;
    const int n = x->x_outs - first < nd ? x->x_outs - first : nd;
    for (int k = 0; k < n && k < ng; k++) row_set(x, in, first + k, dv[k].w_float, gv[k].w_float, blocks);
}

/* rowl <in> <ramp-ms> <delay> <gain> <delay> <gain> ...: the cells in -> 0, 1, ... */
static void tapmatrix_rowl(t_tapmatrix *x, t_symbol *s, int argc, t_atom *argv)
{
    const int in = atom_getfloatarg(0, argc, argv);
    const int blocks = ms2blocks(x, atom_getfloatarg(1, argc, argv));
    (void)s;
    if (bad_in(x, in)) return;
    for (int k = 0; 2 + 2 * k + 1 < argc && k < x->x_outs; k++)
        row_set(x, in, k, atom_getfloatarg(2 + 2 * k, argc, argv), atom_getfloatarg(3 + 2 * k, argc, argv), blocks);
}

/* jump <in> <xfade-ms> <delay-array> <gain-array> [first-out]: dip-and-jump of
   the whole row: fade out over xfade, jump to the new delays at the turning
   point, fade in over xfade. The cells outside first-out .. keep their
   targets. Rows and jumps for this input during the fade-out only change the
   targets after the dip. */
static void tapmatrix_jump(t_tapmatrix *x, t_symbol *s, int argc, t_atom *argv)
{
    const int in = atom_getfloatarg(0, argc, argv), first = atom_getfloatarg(4, argc, argv);
    const int xb = ms2blocks(x, atom_getfloatarg(1, argc, argv));
    const int ins = x->x_ins;
    t_word *dv, *gv;
    int nd, ng;
    (void)s;
    if (bad_in(x, in) || first < 0 || argc < 4 || !getarray(x, atom_getsymbolarg(2, argc, argv), &dv, &nd) ||
        !getarray(x, atom_getsymbolarg(3, argc, argv), &gv, &ng)) return;
    const int n = x->x_outs - first < nd ? x->x_outs - first : nd;
    if (xb < 1) {                   /* no DSP yet or no fade: set at once */
        for (int o = first; o < first + n && o - first < ng; o++)
            tap_set(&x->x_tap[o * ins + in], ms2d(x, dv[o - first].w_float), gv[o - first].w_float, 0);
        return;
    }
    if (!x->x_jleft[in]) {          /* start the dip: targets stay, gains to 0 */
        for (int o = 0; o < x->x_outs; o++) {
            t_tap *t = &x->x_tap[o * ins + in];
            t->pd = t->dto, t->pg = t->gto;
            tap_set(t, t->dto, 0, xb);
        }
        x->x_jleft[in] = xb + 1;    /* xb blocks fade, the jump at the start of the next */
        x->x_jxb[in] = xb;
    }
    for (int o = first; o < first + n && o - first < ng; o++) {
        t_tap *t = &x->x_tap[o * ins + in];
        t->pd = ms2d(x, dv[o - first].w_float), t->pg = gv[o - first].w_float;
    }
}

/* get <in> <delay-array> <gain-array>: the row's current delays (ms) and gains */
static void tapmatrix_get(t_tapmatrix *x, t_symbol *din, t_symbol *gin, t_floatarg fin)
{
    const int in = fin;
    t_word *dv, *gv;
    int nd, ng;
    if (bad_in(x, in) || !getarray(x, din, &dv, &nd) || !getarray(x, gin, &gv, &ng)) return;
    const t_float k = x->x_sr > 0 ? 1000 / x->x_sr : 0;
    for (int o = 0; o < x->x_outs; o++) {
        const t_tap *t = &x->x_tap[o * x->x_ins + in];
        if (o < nd) dv[o].w_float = t->d * k;
        if (o < ng) gv[o].w_float = t->g;
    }
}

static void tapmatrix_ramp(t_tapmatrix *x, t_floatarg f)
{
    x->x_rampms = f < 0 ? 0 : f;
}

/* clear [in]: all gains (or those of one input) to 0 at once */
static void tapmatrix_clear(t_tapmatrix *x, t_symbol *s, int argc, t_atom *argv)
{
    const int one = argc > 0, in = atom_getfloatarg(0, argc, argv);
    (void)s;
    if (one && bad_in(x, in)) return;
    for (int i = 0; i < x->x_ins * x->x_outs; i++)
        if (!one || i % x->x_ins == in)
            x->x_tap[i].g = 0, x->x_tap[i].left = 0, x->x_tap[i].hvalid = 0;
    for (int c = 0; c < x->x_ins; c++)
        if (!one || c == in) x->x_jleft[c] = 0;
}

static void tapmatrix_engine(t_tapmatrix *x, t_symbol *s)
{
    if (s == gensym("fixed")) {
        if (!TAPMATRIX_FIXED) {
            pd_error(x, "tapmatrix~: no fixed engine on this platform");
            return;
        }
        if (x->x_os) {
            pd_error(x, "tapmatrix~: -oversample: float engine only");
            return;
        }
        x->x_fixed = 1;
        tapmatrix_allocq(x);
        for (int i = 0; i < x->x_ins * x->x_outs; i++) x->x_tap[i].q.mv = 0;
    } else if (s == gensym("float")) {
        x->x_fixed = 0;
    } else pd_error(x, "tapmatrix~: engine float or fixed");
}

static void tapmatrix_check(t_tapmatrix *x, t_floatarg f)
{
    x->x_es = x->x_ee = 0;
    x->x_emax = 0;
    x->x_check = f > 0 ? f : 0;
}

static void tapmatrix_stats(t_tapmatrix *x, t_floatarg f)
{
    memset(x->x_st, 0, sizeof(x->x_st));
    x->x_stats = f > 0 ? f : 0;
}

/* "stats <moving-fixed> <moving-float> <us-total> <us-inputs> <us-moving-prep>
   <us-moving-kernel> <us-static-kernel>": per block, averaged over the
   blocks counted: moving taps computed in fixed / in float, then microseconds:
   the whole object, writing the inputs into the delay lines, preparing the
   moving taps, the moving-tap kernel, the static-tap kernel (fixed only) */
static void tapmatrix_stats_report(t_tapmatrix *x)
{
    const float b = x->x_st[0] ? x->x_st[0] : 1, us = 1.f / (CYCLES_PER_US * b);
    t_atom a[8];
    SETSYMBOL(a, gensym("stats"));
    SETFLOAT(a + 1, x->x_st[1] / b);
    SETFLOAT(a + 2, x->x_st[2] / b);
    SETFLOAT(a + 3, x->x_st[7] * us);
    SETFLOAT(a + 4, x->x_st[3] * us);
    SETFLOAT(a + 5, x->x_st[4] * us);
    SETFLOAT(a + 6, x->x_st[5] * us);
    SETFLOAT(a + 7, x->x_st[6] * us);
    outlet_anything(x->x_info, gensym("list"), 8, a);
}

static void tapmatrix_fixedshift(t_tapmatrix *x, t_floatarg f)
{
    x->x_qshift = f;
}

static void tapmatrix_report(t_tapmatrix *x)
{
    t_atom a[3];
    SETSYMBOL(a, gensym("check"));
    SETFLOAT(a + 1, x->x_ee > 0 ? 10 * log10(x->x_es / x->x_ee) : 999);
    SETFLOAT(a + 2, x->x_emax);
    outlet_anything(x->x_info, gensym("list"), 3, a);
}

static t_int *tapmatrix_perform(t_int *w)
{
    t_tapmatrix *x = (t_tapmatrix *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    int inch = (int)(w[3]);
    t_sample *out = (t_sample *)(w[4]);
    const int n = (int)(w[5]);
    const int ins = x->x_ins, outs = x->x_outs, len = x->x_len, mask = x->x_mask, mir = x->x_mir;
    const int stride = mir + len;
    const int wpos = x->x_w;
    if (!x->x_ring || n != x->x_n) {
        memset(out, 0, outs * n * sizeof(t_sample));
        return (w + 6);
    }
    const uint32_t c_start = CYCLES();
    uint32_t c_prep = 0;
    /* jumps: at the dip's turning point the row is silent: new delays, fade in */
    for (int c = 0; c < ins; c++)
        if (x->x_jleft[c] && !--x->x_jleft[c])
            for (int o = 0; o < outs; o++) {
                t_tap *t = &x->x_tap[o * ins + c];
                tap_set(t, t->pd, t->pg, x->x_jxb[c]);
            }
    const int fixed = TAPMATRIX_FIXED && x->x_fixed && x->x_qring && !(n & 7) && n <= 64;
    const t_sample inv_n = TMK(1) / n;
    /* inputs into the rings (before any output is written: in and out may overlap).
       The block is contiguous in the ring up to its end (wpos is a multiple of n
       when len is: both powers of 2); the last mir samples are mirrored in front. */
    for (int c = 0; c < ins; c++) {
        t_sample *r = x->x_ring + c * stride + mir + wpos;
        int16_t *q = fixed ? x->x_qring + c * stride + mir + wpos : 0;
        const t_sample *src = c < inch ? in + c * n : 0;
        if (src) memcpy(r, src, n * sizeof(t_sample));
        else memset(r, 0, n * sizeof(t_sample));
        if (q) fixed_from(r, q, n);
        t_sample *o = x->x_os ? x->x_oring + c * stride + mir + wpos : 0;
        if (o) {
#if TAPMATRIX_AVX2
            if (tm_avx2 && !(n & 7)) halfband_avx2(r, tm_hb, TAPMATRIX_OS_K, o, n);
            else
#endif
            halfband(r, tm_hb, TAPMATRIX_OS_K, o, n);
        }     /* reads the history: the mirror near wpos 0 */
        const int m0 = len - mir;               /* mirror [m0, len) to [0, mir) */
        if (wpos + n > m0) {
            const int from = wpos > m0 ? wpos : m0;
            memcpy(x->x_ring + c * stride + (from - m0), x->x_ring + c * stride + mir + from,
                   (wpos + n - from) * sizeof(t_sample));
            if (o) memcpy(x->x_oring + c * stride + (from - m0), x->x_oring + c * stride + mir + from,
                          (wpos + n - from) * sizeof(t_sample));
            if (q) memcpy(x->x_qring + c * stride + (from - m0), x->x_qring + c * stride + mir + from,
                          (wpos + n - from) * sizeof(int16_t));
        }
    }
    const uint32_t c_in = CYCLES();
    memset(out, 0, outs * n * sizeof(t_sample));
    uint32_t c_mk = 0, c_sk = 0;
    int n_mq = 0, n_mf = 0;
    for (int o = 0; o < outs; o++) {
        t_sample *y = out + o * n;
        int nq = 0, nm = 0;
        for (int c = 0; c < ins; c++) {
            t_tap *t = &x->x_tap[o * ins + c];
            const t_sample *r = x->x_ring + c * stride;
            if (t->left) {          /* moving: per sample */
                /* this block's ends from the ramp's start, not by adding up */
                const t_sample d0 = t->d, g0 = t->g;
                const int k = t->nb - --t->left;
                const t_sample u = k * t->inb;
                t->d = t->left ? clampd(x, t->dfrom + (t->dto - t->dfrom) * u) : t->dto;
                t->g = t->left ? t->gfrom + (t->gto - t->gfrom) * u : t->gto;
                t->hvalid = 0;
                const t_sample dd = (t->d - d0) * inv_n, gd = (t->g - g0) * inv_n;
                const uint32_t cp = CYCLES();
                if (fixed && tm_fabs(dd) < TMK(1.0 / 7)) {
                    /* the ramp's step from its ends: the per-block values carry
                       float rounding (0.0015 samples over a ramp at 2200 samples) */
                    if (!t->q.mv) fixed_ramp_start(&t->q, wpos, d0, (t->dto - t->dfrom) * t->inb * inv_n,
                                           g0, (t->gto - t->gfrom) * t->inb * inv_n);

                    fixed_ramp_block(&t->q, &x->x_mv[nm++], x->x_qring + c * stride, len, mask, mir, n);
                    if (!t->left) t->q.mv = 0;
                    n_mq++;
                    c_prep += CYCLES() - cp;
                    if (x->x_check) moving(r + mir, mask, wpos, d0, dd, g0, gd, x->x_fcheck, n);
                } else if (x->x_os) {
                    const t_sample *orr = x->x_oring + c * stride + mir;
#if TAPMATRIX_AVX2
                    if (tm_avx2 && !(n & 7) && tm_fabs(dd) < TMK(1.0 / 14))
                        moving_os_avx2(r + mir, orr, mask, len, TAPMATRIX_OS_K, wpos, d0, dd, g0, gd, y, n);
                    else
#endif
#if TAPMATRIX_NEON
                    if (!(n & 3) && tm_fabs(dd) < TMK(1.0 / 6))
                        moving_os_neon(r + mir, orr, mask, len, TAPMATRIX_OS_K, wpos, d0, dd, g0, gd, y, n);
                    else
#endif
                    moving_os(r + mir, orr, mask, TAPMATRIX_OS_K, wpos, d0, dd, g0, gd, y, n);
                    n_mf++;
                } else {
                    t->q.mv = 0;
#if TAPMATRIX_AVX2
                    if (tm_avx2 && !(n & 7) && tm_fabs(dd) < TMK(1.0 / 7))
                        moving_avx2(r + mir, mask, len, wpos, d0, dd, g0, gd, y, n);
                    else
#endif
#if TAPMATRIX_NEON
                    if (!(n & 3) && tm_fabs(dd) < TMK(1.0 / 3))
                        moving_neon(r + mir, mask, len, wpos, d0, dd, g0, gd, y, n);
                    else
#endif
                    moving(r + mir, mask, wpos, d0, dd, g0, gd, y, n);
                    n_mf++;
                }
                continue;
            }
            if (t->g == 0) continue;
            if (x->x_os) {          /* static on the 2x line u: d2 = 2d samples of u */
                const t_sample d2 = 2 * t->d;
                const int di2 = (int)d2;
                if (!t->hvalid) {
                    coeffs(d2 - di2, t->g, t->h);
                    t->hvalid = 1;
                }
                /* the newest point u[2e - di2 + 1] (e = w + i): even -> the input ring,
                   odd -> the half-samples; the next older point is in the other ring */
                const t_sample *xr = r + mir, *orr = x->x_oring + c * stride + mir;
                const int ix = (di2 & 1) ? wpos - (di2 - 1) / 2 : wpos - di2 / 2;
                const int io = (di2 & 1) ? wpos - (di2 + 1) / 2 + TAPMATRIX_OS_K : wpos - di2 / 2 + TAPMATRIX_OS_K;
                int ax = ix & mask, ao = io & mask;
                if (ax + n > len) ax -= len;
                if (ao + n > len) ao -= len;
                const t_sample *p = (di2 & 1) ? xr + ax : orr + ao, *qq = (di2 & 1) ? orr + ao : xr + ax;
#if TAPMATRIX_AVX2
                if (tm_avx2 && !(n & 7)) fir2x2_avx2(p, qq, t->h, y, n);
                else
#endif
#if TAPMATRIX_NEON
                if (!(n & 3)) fir2x2_neon(p, qq, t->h, y, n);
                else
#endif
                fir2x2(p, qq, t->h, y, n);
                continue;
            }
            /* static: a = x[e - di + 1] for the block's newest e = w + i */
            const int di = (int)t->d;
            const t_sample frac = t->d - di;
            const int a0 = (wpos - di + 1) & mask;
            /* where the block's reads cross the ring's end, take them from the
               mirror in front of it: it runs on into the ring's start */
            const int a1 = a0 + n > len ? a0 - len : a0;
            if (!t->hvalid) {       /* only when the tap changed */
                coeffs(frac, t->g, t->h);
                for (int k = 0; k < 4; k++) t->qh[k] = to_fixed(t->h[k]);
                t->hvalid = 1;
            }
            if (fixed) {
                memcpy(x->x_qcoef + nq * 4, t->qh, sizeof(t->qh));
                x->x_qsrc[nq++] = x->x_qring + c * stride + mir + a1;
                if (x->x_check) fir4(r + mir + a1, t->h, x->x_fcheck, n);
            } else {
#if TAPMATRIX_AVX2
                if (tm_avx2 && !(n & 7)) fir4_avx2(r + mir + a1, t->h, y, n);
                else
#endif
#if TAPMATRIX_NEON
                if (!(n & 3)) fir4_neon(r + mir + a1, t->h, y, n);
                else
#endif
                fir4(r + mir + a1, t->h, y, n);
            }
        }
#if TAPMATRIX_FIXED
        if (fixed && (nq || nm)) {
            int16_t *qo = x->x_qout, *qm = x->x_qout2;
            const uint32_t c0 = CYCLES();
            if (nq) tapmatrix_fixed_sum(x->x_qsrc, x->x_qcoef, nq, qo, n >> 3, x->x_qshift);
            const uint32_t c1 = CYCLES();
            if (nm) tapmatrix_fixed_moving(x->x_mv, nm, qm, n >> 3);
            c_sk += c1 - c0;
            c_mk += CYCLES() - c1;
            if (!nq) { qo = qm; qm = 0; }   /* one or both: add only what is there */
            else if (!nm) qm = 0;
            if (x->x_check) {       /* float sums per block: double is soft-float here */
                t_sample *f = x->x_fcheck, es = 0, ee = 0;
                for (int i = 0; i < n; i++) {
                    const t_sample v = (qo[i] + (qm ? qm[i] : 0)) * TMK(1.0 / 32768), e = v - f[i];
                    es += f[i] * f[i];
                    ee += e * e;
                    if (tm_fabs(e) > x->x_emax) x->x_emax = tm_fabs(e);
                    f[i] = 0;
                }
                x->x_es += es;
                x->x_ee += ee;
            }
            if (qm) for (int i = 0; i < n; i++) y[i] += (qo[i] + qm[i]) * TMK(1.0 / 32768);
            else for (int i = 0; i < n; i++) y[i] += qo[i] * TMK(1.0 / 32768);
        }
#endif
    }
    if (x->x_check && !--x->x_check) clock_delay(x->x_clock, 0);
    if (x->x_stats) {
        uint32_t *st = x->x_st;
        st[0]++; st[1] += n_mq; st[2] += n_mf;
        st[3] += c_in - c_start; st[4] += c_prep; st[5] += c_mk; st[6] += c_sk;
        st[7] += CYCLES() - c_start;
        if (!--x->x_stats) clock_delay(x->x_sclock, 0);
    }
    x->x_w = (wpos + n) & mask;
    return (w + 6);
}

static void tapmatrix_dsp(t_tapmatrix *x, t_signal **sp)
{
    const int n = sp[0]->s_length;
    x->x_sr = sp[0]->s_sr;
    tapmatrix_alloc(x, n, x->x_sr);
    signal_setmultiout(&sp[1], x->x_outs);
    dsp_add(tapmatrix_perform, 5, x, sp[0]->s_vec, (t_int)sp[0]->s_nchans, sp[1]->s_vec, (t_int)n);
}

static void *tapmatrix_new(t_symbol *s, int argc, t_atom *argv)
{
    t_tapmatrix *x = (t_tapmatrix *)pd_new(tapmatrix_class);
    (void)s;
    /* flags first: -float (the default) or -fixed; -oversample */
    while (argc && argv->a_type == A_SYMBOL && argv->a_w.w_symbol->s_name[0] == '-') {
        const t_symbol *f = argv->a_w.w_symbol;
        if (f == gensym("-fixed")) {
            if (TAPMATRIX_FIXED) x->x_fixed = 1;
            else pd_error(x, "tapmatrix~: no fixed engine on this platform: float");
        } else if (f == gensym("-float")) x->x_fixed = 0;
        else if (f == gensym("-oversample")) x->x_os = 1;
        else pd_error(x, "tapmatrix~: unknown flag %s", f->s_name);
        argc--, argv++;
    }
    if (x->x_os && x->x_fixed) {
        pd_error(x, "tapmatrix~: -oversample: float engine only");
        x->x_fixed = 0;
    }
    x->x_ins = atom_getfloatarg(0, argc, argv);
    x->x_outs = atom_getfloatarg(1, argc, argv);
    x->x_maxms = atom_getfloatarg(2, argc, argv);
    x->x_rampms = argc > 3 ? atom_getfloatarg(3, argc, argv) : 20;
    if (x->x_ins < 1) x->x_ins = 1;
    if (x->x_outs < 1) x->x_outs = 1;
    if (x->x_maxms <= 0) x->x_maxms = 100;
    x->x_sr = sys_getsr();
    /* PSRAM: in internal RAM it saved 4 % at 8x32 (2026-11-22), for 32 KB */
    x->x_tap = (t_tap *)getbytes(x->x_ins * x->x_outs * sizeof(t_tap));
    x->x_jleft = (int *)getbytes(x->x_ins * sizeof(int));
    x->x_jxb = (int *)getbytes(x->x_ins * sizeof(int));
    x->x_qcoef = (int16_t *)getbytes(x->x_ins * 4 * sizeof(int16_t));
    x->x_qsrc = (const int16_t **)getbytes(x->x_ins * sizeof(int16_t *));
    x->x_qshift = 15;
    x->x_clock = clock_new(x, (t_method)tapmatrix_report);
    x->x_sclock = clock_new(x, (t_method)tapmatrix_stats_report);
    outlet_new(&x->x_obj, &s_signal);
    x->x_info = outlet_new(&x->x_obj, 0);
    return (x);
}

static void tapmatrix_free(t_tapmatrix *x)
{
    tapmatrix_freebuf(x);
    freebytes(x->x_tap, x->x_ins * x->x_outs * sizeof(t_tap));
    freebytes(x->x_jleft, x->x_ins * sizeof(int));
    freebytes(x->x_jxb, x->x_ins * sizeof(int));
    freebytes(x->x_qcoef, x->x_ins * 4 * sizeof(int16_t));
    freebytes(x->x_qsrc, x->x_ins * sizeof(int16_t *));
    clock_free(x->x_clock);
    clock_free(x->x_sclock);
}

void tapmatrix_tilde_setup(void)
{
    tm_avx2_init();
    halfband_coeffs(tm_hb, TAPMATRIX_OS_K, TAPMATRIX_OS_BETA);
    tapmatrix_class = class_new(gensym("tapmatrix~"), (t_newmethod)tapmatrix_new,
        (t_method)tapmatrix_free, sizeof(t_tapmatrix), CLASS_MULTICHANNEL, A_GIMME, 0);
    CLASS_MAINSIGNALIN(tapmatrix_class, t_tapmatrix, x_f);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_dsp, gensym("dsp"), A_CANT, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_tap, gensym("tap"), A_GIMME, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_row, gensym("row"), A_GIMME, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_rowl, gensym("rowl"), A_GIMME, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_jump, gensym("jump"), A_GIMME, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_get, gensym("get"), A_FLOAT, A_SYMBOL, A_SYMBOL, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_ramp, gensym("ramp"), A_FLOAT, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_clear, gensym("clear"), A_GIMME, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_engine, gensym("engine"), A_SYMBOL, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_check, gensym("check"), A_FLOAT, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_stats, gensym("stats"), A_FLOAT, 0);
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_fixedshift, gensym("fixedshift"), A_FLOAT, 0);
}
