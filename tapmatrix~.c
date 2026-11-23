/*
 * [tapmatrix~ [-float|-q15] <ins> <outs> <maxdelay-ms> [ramp-ms]]
 *
 * A matrix of fractional delays: every input -> output cell is a tap with its
 * own delay and gain. Input: one multichannel signal (<ins> channels, missing
 * ones are silent). Output: one multichannel signal of <outs> channels, each
 * the sum of its taps. For WFS (sources x speakers), multi-tap delays, delay
 * panning and matrix mixing. Delay read with Pd's 4-point interpolation
 * (as delread4~). Design, measurements and decisions: README.md here.
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
 *   engine float|q15                            switch the engine (also the flag
 *                                               at creation; float is the default).
 *                                               q15: 16-bit fixed point (ESP32-P4:
 *                                               PIE SIMD; host: the same in C, for tests)
 *   check <blocks>                              accuracy test, q15 only: for that many
 *                                               blocks also compute everything in
 *                                               float and compare; then the right
 *                                               outlet sends "check <snr-db> <peak-error>"
 *   stats <blocks>                              time measurement: averaged over that
 *                                               many blocks, the right outlet sends
 *                                               "stats <moving-q15> <moving-float>
 *                                               <us-total> <us-inputs> <us-moving-prep>
 *                                               <us-moving-kernel> <us-static-kernel>"
 *   q15shift <n>                                accumulator shift for PIE (test)
 *
 * How it computes (per block of n samples): the inputs go into ring buffers
 * (PSRAM, via getbytes; the last n+3 samples mirrored in front, so a block's
 * reads are contiguous, also across the ring's end). A tap that doesn't move
 * in this block is a 4-tap FIR with the gain folded into its coefficients,
 * added straight into the output. A moving tap ramps its delay and gain per
 * sample; its read position is a 32.32 integer. Ramps are counted in whole
 * blocks, each block's values computed from the ramp's ends.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "m_pd.h"

#if defined(ESP_PLATFORM)
#include "sdkconfig.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#define CYCLES() esp_cpu_get_cycle_count()
#define CYCLES_PER_US ((float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)
/* scratch touched for every tap and block: internal RAM, not PSRAM (getbytes) */
static void *fastmem(size_t n) { void *p = heap_caps_calloc(1, n, MALLOC_CAP_INTERNAL); return p ? p : getbytes(n); }
static void fastfree(void *p, size_t n) { if (esp_ptr_internal(p)) heap_caps_free(p); else freebytes(p, n); }
#else
#include <time.h>
static inline uint32_t ns_now(void)    /* wraps after 4.3 s: only differences are used */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000000u + ts.tv_nsec);
}
#define CYCLES() ns_now()
#define CYCLES_PER_US 1000.f              /* here: nanoseconds */
#define fastmem getbytes
#define fastfree freebytes
#endif
/* One moving tap for one block, prepared for the Q15 kernel (112 bytes,
   16-aligned). Chunk c = samples 8c .. 8c+7. Lane i reads a 5-point window
   base[c][i .. i+4]. With t = a[c] + v[i] (16-bit, never overflows): the
   fraction of the read position past x[j] is t & 0x7fff (Q15), and lanes with
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
} __attribute__((aligned(16))) t_mvtap;     /* offsets used by tapmatrix_pie.S */
_Static_assert(sizeof(void *) != 4 || sizeof(t_mvtap) == 112, "t_mvtap: tapmatrix_pie.S expects 112 bytes");

#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TAPMATRIX_PIE 1
/* tapmatrix_pie.S: out[i] = sum over taps t, k=0..3 of coef[4t+k] * src[t][i-k],
   i = 0 .. 8*nchunks-1, Q15, the accumulator shifted by `shift` */
void tapmatrix_q15_sum(const int16_t *const *src, const int16_t *coef, int ntaps,
                       int16_t *out, int nchunks, int shift);
/* moving taps, n = 8*nchunks samples (<= 64), Q15 */
void tapmatrix_q15_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks);
#define TAPMATRIX_Q15 1
#else
#define TAPMATRIX_PIE 0
#if defined(ESP_PLATFORM)
#define TAPMATRIX_Q15 0             /* other ESP32s: no q15 engine */
#else
#define TAPMATRIX_Q15 1             /* host: q15 with the C kernels below, for tests */
#endif
/* Elsewhere the same integer arithmetic in C, to test the Q15 engine on a host:
   vmul = (a*b) >> 15, saturated; add/sub saturated; the accumulator 64-bit. */
static inline int16_t sat16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : v; }
static inline int16_t vmul(int16_t a, int16_t b) { return sat16(((int32_t)a * b) >> 15); }
static void tapmatrix_q15_sum(const int16_t *const *src, const int16_t *coef, int ntaps,
                              int16_t *out, int nchunks, int shift)
{
    for (int i = 0; i < 8 * nchunks; i++) {
        int64_t acc = 0;
        for (int t = 0; t < ntaps; t++)
            for (int k = 0; k < 4; k++) acc += (int32_t)coef[4 * t + k] * src[t][i - k];
        out[i] = sat16((int32_t)(acc >> shift));
    }
}
static void tapmatrix_q15_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks)
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

static t_class *tapmatrix_class;

typedef struct {
    float d, g;                     /* now: delay in samples, gain */
    float dfrom, dto, gfrom, gto;   /* ramp: from, to */
    int nb, left;                   /* ramp: blocks in all, left */
    float inb;                      /* 1 / nb */
    /* q15 engine, moving: the ramp's fixed parts, set in its first block */
    int mv;                         /* set */
    int up;                         /* the read position runs faster than the input */
    uint64_t T, inc;                /* read position (32.32) of the next block, step */
    int32_t G, G8;                  /* gain (Q30) of the next block, step per 8 samples */
    int16_t v[8], rg[8];
    int hvalid;                     /* h, qh hold the coefficients of d, g */
    float h[4];
    int16_t qh[4];
    float pd, pg;                   /* jump: delay (samples) and gain after the dip */
} t_tap;

/* float -> Q15, rounded, saturated (no lrintf: a library call) */
static inline int16_t to_q15(float v)
{
    v *= 32768.f;
    if (v >= 32767.f) return 32767;
    if (v <= -32768.f) return -32768;
    return (int16_t)(int32_t)(v + (v >= 0 ? 0.5f : -0.5f));
}

typedef struct _tapmatrix {
    t_object x_obj;
    t_float x_f;
    int x_ins, x_outs;
    float x_maxms, x_rampms;
    float x_sr;
    int x_n;                        /* block length the buffers are made for */
    int x_len, x_mask, x_mir;       /* ring: length (power of 2), mirror n+3 */
    int x_w;                        /* logical write position */
    float *x_ring;                  /* [ins][mir + len] */
    t_tap *x_tap;                   /* [outs][ins] */
    int *x_jleft;                   /* [ins] jump: blocks to the dip's turning point + 1, 0 = none */
    int *x_jxb;                     /* [ins] jump: fade length in blocks */
    int x_q15;
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
    uint32_t x_st[8];               /* blocks, q15 moving, float moving, cycles:
                                       inputs, prep, moving kernel, static kernel, all */
    double x_es, x_ee;              /* signal and error energy */
    float x_emax;
    float *x_fcheck;                /* [n] float reference of the q15 part */
    t_outlet *x_info;
    t_clock *x_clock;               /* reports the check outside the DSP tick */
    t_clock *x_sclock;              /* and the stats */
} t_tapmatrix;

static void tapmatrix_freebuf(t_tapmatrix *x)
{
    if (x->x_ring) freebytes(x->x_ring, x->x_ins * (x->x_mir + x->x_len) * sizeof(float));
    if (x->x_qring) freebytes(x->x_qring, x->x_ins * (x->x_mir + x->x_len) * sizeof(int16_t));
    if (x->x_qmem) fastfree(x->x_qmem, 2 * x->x_n * sizeof(int16_t) + 32);
    if (x->x_mvmem) fastfree(x->x_mvmem, x->x_ins * sizeof(t_mvtap) + 16);
    x->x_mvmem = 0; x->x_mv = 0;
    if (x->x_fcheck) freebytes(x->x_fcheck, x->x_n * sizeof(float));
    x->x_ring = 0; x->x_qring = 0; x->x_qmem = 0; x->x_qout = 0; x->x_fcheck = 0;
}

static void tapmatrix_allocq(t_tapmatrix *x)
{
    if (!x->x_q15 || x->x_qring || !x->x_n) return;
    x->x_qring = (int16_t *)getbytes(x->x_ins * (x->x_mir + x->x_len) * sizeof(int16_t));
    x->x_qmem = fastmem(2 * x->x_n * sizeof(int16_t) + 32);
    x->x_qout = (int16_t *)(((uintptr_t)x->x_qmem + 15) & ~(uintptr_t)15);
    x->x_qout2 = x->x_qout + ((x->x_n + 7) & ~7);
    x->x_mvmem = fastmem(x->x_ins * sizeof(t_mvtap) + 16);
    x->x_mv = (t_mvtap *)(((uintptr_t)x->x_mvmem + 15) & ~(uintptr_t)15);
    x->x_fcheck = (float *)getbytes(x->x_n * sizeof(float));
}

/* (re)make the buffers for block length n */
static void tapmatrix_alloc(t_tapmatrix *x, int n, float sr)
{
    int need = (int)(x->x_maxms * 0.001f * sr) + n + 8, len = 64;
    while (len < need) len <<= 1;
    if (x->x_ring && x->x_n == n && x->x_len == len) return;
    tapmatrix_freebuf(x);
    x->x_n = n;
    x->x_len = len;
    x->x_mask = len - 1;
    x->x_mir = n + 3;
    x->x_w = 0;
    for (int i = 0; i < x->x_ins * x->x_outs; i++) x->x_tap[i].mv = 0;
    x->x_ring = (float *)getbytes(x->x_ins * (x->x_mir + len) * sizeof(float));
    tapmatrix_allocq(x);
}

static float clampd(t_tapmatrix *x, float d)
{
    float hi = x->x_len - x->x_n - 4;
    return (d < 1 ? 1 : d > hi ? hi : d);
}

/* delay ms -> samples, as delread4~ computes it */
static float ms2d(t_tapmatrix *x, float ms)
{
    float d = ms * (float)(x->x_sr * 0.001);
    return x->x_n ? clampd(x, d) : d;
}

/* ramp ms -> whole blocks (0 before the first DSP run: set at once) */
static int ms2blocks(t_tapmatrix *x, float ms)
{
    return (x->x_n && x->x_sr > 0) ? (int)(ms * 0.001f * x->x_sr / x->x_n + 0.5f) : 0;
}

/* one cell to delay d (samples) and gain g over `blocks` blocks; a silent tap
   (gain 0 now) jumps to its new delay and fades in */
static void tap_set(t_tap *t, float d, float g, int blocks)
{
    t->hvalid = 0;
    t->mv = 0;
    if (t->g == 0) t->d = d;
    t->dfrom = t->d; t->gfrom = t->g;
    t->dto = d; t->gto = g;
    if (blocks < 1) {
        t->d = d; t->g = g; t->left = 0;
    } else {
        t->nb = t->left = blocks;
        t->inb = 1.f / blocks;
    }
}

/* a cell of input `in` from a row message: while the row dips (jump), only
   the targets after the dip change */
static void row_set(t_tapmatrix *x, int in, int out, float ms, float g, int blocks)
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
    float ms = atom_getfloatarg(2, argc, argv), g = atom_getfloatarg(3, argc, argv);
    float ramp = argc > 4 ? atom_getfloatarg(4, argc, argv) : x->x_rampms;
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
    const float k = x->x_sr > 0 ? 1000.f / x->x_sr : 0;
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
    if (s == gensym("q15")) {
        if (!TAPMATRIX_Q15) {
            pd_error(x, "tapmatrix~: no q15 engine on this platform");
            return;
        }
        x->x_q15 = 1;
        tapmatrix_allocq(x);
        for (int i = 0; i < x->x_ins * x->x_outs; i++) x->x_tap[i].mv = 0;
    } else if (s == gensym("float")) {
        x->x_q15 = 0;
    } else pd_error(x, "tapmatrix~: engine float or q15");
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

/* "stats <moving-q15> <moving-float> <us-total> <us-inputs> <us-moving-prep>
   <us-moving-kernel> <us-static-kernel>": per block, averaged over the
   blocks counted: moving taps computed in q15 / in float, then microseconds:
   the whole object, writing the inputs into the delay lines, preparing the
   moving taps, the moving-tap kernel, the static-tap kernel (q15 only) */
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

static void tapmatrix_q15shift(t_tapmatrix *x, t_floatarg f)
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

/* 4-point coefficients of delread4~, for a (newest), b, c, d (oldest);
   frac goes from b (0) to c (1) */
static inline void coeffs(float frac, float g, float h[4])
{
    /* Pd's formula is linear in a, b, c, d: evaluate it for one input at a time */
    const float f1 = 1.f - frac;
    for (int k = 0; k < 4; k++) {
        const float a = k == 0, b = k == 1, c = k == 2, d = k == 3, cb = c - b;
        h[k] = g * (b + frac * (cb - (1.f / 6.f) * f1 * ((d - a - 3.f * cb) * frac + (d + 2.f * a - 3.f * b))));
    }
}

/* one static tap: out[i] += h0 x[i] + h1 x[i-1] + h2 x[i-2] + h3 x[i-3] */
static inline void fir4(const float *x, const float h[4], float *out, int n)
{
    const float h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3];
    for (int i = 0; i < n; i++)
        out[i] += h0 * x[i] + h1 * x[i - 1] + h2 * x[i - 2] + h3 * x[i - 3];
}

/* x samples -> 32.32 fixed point; the float splits exactly into two halves */
static inline int64_t fix32(float x)
{
    const float hi = floorf(x);
    const float lo = (x - hi) * 65536.f;            /* exact, [0, 65536) */
    const float lohi = floorf(lo);
    return (int64_t)(int32_t)hi * 4294967296ll + (int64_t)(int32_t)lohi * 65536 +
        (int64_t)(int32_t)((lo - lohi) * 65536.f);
}

/* one moving tap, per sample: delay d0 -> d0 + dd*n, gain g0 -> g0 + gd*n.
   The read position t = w + i - d is a 32.32 integer: between x[j] (older)
   and x[j+1] */
static void moving(const float *ring, int mask, int w, float d0, float dd, float g0, float gd,
                   float *out, int n)
{
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd);
    float g = g0;
    for (int i = 0; i < n; i++, T += inc, g += gd) {
        const int j = (int)(uint32_t)(T >> 32);
        const float fp = (float)((uint32_t)T >> 8) * (1.f / 16777216.f), frac = 1.f - fp;
        const float c = ring[j & mask], b = ring[(j + 1) & mask],
            a = ring[(j + 2) & mask], d = ring[(j - 1) & mask], cb = c - b;
        out[i] += g * (b + frac * (cb - (1.f / 6.f) * fp * ((d - a - 3.f * cb) * frac + (d + 2.f * a - 3.f * b))));
    }
}

/* a moving tap's block, for the Q15 kernel: 0 if its reads wrap around the ring
   (then the float path takes it) */
/* x >= 0 samples -> 32.32, without floorf (a library call) */
static inline uint64_t fixpos(float x)
{
    const int32_t hi = (int32_t)x;
    const float r = (x - (float)hi) * 65536.f;
    const int32_t mid = (int32_t)r;
    const int32_t lo = (int32_t)((r - (float)mid) * 65536.f);
    return ((uint64_t)(uint32_t)hi << 32) | ((uint32_t)mid << 16) | (uint32_t)lo;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
/* The float engine with NEON (armv7 with -mfpu=neon, arm64): 4 output samples
   per vector. The same results as fir4() and moving() up to float rounding. */
#include <arm_neon.h>
#define TAPMATRIX_NEON 1

static void fir4_neon(const float *x, const float h[4], float *out, int n)
{
    const float32x4_t h0 = vdupq_n_f32(h[0]), h1 = vdupq_n_f32(h[1]),
        h2 = vdupq_n_f32(h[2]), h3 = vdupq_n_f32(h[3]);
    for (int i = 0; i < n; i += 4) {
        float32x4_t acc = vld1q_f32(out + i);
        acc = vmlaq_f32(acc, h0, vld1q_f32(x + i));
        acc = vmlaq_f32(acc, h1, vld1q_f32(x + i - 1));
        acc = vmlaq_f32(acc, h2, vld1q_f32(x + i - 2));
        acc = vmlaq_f32(acc, h3, vld1q_f32(x + i - 3));
        vst1q_f32(out + i, acc);
    }
}

/* A moving tap, 4 samples per vector. Lane i of a chunk reads a 5-point
   window ring[X + i .. X + i + 4]; lanes in L use points 0..3 (d c b a), the
   others 1..4: within 4 samples the read position steps by at most one
   sample (|dd| < 1/3). ring: logical 0 of the ring (the mirror in front of it). */
static void moving_neon(const float *ring, int mask, int len, int w, float d0, float dd,
                        float g0, float gd, float *out, int n)
{
    uint64_t T = ((uint64_t)(uint32_t)w << 32) - (uint64_t)fix32(d0);
    const uint64_t inc = (1ull << 32) - (uint64_t)fix32(dd), inc4 = inc * 4;
    const uint32_t step = (uint32_t)inc;    /* the fraction's step per sample */
    const int up = dd < 0;          /* the read position runs faster than the input */
    const uint32_t off = up ? 1 : 2;
    const uint32_t lo[4] = { 0, step, 2 * step, 3 * step };
    const uint32x4_t offs = vld1q_u32(lo);
    const float gl[4] = { 0, gd, 2 * gd, 3 * gd };
    const float32x4_t rg = vld1q_f32(gl), sixth = vdupq_n_f32(1.f / 6.f),
        one = vdupq_n_f32(1.f), two = vdupq_n_f32(2.f), three = vdupq_n_f32(3.f),
        tofrac = vdupq_n_f32(1.f / 16777216.f);
    float g = g0;
    for (int i = 0; i < n; i += 4, T += inc4, g += 4 * gd) {
        const uint32_t J = (uint32_t)(T >> 32), F0 = (uint32_t)T;
        int X = (int)((J - off) & mask);
        if (X + 8 > len) X -= len;          /* near the end: the mirror in front */
        const float *p = ring + X;
        const uint32x4_t f = vaddq_u32(vdupq_n_u32(F0), offs);
        /* lanes past a step: the fraction wrapped (falling: it grew; rising: it shrank) */
        const uint32x4_t wrapped = up ? vcltq_u32(f, vdupq_n_u32(F0)) : vcgtq_u32(f, vdupq_n_u32(F0));
        const uint32x4_t L = up ? vmvnq_u32(wrapped) : wrapped;
        const float32x4_t w0 = vld1q_f32(p), w1 = vld1q_f32(p + 1), w2 = vld1q_f32(p + 2),
            w3 = vld1q_f32(p + 3), w4 = vld1q_f32(p + 4);
        const float32x4_t d = vbslq_f32(L, w0, w1), c = vbslq_f32(L, w1, w2),
            b = vbslq_f32(L, w2, w3), a = vbslq_f32(L, w3, w4);
        const float32x4_t fp = vmulq_f32(vcvtq_f32_u32(vshrq_n_u32(f, 8)), tofrac);
        const float32x4_t frac = vsubq_f32(one, fp), cb = vsubq_f32(c, b);
        /* y = b + frac*(cb - (1/6) fp ((d - a - 3cb) frac + (d + 2a - 3b))), as delread4~ */
        float32x4_t t1 = vmlaq_f32(vsubq_f32(d, a), vnegq_f32(three), cb);
        float32x4_t t2 = vmlsq_f32(vmlaq_f32(d, two, a), three, b);
        float32x4_t t = vmlaq_f32(t2, t1, frac);
        t = vmlsq_f32(cb, vmulq_f32(sixth, fp), t);
        const float32x4_t y = vmlaq_f32(b, frac, t);
        const float32x4_t gv = vaddq_f32(vdupq_n_f32(g), rg);
        vst1q_f32(out + i, vmlaq_f32(vld1q_f32(out + i), y, gv));
    }
}
#else
#define TAPMATRIX_NEON 0
#endif

/* A moving tap's first block in the q15 engine: what stays the same for the
   ramp (delay and gain change by the same amount every block). |dd| < 1/7. */
static void ramp_start(t_tap *t, int w, float d0, float dd, float g0, float gd)
{
    const int32_t ddf = (int32_t)(dd * 2147483648.f);   /* the bits below 2^-31 don't matter */
    t->inc = (1ull << 32) - (uint64_t)((int64_t)ddf * 2);
    t->T = ((uint64_t)(uint32_t)w << 32) - fixpos(d0);
    t->up = ddf < 0;
    /* lane offsets in Q15 of a sample: the fraction moves by |dd| per sample
       (up) or by -dd; t = a + v stays within int16 */
    const uint32_t D = (uint32_t)(t->up ? -(int64_t)ddf : ddf) * 2;   /* |dd| * 2^32 */
    for (int i = 0; i < 8; i++) {
        const int16_t o = (int16_t)(((uint32_t)i * D) >> 17);
        t->v[i] = t->up ? o : -o;
    }
    const float g8 = 8 * gd * 1073741824.f;
    t->G = (int32_t)(g0 * 1073741824.f + 0.5f);
    t->G8 = (int32_t)(g8 + (g8 < 0 ? -0.5f : 0.5f));
    for (int i = 0; i < 8; i++) t->rg[i] = (int16_t)(((int64_t)i * t->G8 / 8) >> 15);
    t->mv = 1;
}

/* every block of the ramp: the chunks' windows, fractions and gains, then on */
static void ramp_block(t_tapmatrix *x, t_tap *t, t_mvtap *m, const int16_t *qr, int n)
{
    memcpy(m->v, t->v, sizeof(m->v));
    memcpy(m->rg, t->rg, sizeof(m->rg));
    const int up = t->up, len = x->x_len, mask = x->x_mask, mir = x->x_mir;
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
    const int q15 = TAPMATRIX_Q15 && x->x_q15 && x->x_qring && !(n & 7) && n <= 64;
    const float inv_n = 1.f / n;
    /* inputs into the rings (before any output is written: in and out may overlap).
       The block is contiguous in the ring up to its end (wpos is a multiple of n
       when len is: both powers of 2); the last mir samples are mirrored in front. */
    for (int c = 0; c < ins; c++) {
        float *r = x->x_ring + c * stride + mir + wpos;
        int16_t *q = q15 ? x->x_qring + c * stride + mir + wpos : 0;
        const t_sample *src = c < inch ? in + c * n : 0;
        if (src) memcpy(r, src, n * sizeof(float));
        else memset(r, 0, n * sizeof(float));
        if (q) {
            for (int i = 0; i < n; i++) {
                int32_t v = (int32_t)(r[i] * 32768.f + __builtin_copysignf(0.5f, r[i]));
                v = v > 32767 ? 32767 : v;
                q[i] = (int16_t)(v < -32768 ? -32768 : v);
            }
        }
        const int m0 = len - mir;               /* mirror [m0, len) to [0, mir) */
        if (wpos + n > m0) {
            const int from = wpos > m0 ? wpos : m0;
            memcpy(x->x_ring + c * stride + (from - m0), x->x_ring + c * stride + mir + from,
                   (wpos + n - from) * sizeof(float));
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
            const float *r = x->x_ring + c * stride;
            if (t->left) {          /* moving: per sample */
                /* this block's ends from the ramp's start, not by adding up */
                const float d0 = t->d, g0 = t->g;
                const int k = t->nb - --t->left;
                const float u = k * t->inb;
                t->d = t->left ? clampd(x, t->dfrom + (t->dto - t->dfrom) * u) : t->dto;
                t->g = t->left ? t->gfrom + (t->gto - t->gfrom) * u : t->gto;
                t->hvalid = 0;
                const float dd = (t->d - d0) * inv_n, gd = (t->g - g0) * inv_n;
                const uint32_t cp = CYCLES();
                if (q15 && fabsf(dd) < 1.f / 7) {
                    /* the ramp's step from its ends: the per-block values carry
                       float rounding (0.0015 samples over a ramp at 2200 samples) */
                    if (!t->mv) ramp_start(t, wpos, d0, (t->dto - t->dfrom) * t->inb * inv_n,
                                           g0, (t->gto - t->gfrom) * t->inb * inv_n);

                    ramp_block(x, t, &x->x_mv[nm++], x->x_qring + c * stride, n);
                    if (!t->left) t->mv = 0;
                    n_mq++;
                    c_prep += CYCLES() - cp;
                    if (x->x_check) moving(r + mir, mask, wpos, d0, dd, g0, gd, x->x_fcheck, n);
                } else {
                    t->mv = 0;
#if TAPMATRIX_NEON
                    if (!(n & 3) && fabsf(dd) < 1.f / 3)
                        moving_neon(r + mir, mask, len, wpos, d0, dd, g0, gd, y, n);
                    else
#endif
                    moving(r + mir, mask, wpos, d0, dd, g0, gd, y, n);
                    n_mf++;
                }
                continue;
            }
            if (t->g == 0) continue;
            /* static: a = x[e - di + 1] for the block's newest e = w + i */
            const int di = (int)t->d;
            const float frac = t->d - di;
            const int a0 = (wpos - di + 1) & mask;
            /* where the block's reads cross the ring's end, take them from the
               mirror in front of it: it runs on into the ring's start */
            const int a1 = a0 + n > len ? a0 - len : a0;
            if (!t->hvalid) {       /* only when the tap changed */
                coeffs(frac, t->g, t->h);
                for (int k = 0; k < 4; k++) t->qh[k] = to_q15(t->h[k]);
                t->hvalid = 1;
            }
            if (q15) {
                memcpy(x->x_qcoef + nq * 4, t->qh, sizeof(t->qh));
                x->x_qsrc[nq++] = x->x_qring + c * stride + mir + a1;
                if (x->x_check) fir4(r + mir + a1, t->h, x->x_fcheck, n);
            } else {
#if TAPMATRIX_NEON
                if (!(n & 3)) fir4_neon(r + mir + a1, t->h, y, n);
                else
#endif
                fir4(r + mir + a1, t->h, y, n);
            }
        }
#if TAPMATRIX_Q15
        if (q15 && (nq || nm)) {
            int16_t *qo = x->x_qout, *qm = x->x_qout2;
            const uint32_t c0 = CYCLES();
            if (nq) tapmatrix_q15_sum(x->x_qsrc, x->x_qcoef, nq, qo, n >> 3, x->x_qshift);
            const uint32_t c1 = CYCLES();
            if (nm) tapmatrix_q15_moving(x->x_mv, nm, qm, n >> 3);
            c_sk += c1 - c0;
            c_mk += CYCLES() - c1;
            if (!nq) { qo = qm; qm = 0; }   /* one or both: add only what is there */
            else if (!nm) qm = 0;
            if (x->x_check) {       /* float sums per block: double is soft-float here */
                float *f = x->x_fcheck, es = 0, ee = 0;
                for (int i = 0; i < n; i++) {
                    const float v = (qo[i] + (qm ? qm[i] : 0)) * (1.f / 32768.f), e = v - f[i];
                    es += f[i] * f[i];
                    ee += e * e;
                    if (fabsf(e) > x->x_emax) x->x_emax = fabsf(e);
                    f[i] = 0;
                }
                x->x_es += es;
                x->x_ee += ee;
            }
            if (qm) for (int i = 0; i < n; i++) y[i] += (qo[i] + qm[i]) * (1.f / 32768.f);
            else for (int i = 0; i < n; i++) y[i] += qo[i] * (1.f / 32768.f);
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
    /* flags first: -float (the default) or -q15 */
    while (argc && argv->a_type == A_SYMBOL && argv->a_w.w_symbol->s_name[0] == '-') {
        const t_symbol *f = argv->a_w.w_symbol;
        if (f == gensym("-q15")) {
            if (TAPMATRIX_Q15) x->x_q15 = 1;
            else pd_error(x, "tapmatrix~: no q15 engine on this platform: float");
        } else if (f == gensym("-float")) x->x_q15 = 0;
        else pd_error(x, "tapmatrix~: unknown flag %s", f->s_name);
        argc--, argv++;
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
    class_addmethod(tapmatrix_class, (t_method)tapmatrix_q15shift, gensym("q15shift"), A_FLOAT, 0);
}
