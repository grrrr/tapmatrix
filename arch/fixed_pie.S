/*
 * ESP32-P4 PIE kernel for [tapmatrix~] (tapmatrix_tilde.c), Q15:
 *   out[i] = (sum over taps t, k = 0..3 of coef[4t+k] * src[t][i-k]) >> shift
 *   for i = 0 .. 8*nchunks-1, 8 samples per vector, all taps of one output
 *   accumulated in QACC before one store.
 * void tapmatrix_q15_sum(const int16_t *const *src, const int16_t *coef,
 *                        int ntaps, int16_t *out, int nchunks, int shift);
 *   a0 src, a1 coef, a2 ntaps (>= 1), a3 out (16-byte aligned), a4 nchunks (>= 1), a5 shift
 * src[t] need not be aligned: unaligned vector loads are switched on (cfg bit 1,
 * as esp-dsp's dsps_fird_s16_arp4 does; only bits 0..1 written) and restored. No hardware loops.
 * esp.vldbc.16.ip takes its address register from x8..x15 only: s0 here.
 */
/* Unaligned vector loads: cfg bit 1 (as esp-dsp's dsps_fird_s16_arp4 does).
 * The PIE encodings differ between chip v1.x (-march=..._xesppie) and v3.x
 * (..._xesploop_xespv): ESP-IDF picks the flag per revision, so this file must
 * not set -march itself. With the v1 encoding a v3 chip faulted on the first
 * PIE instruction (2026-11-22). */
    .macro UNALIGNED_ON
    esp.movx.r.cfg  t5
    ori     t6, t5, 2
    esp.movx.w.cfg  t6
    .endm
    .macro UNALIGNED_OFF
    esp.movx.w.cfg  t5
    .endm

    .text
    .align  4
    .global tapmatrix_q15_sum
    .type   tapmatrix_q15_sum, @function
tapmatrix_q15_sum:
    addi    sp, sp, -16
    sw      s0, 0(sp)
    UNALIGNED_ON
    li      t4, -2              // one sample back
    li      a6, 0               // byte offset of the chunk
.Lchunk:
    esp.zero.qacc
    mv      t0, a0
    mv      s0, a1
    mv      t2, a2
.Ltap:
    lw      t3, 0(t0)
    addi    t0, t0, 4
    add     t3, t3, a6
    esp.vldbc.16.ip     q0, s0, 2       // coef k = 0, broadcast
    esp.vld.128.xp      q1, t3, t4      // src[i .. i+7]
    esp.vmulas.s16.qacc q0, q1
    esp.vldbc.16.ip     q2, s0, 2
    esp.vld.128.xp      q3, t3, t4      // src[i-1 ..]
    esp.vmulas.s16.qacc q2, q3
    esp.vldbc.16.ip     q4, s0, 2
    esp.vld.128.xp      q5, t3, t4
    esp.vmulas.s16.qacc q4, q5
    esp.vldbc.16.ip     q6, s0, 2
    esp.vld.128.xp      q7, t3, t4
    esp.vmulas.s16.qacc q6, q7
    addi    t2, t2, -1
    bnez    t2, .Ltap
    esp.srcmb.s16.qacc  q0, a5, 0       // q0 = qacc >> shift, saturated
    esp.vst.128.ip      q0, a3, 16
    addi    a6, a6, 16
    addi    a4, a4, -1
    bnez    a4, .Lchunk
    UNALIGNED_OFF
    lw      s0, 0(sp)
    addi    sp, sp, 16
    ret
    .size   tapmatrix_q15_sum, . - tapmatrix_q15_sum

/*
 * Moving taps (Q15): for chunk c (8 samples), all taps accumulated in QACC:
 *   t = a[c] + v[i]; q = t & 0x7fff (fraction); u = 1-q; P = q*u; Q = P*q
 *   b = q + Q/2, c = u + P/2 - Q/2, a = -(P+Q)/6, d = -(2P-Q)/6   (times the gain)
 *   window points 0..4 = base[c][i .. i+4]; mode[c] = 1: all lanes use points
 *   0..3 (d c b a), 2: points 1..4, 0: lanes with t < 0 points 0..3, the
 *   others 1..4.
 * void tapmatrix_q15_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks);
 *   a0 taps (16-aligned, 112 bytes each: base[8] @0, g0[8] @32, a[8] @48,
 *   mode[8] @64, v[8] @80, rg[8] @96), a1 ntaps (>= 1), a2 out (16-aligned),
 *   a3 nchunks
 * vmul.s16 shifts by SAR: set to 15 (Q15 products).
 */
    .section .rodata
    .align  1
.Lmv_const:
    .short  32767, 16384, -5461
    .text
    .align  4
    .global tapmatrix_q15_moving
    .type   tapmatrix_q15_moving, @function
tapmatrix_q15_moving:
    addi    sp, sp, -16
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    UNALIGNED_ON
    li      a4, 15
    esp.movx.w.sar  a4          // vmul.s16: (a*b) >> 15
    li      s1, 15              // final shift of QACC
    li      t4, 2               // window: one sample on
    li      a6, 0               // chunk
.Lmv_chunk:
    esp.zero.qacc
    mv      s0, a0
    mv      a7, a1
.Lmv_tap:
    slli    t2, a6, 1           // 2 * c
    add     a4, s0, t2
    addi    a4, a4, 48
    esp.vldbc.16.ip     q0, a4, 2       // a[c]
    addi    a5, s0, 80
    esp.vld.128.ip      q2, a5, 16      // v
    esp.vadd.s16        q0, q0, q2      // t
    la      a5, .Lmv_const
    esp.vldbc.16.ip     q1, a5, 2       // 32767
    esp.andq            q0, q0, q1      // q = t & 0x7fff
    esp.vsub.s16        q1, q1, q0      // u
    esp.vmul.s16        q2, q0, q1      // P
    esp.vmul.s16        q3, q2, q0      // Q
    esp.vldbc.16.ip     q4, a5, 2       // 1/2
    esp.vmul.s16        q5, q3, q4      // Q/2
    esp.vadd.s16        q6, q0, q5      // b
    esp.vmul.s16        q7, q2, q4      // P/2
    esp.vadd.s16        q1, q1, q7
    esp.vsub.s16        q1, q1, q5      // c
    esp.vadd.s16        q4, q2, q3      // P + Q
    esp.vldbc.16.ip     q5, a5, 2       // -1/6
    esp.vmul.s16        q4, q4, q5      // a
    esp.vadd.s16        q7, q2, q2
    esp.vsub.s16        q7, q7, q3
    esp.vmul.s16        q7, q7, q5      // d
    add     a4, s0, t2          // gain: g0[c] + rg[i]
    addi    a4, a4, 32
    esp.vldbc.16.ip     q0, a4, 2
    addi    a5, s0, 96
    esp.vld.128.ip      q2, a5, 16
    esp.vadd.s16        q0, q0, q2
    esp.vmul.s16        q4, q4, q0
    esp.vmul.s16        q6, q6, q0
    esp.vmul.s16        q1, q1, q0
    esp.vmul.s16        q7, q7, q0
    slli    t0, a6, 2
    add     t1, s0, t0
    lw      t3, 0(t1)           // base[c]
    add     t1, s0, t2
    lh      t1, 64(t1)          // mode[c]
    beqz    t1, .Lmv_masked
    addi    t1, t1, -1          // 1: points 0..3, 2: points 1..4
    slli    t1, t1, 1
    add     t3, t3, t1
    esp.vld.128.xp      q0, t3, t4      // d c b a, no masks
    esp.vmulas.s16.qacc q7, q0
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q1, q0
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q6, q0
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q4, q0
    j       .Lmv_next
.Lmv_masked:
    add     a4, s0, t2          // L = (t < 0): points 0..3
    addi    a4, a4, 48
    esp.vldbc.16.ip     q2, a4, 2
    addi    a5, s0, 80
    esp.vld.128.ip      q3, a5, 16
    esp.vadd.s16        q2, q2, q3      // t again
    esp.vsub.s16        q3, q3, q3      // 0
    esp.vcmp.lt.s16     q2, q2, q3      // L
    esp.notq            q3, q2          // H: points 1..4
    esp.andq            q5, q7, q2      // point 0: d & L
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q1, q2      // point 1: c & L | d & H
    esp.andq            q7, q7, q3
    esp.orq             q5, q5, q7
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q6, q2      // point 2: b & L | c & H
    esp.andq            q1, q1, q3
    esp.orq             q5, q5, q1
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q4, q2      // point 3: a & L | b & H
    esp.andq            q6, q6, q3
    esp.orq             q5, q5, q6
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q4, q3      // point 4: a & H
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
.Lmv_next:
    addi    s0, s0, 112
    addi    a7, a7, -1
    bnez    a7, .Lmv_tap
    esp.srcmb.s16.qacc  q0, s1, 0
    esp.vst.128.ip      q0, a2, 16
    addi    a6, a6, 1
    blt     a6, a3, .Lmv_chunk
    UNALIGNED_OFF
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    addi    sp, sp, 16
    ret
    .size   tapmatrix_q15_moving, . - tapmatrix_q15_moving
