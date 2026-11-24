/*
 * The two kernels of the fixed engine on the ESP32-P4 (arch/fixed.h has the
 * same arithmetic in C, which is the easier way to read them).
 *
 * A short guide to what is used here
 * ----------------------------------
 * The P4's cores are 32-bit RISC-V (registers a0..a7 arguments, t0..t6
 * scratch, s0..s1 saved; lw/lh/sw load/store a word/halfword, addi/add/slli
 * arithmetic, bnez/beqz/blt branches). The PIE extension ("esp." prefix) adds:
 *   q0..q7     eight 128-bit vector registers, here always 8 lanes of int16
 *   QACC       one accumulator with 8 lanes of 40 bits: products add up there
 *              without overflow; it is read out once per 8 output samples
 *   SAR        the shift used by esp.vmul.s16
 *   cfg        a control register; bit 1 allows unaligned 128-bit loads
 * Instructions (all lane by lane, 8 lanes):
 *   esp.vld.128.ip  q, a, k     load 8 samples from address a, then a += k
 *   esp.vld.128.xp  q, a, r     load 8 samples from address a, then a += r (a register)
 *   esp.vldbc.16.ip q, a, k     load ONE int16 from a and copy it to all 8 lanes
 *                               ("broadcast"), then a += k
 *   esp.vst.128.ip  q, a, k     store 8 samples, then a += k
 *   esp.vmulas.s16.qacc qx, qy  QACC += qx * qy (full 32-bit products)
 *   esp.srcmb.s16.qacc q, r, 0  q = QACC >> r, rounded and saturated to int16
 *   esp.zero.qacc               QACC = 0
 *   esp.vmul.s16 q, qx, qy      q = (qx * qy) >> SAR, saturated (SAR = 15:
 *                               the product of two fixed-point numbers)
 *   esp.vadd.s16 / esp.vsub.s16 add / subtract, saturated
 *   esp.andq / esp.orq / esp.notq   bitwise and / or / not (masks)
 *   esp.vcmp.lt.s16 q, qx, qy   q = all ones where qx < qy, else 0
 *   esp.movx.r.cfg / .w.cfg / .w.sar   read / write cfg, write SAR
 * Every vector load reads 8 consecutive int16 = 16 bytes.
 *
 * Kernel 1: the static taps of one output
 * ---------------------------------------
 *   out[i] = (sum over taps t, k = 0..3 of coef[4t+k] * src[t][i-k]) >> shift
 *   for i = 0 .. 8*nchunks-1.
 * A static tap is a 4-point FIR: 4 coefficients (its delay's fraction and its
 * gain, fixed for the block) times 4 consecutive input samples. Per chunk of 8
 * output samples and per tap: broadcast coefficient k to all lanes, load the 8
 * input samples src[i-k .. i-k+7], multiply-accumulate. After all taps of the
 * output, one shift and one store: 8 outputs.
 * void tapmatrix_fixed_sum(const int16_t *const *src, const int16_t *coef,
 *                        int ntaps, int16_t *out, int nchunks, int shift);
 *   a0 src (pointers to each tap's newest needed sample), a1 coef (4 per tap),
 *   a2 ntaps (>= 1), a3 out (16-byte aligned), a4 nchunks (>= 1), a5 shift
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
    esp.movx.r.cfg  t5          // t5 = cfg (kept to restore it)
    ori     t6, t5, 2           // set bit 1: unaligned loads allowed
    esp.movx.w.cfg  t6
    .endm
    .macro UNALIGNED_OFF
    esp.movx.w.cfg  t5          // cfg as it was
    .endm

    .text
    .align  4
    .global tapmatrix_fixed_sum
    .type   tapmatrix_fixed_sum, @function
tapmatrix_fixed_sum:
    addi    sp, sp, -16         // s0 is callee-saved: keep it on the stack
    sw      s0, 0(sp)
    UNALIGNED_ON
    li      t4, -2              // load step: one sample (2 bytes) back
    li      a6, 0               // byte offset of the chunk: 16 per 8 samples
.Lchunk:                        // for each chunk of 8 output samples:
    esp.zero.qacc               //   QACC = 0
    mv      t0, a0              //   t0: walks through src[]
    mv      s0, a1              //   s0: walks through coef[]
    mv      t2, a2              //   t2: taps left
.Ltap:                          //   for each tap:
    lw      t3, 0(t0)           //     t3 = src[t]
    addi    t0, t0, 4
    add     t3, t3, a6          //     + this chunk: &src[t][8c]
    esp.vldbc.16.ip     q0, s0, 2       // q0 = coef k=0 in all lanes
    esp.vld.128.xp      q1, t3, t4      // q1 = src[t][8c .. 8c+7], then t3 one sample back
    esp.vmulas.s16.qacc q0, q1          // QACC += coef0 * x[i]
    esp.vldbc.16.ip     q2, s0, 2       // coef k=1
    esp.vld.128.xp      q3, t3, t4      // src[t][8c-1 .. 8c+6]
    esp.vmulas.s16.qacc q2, q3          // QACC += coef1 * x[i-1]
    esp.vldbc.16.ip     q4, s0, 2       // coef k=2
    esp.vld.128.xp      q5, t3, t4      // x[i-2]
    esp.vmulas.s16.qacc q4, q5
    esp.vldbc.16.ip     q6, s0, 2       // coef k=3
    esp.vld.128.xp      q7, t3, t4      // x[i-3]
    esp.vmulas.s16.qacc q6, q7
    addi    t2, t2, -1
    bnez    t2, .Ltap           //   next tap
    esp.srcmb.s16.qacc  q0, a5, 0       // q0 = QACC >> shift, saturated: 8 outputs
    esp.vst.128.ip      q0, a3, 16      // store them, out += 8 samples
    addi    a6, a6, 16
    addi    a4, a4, -1
    bnez    a4, .Lchunk         // next chunk
    UNALIGNED_OFF
    lw      s0, 0(sp)
    addi    sp, sp, 16
    ret
    .size   tapmatrix_fixed_sum, . - tapmatrix_fixed_sum

/*
 * Kernel 2: the moving taps of one output
 * ---------------------------------------
 * A moving tap's delay and gain change every sample (a linear ramp), so its 4
 * interpolation coefficients differ per sample and lane: they are computed
 * here, 8 at a time, from each lane's fraction q of the read position
 * (Pd's 4-point formula of delread4~, rearranged for integers):
 *   u = 1 - q, P = q*u, Q = P*q
 *   coefficients of the 4 points, oldest first:
 *   d = -(2P - Q)/6, c = u + P/2 - Q/2, b = q + Q/2, a = -(P + Q)/6
 *   each times the lane's gain g0[c] + rg[i].
 * The points: lane i needs 4 consecutive input samples around its read
 * position. Within 8 samples the read position moves by 8 * (1 - dd) samples
 * with |dd| < 1/7, so relative to the lane index it steps by at most one
 * sample: lane i finds its 4 points among the 5 samples base[c][i .. i+4]
 * (5 loads, one sample apart, give all 8 windows). A lane uses points 0..3 or
 * points 1..4 of its window. fixed_ramp_block() (arch/fixed.h) prepares, per
 * chunk, base[c], a[c] and mode[c]; per lane the ramp's constant offsets v[i]:
 *   t = a[c] + v[i]   (int16, never overflows)
 *   q = t & 0x7fff    the fraction (15 bits)
 *   t < 0             the lane uses points 0..3 ("L"), else points 1..4
 *   mode[c] = 1 / 2   no step within the chunk: all lanes use points 0..3 / 1..4
 *                     (the fast path: 4 loads, no masks); 0: masked, 5 loads
 * void tapmatrix_fixed_moving(const t_mvtap *taps, int ntaps, int16_t *out, int nchunks);
 *   a0 taps (16-aligned, 112 bytes each: base[8] @0, g0[8] @32, a[8] @48,
 *   mode[8] @64, v[8] @80, rg[8] @96), a1 ntaps (>= 1), a2 out (16-aligned),
 *   a3 nchunks
 */
    .section .rodata
    .align  1
.Lmv_const:                     // 1, 1/2 and -1/6 in fixed point
    .short  32767, 16384, -5461
    .text
    .align  4
    .global tapmatrix_fixed_moving
    .type   tapmatrix_fixed_moving, @function
tapmatrix_fixed_moving:
    addi    sp, sp, -16         // keep s0, s1 (callee-saved)
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    UNALIGNED_ON
    li      a4, 15
    esp.movx.w.sar  a4          // SAR = 15: vmul.s16 gives (a*b) >> 15
    li      s1, 15              // final shift of QACC
    li      t4, 2               // load step through the window: one sample on
    li      a6, 0               // chunk index c
.Lmv_chunk:                     // for each chunk of 8 output samples:
    esp.zero.qacc               //   QACC = 0
    mv      s0, a0              //   s0: the current tap
    mv      a7, a1              //   a7: taps left
.Lmv_tap:                       //   for each moving tap:
    // --- the lanes' fractions: t = a[c] + v[i], q = t & 0x7fff, u = 1 - q
    slli    t2, a6, 1           //     t2 = 2c: offset of the int16 entries of chunk c
    add     a4, s0, t2
    addi    a4, a4, 48
    esp.vldbc.16.ip     q0, a4, 2       // q0 = a[c] in all lanes
    addi    a5, s0, 80
    esp.vld.128.ip      q2, a5, 16      // q2 = v[0..7]
    esp.vadd.s16        q0, q0, q2      // q0 = t
    la      a5, .Lmv_const
    esp.vldbc.16.ip     q1, a5, 2       // q1 = 32767 (1) in all lanes
    esp.andq            q0, q0, q1      // q0 = q (drop the sign bit)
    esp.vsub.s16        q1, q1, q0      // q1 = u = 1 - q
    // --- P, Q and the 4 coefficients
    esp.vmul.s16        q2, q0, q1      // q2 = P = q*u
    esp.vmul.s16        q3, q2, q0      // q3 = Q = P*q
    esp.vldbc.16.ip     q4, a5, 2       // q4 = 1/2
    esp.vmul.s16        q5, q3, q4      // q5 = Q/2
    esp.vadd.s16        q6, q0, q5      // q6 = b = q + Q/2
    esp.vmul.s16        q7, q2, q4      // q7 = P/2
    esp.vadd.s16        q1, q1, q7      //      u + P/2
    esp.vsub.s16        q1, q1, q5      // q1 = c = u + P/2 - Q/2
    esp.vadd.s16        q4, q2, q3      //      P + Q
    esp.vldbc.16.ip     q5, a5, 2       // q5 = -1/6
    esp.vmul.s16        q4, q4, q5      // q4 = a = -(P + Q)/6
    esp.vadd.s16        q7, q2, q2      //      2P
    esp.vsub.s16        q7, q7, q3      //      2P - Q
    esp.vmul.s16        q7, q7, q5      // q7 = d = -(2P - Q)/6
    // --- times the gain ramp: g = g0[c] + rg[i]
    add     a4, s0, t2
    addi    a4, a4, 32
    esp.vldbc.16.ip     q0, a4, 2       // q0 = g0[c] in all lanes
    addi    a5, s0, 96
    esp.vld.128.ip      q2, a5, 16      // q2 = rg[0..7]
    esp.vadd.s16        q0, q0, q2      // q0 = g per lane
    esp.vmul.s16        q4, q4, q0      // a *= g
    esp.vmul.s16        q6, q6, q0      // b *= g
    esp.vmul.s16        q1, q1, q0      // c *= g
    esp.vmul.s16        q7, q7, q0      // d *= g
    // --- the window: base[c] (a pointer) and mode[c]
    slli    t0, a6, 2
    add     t1, s0, t0
    lw      t3, 0(t1)           //     t3 = base[c]
    add     t1, s0, t2
    lh      t1, 64(t1)          //     t1 = mode[c]
    beqz    t1, .Lmv_masked     //     0: some lanes step within the chunk
    // --- fast path: all lanes use the same 4 points (mode 1: 0..3, 2: 1..4)
    addi    t1, t1, -1          //     mode 1 -> start at point 0, 2 -> point 1
    slli    t1, t1, 1
    add     t3, t3, t1
    esp.vld.128.xp      q0, t3, t4      // q0 = point 0 (oldest) of every lane's 4
    esp.vmulas.s16.qacc q7, q0          // QACC += d * point 0
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q1, q0          // QACC += c * point 1
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q6, q0          // QACC += b * point 2
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q4, q0          // QACC += a * point 3
    j       .Lmv_next
.Lmv_masked:
    // --- masked path: lanes in L use window points 0..3, the others (H) 1..4.
    //     Each of the 5 window points gets the coefficient of the lanes that
    //     use it (L: d c b a at points 0..3, H: d c b a at points 1..4)
    add     a4, s0, t2
    addi    a4, a4, 48
    esp.vldbc.16.ip     q2, a4, 2       // a[c]
    addi    a5, s0, 80
    esp.vld.128.ip      q3, a5, 16      // v
    esp.vadd.s16        q2, q2, q3      // t again
    esp.vsub.s16        q3, q3, q3      // q3 = 0
    esp.vcmp.lt.s16     q2, q2, q3      // q2 = L: all ones where t < 0
    esp.notq            q3, q2          // q3 = H = not L
    esp.andq            q5, q7, q2      // point 0: d in L lanes, 0 in H
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q1, q2      // point 1: c in L lanes | d in H lanes
    esp.andq            q7, q7, q3
    esp.orq             q5, q5, q7
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q6, q2      // point 2: b (L) | c (H)
    esp.andq            q1, q1, q3
    esp.orq             q5, q5, q1
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q4, q2      // point 3: a (L) | b (H)
    esp.andq            q6, q6, q3
    esp.orq             q5, q5, q6
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
    esp.andq            q5, q4, q3      // point 4: 0 (L) | a (H)
    esp.vld.128.xp      q0, t3, t4
    esp.vmulas.s16.qacc q5, q0
.Lmv_next:
    addi    s0, s0, 112         //   next tap (112 bytes each)
    addi    a7, a7, -1
    bnez    a7, .Lmv_tap
    esp.srcmb.s16.qacc  q0, s1, 0       // q0 = QACC >> 15, saturated: 8 outputs
    esp.vst.128.ip      q0, a2, 16      // store, out += 8 samples
    addi    a6, a6, 1
    blt     a6, a3, .Lmv_chunk  // next chunk
    UNALIGNED_OFF
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    addi    sp, sp, 16
    ret
    .size   tapmatrix_fixed_moving, . - tapmatrix_fixed_moving
