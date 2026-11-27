# Notes

## Decisions and findings

- **A Pd object, not separate code.** It keeps Pd's infrastructure (patches, messages, multichannel signals, abstractions) and is useful beyond WFS.
- **Not a fixed-point Pd.** Pd's signals between objects are float. A fixed-point `*~` inside a patch would convert at every object and lose what SIMD gains. The fixed point lives inside one object instead: float in, Q15 inside, float out.
- **Restructuring first.** One loop per tap instead of five objects, coefficients cached: 2.5× faster than Pd's objects, in float, at the same precision. SIMD comes on top: 6.6× per static tap.
- **Not esp-dsp's FIR.** `dsps_fird_s16` (P4: `_arp4`) vectorises across the taps of one output sample. With 4 taps that wastes half the 8 lanes and pays for the accumulator per sample: it was 2.4× slower than the float FIR. `dsps_fird_f32_arp4` crashed with 4 taps (esp-dsp 1.8.2 has no test for it). `dsps_fird_s16_arp4` takes its coefficients in the opposite order from `dsps_fird_s16_ansi`. Our kernels vectorise across 8 output samples instead.
- **No crossfade for moving taps.** A crossfade per block between the start and end coefficients is cheap, but wrong for real motion. Worst case at 10 kHz: −65 dB at 0.05 samples of delay change per block (~0.25 m/s), −14 dB at 1 sample, −3 dB at 2 samples (10 m/s Doppler is 1.9 samples per block). So moving taps interpolate per sample, in SIMD.
- **Fractions and masks in the kernel.** In C, a lane's fraction and step cost 13 instructions; per sample that was more than the kernel itself. In the kernel: t = a + v (16 bit, never overflows), q = t & 0x7fff, mask t < 0. The fraction is then at most 1 LSB (3·10⁻⁵ samples) off.
- **Ramps from their ends.** Each block's delay is computed from the ramp's start, not added up: the added-up float drifted by 0.0015 samples over a ramp at 2200 samples. The moving kernel's step is computed from the ramp's ends for the same reason (the first block's rounded value lost 10 dB).
- **PIE encodings differ between chip revisions.** ESP-IDF builds v1.x with `-march=…_xesppie` and v3.x with `…_xesploop_xespv`. The assembly must not set `-march` itself: with the v1 encoding a v3 chip faulted on the first PIE instruction ("Store access fault" on `esp.movx.w.cfg`, "Unknown" on `esp.zero.qacc`).
- **`esp.vldbc.16.ip` takes its address register from x8..x15 only** (`s0`, `s1`, `a0`–`a5`).
- **Memory.** Delay lines and tap state in PSRAM. In internal RAM the 8×32 tap array (32 KB) saved 4 %: not worth it. The per-output scratch of the moving kernel (≤ 112 bytes per input) is in internal RAM. Unaligned 128-bit loads cost the same as aligned ones, from PSRAM as from internal RAM.
- **Measure in the object.** `check` and `stats` exist because estimates were wrong twice. A `check` computing in `double` was itself the cost (the P4 has no double-precision FPU).


## Open

- **8 × 32 with all taps moving** uses ~1000 µs of the 1333 µs block. With the board's other audio work that leaves no headroom. Per block: moving kernel ~183k cycles, C prep ~133k, inputs ~19k. The prep loop is about 35 instructions per 8 samples per tap. Merging it into the kernel is the next step.
- **Static taps on x86.** `fir4` adds each tap into the output buffer, so the buffer is read and written once per tap; summing all taps of an output in registers (as the PIE kernel does) would save that.
- **Double precision.** A Pd64 build computes in `double` with the scalar kernels. SIMD for double exists on x86 (AVX2: 4 lanes) and arm64 (NEON: 2 lanes), not on armv7; not written yet (half the lanes of float, so at most half the gain).
- **Other architectures.** The float engine runs everywhere. The `fixed` C kernels are integer only: the natural choice for the ESP32-C3 (no FPU); not enabled or measured there. On the ESP32-S3, its `ee.*` SIMD would replace PIE (not ported).
- **Fixed-point quality.** ~75 dB (16-bit data). Q31 data (two 16-bit passes) would double the kernels' share.
- **The WFS prefilter** (`biquad~` ×4, `lop~` per source, ~20 µs per source) stays in Pd objects.


## Accuracy (2026-11-22)

Host tests against Pd's own chain (`delwrite~` → `line~` → `delread4~` → `*~`) and against an ideal double-precision computation (sine + noise, delays 13–20 ms, ramps over 200–300 ms):

| | Pd's chain vs ideal | `float` vs ideal | `fixed` vs ideal (source at −6 dBFS) |
|---|---|---|---|
| static tap | 111.6 dB | 111.6 dB (identical to Pd's chain: 145 dB) | 82.4 dB |
| ramp 13.37 → 20.11 ms, gain 0.7 → 0.3 | 81.1 dB | 103.1 dB | 71.7 dB |
| ramp 20.11 → 5.5 ms, gain 0.3 → 1 | 60.8 dB | 106.4 dB | 73.6 dB |

`-oversample` against the plain 4-point interpolation (`tests/ostest.py`, a static tap's frequency response from its impulse response; a moving tap: a 15 kHz sine through a delay ramp of 20 samples):

| | plain | `-oversample` |
|---|---|---|
| static, fraction 0.5: 12 / 16 / 20 kHz | −1.07 / −3.25 / −8.42 dB | 0.00 / 0.00 / 0.00 dB |
| static, fraction 0.25: 12 / 16 / 20 kHz | −0.73 / −1.97 / −3.84 dB | −0.07 / −0.23 / −0.53 dB |
| 15 kHz while the delay moves: level ripple (mean) | 2.46 dB (−1.45 dB) | 0.17 dB (−0.11 dB) |

With AVX2 (x86, FMA: one rounding per multiply-add) the two ramps give 103.5 and 105.1 dB, the static tap the same. While delay and gain ramp, the float engine is better than Pd's chain: `line~` adds its per-block step in float 150–225 times and drifts by up to ~0.02 samples. `fixed` against `float` on the board (`check`, noise at −17 dBFS): static 77.3 dB, all taps moving 70–75 dB.

## Speed (2026-11-22)

µs per 64-sample block (48 kHz: the block is 1333 µs), measured in [pdesp](#origin-pdesp) (libpd on the ESP32-P4) with its per-object DSP profiler. The Pd version of `c4x16` is the WFS bench patch with Pd's objects.

| | ESP32-P4 rev 1 (360 MHz) | ESP32-P4 rev 3 (400 MHz) |
|---|---|---|
| 4×16 with Pd objects (`delread4~`, `line~` ×2, `*~`, sums) | ~920 | |
| `[tapmatrix~ 4 16]` float, static | 348 | 297 |
| same, all 64 taps moving | 784 | 643 |
| same, `fixed`, static | 186 | 142 |
| same, `fixed`, all moving | | 269 |
| `[tapmatrix~ 8 32]` float, static | > 1333 | > 1333 |
| `[tapmatrix~ 8 32]` `fixed`, static | 569 | 363 |
| `[tapmatrix~ 8 32]` `fixed`, all 256 taps moving | | ~1000 (board load 0.92–0.96: at the limit) |

The PIE kernels alone: static 3.7 cycles per tap and sample, moving 11.2 (10.0 in 8-sample chunks without a step). The float FIR: ~37.

**x86_64 with AVX2** (Intel i7-6800K, 3.4 GHz, Debian, gcc -O3, Pd 0.56-5; 2026-11-24). `[tapmatrix~ 8 32]` float, µs per 64-sample block from `stats`:

| | without AVX2 (`-DTAPMATRIX_NO_AVX2`, gcc's own SSE2) | with AVX2 + FMA |
|---|---|---|
| static | 11.0 | 7.7 |
| all 256 taps moving | 77.3 | 24.2 |

With `-oversample` (best of 3; the halfband for the 8 inputs takes 2.4 µs): static 6.8 → 11.1 µs, all moving 24.4 → 33.4 µs.

The AVX2 kernels are chosen at run time (`__builtin_cpu_supports`): the binaries stay baseline x86 and run on CPUs without AVX2.

**i.MX6ULL** (one Cortex-A7 at 792 MHz with NEON, Debian 10, Pd 0.56-2 built for it; 2026-11-22). CPU time of `pd -batch` per 10 s of audio, 48 kHz, 64-sample blocks; the WFS patches are the same bench patches (`c4x16` etc.) with Pd's objects:

| patch | generic armv7 build | built with `-mcpu=cortex-a7 -mfpu=neon-vfpv4 -O3 -ffast-math` |
|---|---|---|
| WFS `c1x8` / `c2x16` / `c4x16`, Pd objects | 7.9 / 28.7 / 58.2 % | |
| `[tapmatrix~ 4 16]` static | 11.9 % | 7.1 % |
| same, all 64 taps moving | 35.6 % | 36.3 % |
| `[tapmatrix~ 8 32]` static | 48.5 % | 28.1 % |
| same, all 256 taps moving | 147 % | 150 % |

The static FIR loop costs ~31 cycles per tap and sample in the generic build and ~18 with the NEON flags (GCC vectorises it only with `-ffast-math`, as NEON floats are not fully IEEE). The moving loop (~92 cycles per tap and sample) does not vectorise. So 8×32 with moving sources needs a NEON engine (float, intrinsics, the structure of the PIE kernels: 4 output samples per vector). Not written yet (open).


**Silent moving taps skipped** (Apple M1 Pro, macOS 15, Apple clang 14 -O3 -ffast-math (pd-lib-builder), NEON, Pd 0.56-2 `-batch`; 2026-11-27). `[tapmatrix~ 16 64]`, 16 sources, a `rowl` per source every 20 ms with a 20 ms ramp (so all 1024 cells ramp all the time), 12 of 64 cells per source on, the rest at gain 0; µs per 64-sample block from `stats` (`us-total`), median of 5 runs of 2000 blocks, moving taps computed per block in brackets:

| | before (main) | `skip 0` | `skip 1` (the default) |
|---|---|---|---|
| float | 70.4 (1024) | 71.9 (1024) | 32.1 (194) |
| `-oversample` | 101.1 (1024) | 101.3 (1024) | 40.1 (194) |
| `-fixed` (host C kernels: not a speed figure) | 546 (1024) | 546 (1024) | 149 (194) |

Before, only static taps with gain 0 were skipped: a moving source paid for all its speakers. What is left at `skip 1`: the 194 audible moving taps and the per-cell work of the block loop (ramp ends, branches) for all 1024.

## CI and releases

`.gitlab-ci.yml` builds the external on Debian trixie for Linux (amd64, arm64, armv7), Windows (amd64, i386) and macOS (`d_fat`: x86_64 + arm64).
It runs the tests, and for a version tag (`v1.0` or `1.0`) packages a binary and a sources `.dek` with deken and uploads them to puredata.info.
Set `DEKEN_USERNAME` and `DEKEN_PASSWORD` as masked, protected CI variables.
Each architecture has its own extension (`l_amd64`, `l_arm64`, `l_arm`, `m_amd64`, `m_i386`, `d_fat`; double precision: `linux-amd64-64.so`), so one package holds all of them, together with the abstraction fallback.
The Linux binaries must still load on older systems: `tests/check_glibc.sh` fails the build if one needs a glibc newer than `GLIBC_MAX` (2.28; today they need 2.17).
On armhf, trixie defaults to a 64-bit `time_t`, which would make `clock_gettime` require glibc 2.34: `tapmatrix~.c` undefines `_TIME_BITS` there (only the `stats` timer uses time types).


## Origin: pdesp

`tapmatrix~` was written for pdesp (Pure Data on the ESP32-P4), to run a Wave Field Synthesis renderer of 8 sources × 32 speakers on the chip. There it is linked into the firmware and has the `fixed` engine with PIE SIMD kernels: `arch/fixed_pie.S`, built only for the ESP32-P4 (on desktop systems the `fixed` engine uses the same arithmetic in C, for testing). pdesp keeps a copy of `tapmatrix~.c` (`components/pd_objects/tapmatrix_tilde.c`): change both together. Since the split into `arch/`, the copy needs the `arch/` headers next to it (or this repository as a component).

