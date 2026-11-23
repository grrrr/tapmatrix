# tapmatrix~: a matrix of fractional delays for Pd

`[tapmatrix~ [-float|-q15] <ins> <outs> <maxdelay-ms> [ramp-ms]]` takes one multichannel signal with `<ins>` channels and gives one with `<outs>` channels. Every input → output cell is a tap with its own delay and gain. Each output is the sum of its taps. Delays are read with Pd's 4-point interpolation, the same as `delread4~`.

What it is for: Wave Field Synthesis (sources × speakers), multi-tap delays, delay-based panning, matrix mixing with delays. It replaces, per cell, the chain `delwrite~` → `line~` → `delread4~` → `line~` → `*~` → `+~`: in float 2.5× faster than that chain, and more accurate while delays ramp.

Needs Pd ≥ 0.56 (multichannel signals). Help patch: [tapmatrix~-help.pd](tapmatrix~-help.pd).

## Install

From Pd: *Help → Find externals*, search `tapmatrix~` (once a release is uploaded to [deken](https://deken.puredata.info)). Or build it yourself:

```sh
git clone --recurse-submodules <this repository>
make PDINCLUDEDIR=/path/to/pure-data/src        # Pd >= 0.56 headers
make install PDLIBDIR=~/Documents/Pd/externals
```

The build uses [pd-lib-builder](https://github.com/pure-data/pd-lib-builder) (a submodule): Linux, macOS, Windows (MinGW). For double-precision Pd (Pd64) add `floatsize=64 extension=<system>-<cpu>-64.<ext>` (e.g. `linux-amd64-64.so`): everything then computes in `double`, the NEON kernels are left out.

### Fallback without the binary

`tapmatrix~.pd` is an abstraction with the same arguments, inlets, outlets and messages (`tap`, `row`, `rowl`, `jump`, `get`, `ramp`, `clear`; not the flags, `engine`, `check`, `stats`, `q15shift`), built from vanilla objects: per cell `delread4~`, two `line~` and `*~` (helpers `tapmatrix.in~`, `tapmatrix.out~`, `tapmatrix.tap~`). It gives the same output as the external within float rounding, with the same latency, but costs several times as much. It is installed next to the binary: in one folder Pd tries the binary first, so the abstraction only runs where no binary loads (another platform, Pd64 without a 64-bit build). A patch that should prefer an installed binary over a fallback copy elsewhere on its path declares `[declare -stdlib tapmatrix~]`: Pd searches binaries and abstractions folder by folder, so an abstraction in the patch's folder would otherwise win.

## Creation and messages

`[tapmatrix~ [-float|-q15] <ins> <outs> <maxdelay-ms> [ramp-ms]]`

- `-float` (the default): computes in Pd's sample type (32-bit float; `double` in a Pd64 build).
- `-q15`: computes in 16-bit fixed point. On the ESP32-P4 with its PIE SIMD unit, several times faster than float; on other platforms the same arithmetic in plain C, slow, for testing only. Signals must stay within ±1.
- `<ins>`, `<outs>`: channels of the input and output signal. `<maxdelay-ms>`: the longest delay. `[ramp-ms]`: default ramp time (20 ms).

| message | what |
|---|---|
| `tap <in> <out> <delay-ms> <gain> [ramp-ms]` | set a cell. Delay and gain ramp there over `ramp-ms` (default: `ramp`), counted in whole blocks. A silent tap (gain 0) jumps to its new delay and fades in. |
| `row <in> <ramp-ms> <delay-array> <gain-array> [first-out]` | a whole row: the cells `in` → `first-out` … `outs`−1 from two Pd arrays (delays in ms, gains), read directly (no atoms). One message per input instead of one per cell. During a `jump` of that input it only sets the targets after the dip. |
| `rowl <in> <ramp-ms> <delay> <gain> <delay> <gain> …` | the same from a list (outputs 0, 1, …) |
| `jump <in> <xfade-ms> <delay-array> <gain-array> [first-out]` | dip-and-jump of the row: all its gains fade to 0 over `xfade` (delays as they were), at the turning point the delays jump, the gains fade in over `xfade`. Block-exact, no timer outside. `row`/`jump` for the input during the fade-out replace the targets. Cells outside `first-out` … keep theirs. |
| `get <in> <delay-array> <gain-array>` | the row's current delays (ms) and gains, also mid-ramp |
| `ramp <ms>` | default ramp time for later `tap` messages |
| `clear [in]` | all gains (or one input's row) to 0 at once; also ends a running jump |
| `engine float` / `engine q15` | switch the engine while running (the same as the creation flag) |
| `check <blocks>` | **accuracy test**, `q15` only. For that many blocks the object also computes everything in float and compares the two. Then the right outlet sends `check <snr-db> <peak-error>`: signal-to-error ratio of q15 against float in dB, and the largest difference in sample values. Costs about as much again while it runs. |
| `stats <blocks>` | **time measurement**. Averaged over that many blocks, the right outlet sends `stats <moving-q15> <moving-float> <us-total> <us-inputs> <us-moving-prep> <us-moving-kernel> <us-static-kernel>`: per block, how many moving taps were computed in q15 and in float, then microseconds for the whole object, for writing the inputs into the delay lines, for preparing the moving taps, for the moving-tap computation and for the static-tap computation (the last three: q15). Measured with the cycle counter on the ESP32, with the system clock elsewhere. |
| `q15shift <n>` | the accumulator shift of the PIE kernels (15; a test switch) |

The help patch shows both diagnostics with labelled number boxes.

Limits: delays from 1 sample to `<maxdelay-ms>`. In `q15`, the signals must stay within ±1 (they saturate). A tap that moves faster than 1/7 sample per sample (about 50 000 m/s for WFS: in practice only a jump with a very short ramp) takes the float path in that block.

## How it computes

Per block of n samples:

1. **Delay lines.** One ring buffer per input, a power of 2 long, in PSRAM (`getbytes`). The last n + 3 samples are mirrored in front of the ring. A block's reads are then contiguous, also where they cross the ring's end: the mirror runs on into the ring's start. In `q15` a second ring holds the inputs as int16.
2. **A tap that doesn't move in this block** is a 4-tap FIR over the delayed input: the 4-point coefficients for its fraction, times its gain. They are computed once when the tap changes and then cached. The result is added straight into its output.
3. **A moving tap** ramps delay and gain per sample. Its read position is a 32.32 integer (exact for the float increment, without per-sample float→int conversion). Each block's delay and gain are computed from the ramp's start and end, not added up block by block.
4. **`q15`**: the static taps of an output are summed by one PIE kernel (`tapmatrix_q15_sum`): 8 samples per vector, per tap and point one broadcast coefficient × one unaligned load × `esp.vmulas.s16.qacc`, all taps of the output in one accumulator, one saturating store. The moving taps of an output are summed by a second kernel (`tapmatrix_q15_moving`): per 8 samples it computes each lane's fraction q, the coefficients b = q + Q/2, c = (1−q) + P/2 − Q/2, a = −(P+Q)/6, d = −(2P−Q)/6 (P = q(1−q), Q = Pq), multiplies them by the gain ramp and accumulates them over a 5-point window. Lane masks pick points 0..3 or 1..4 where the read position steps by a sample within the 8. The C side only prepares, per 8 samples, a window pointer, the fraction offset and the gain. The lane offsets and the gain ramp stay the same for a whole ramp and are computed once, at its first block.

## Accuracy (2026-11-22)

Host tests against Pd's own chain (`delwrite~` → `line~` → `delread4~` → `*~`) and against an ideal double-precision computation (sine + noise, delays 13–20 ms, ramps over 200–300 ms):

| | Pd's chain vs ideal | `float` vs ideal | `q15` vs ideal (source at −6 dBFS) |
|---|---|---|---|
| static tap | 111.6 dB | 111.6 dB (identical to Pd's chain: 145 dB) | 82.4 dB |
| ramp 13.37 → 20.11 ms, gain 0.7 → 0.3 | 81.1 dB | 103.1 dB | 71.7 dB |
| ramp 20.11 → 5.5 ms, gain 0.3 → 1 | 60.8 dB | 106.4 dB | 73.6 dB |

While delay and gain ramp, the float engine is better than Pd's chain: `line~` adds its per-block step in float 150–225 times and drifts by up to ~0.02 samples. `q15` against `float` on the board (`check`, noise at −17 dBFS): static 77.3 dB, all taps moving 70–75 dB.

## Speed (2026-11-22)

µs per 64-sample block (48 kHz: the block is 1333 µs), measured in [pdesp](#origin-pdesp) (libpd on the ESP32-P4) with its per-object DSP profiler. The Pd version of `c4x16` is the WFS bench patch with Pd's objects.

| | ESP32-P4 rev 1 (360 MHz) | ESP32-P4 rev 3 (400 MHz) |
|---|---|---|
| 4×16 with Pd objects (`delread4~`, `line~` ×2, `*~`, sums) | ~920 | |
| `[tapmatrix~ 4 16]` float, static | 348 | 297 |
| same, all 64 taps moving | 784 | 643 |
| same, `q15`, static | 186 | 142 |
| same, `q15`, all moving | | 269 |
| `[tapmatrix~ 8 32]` float, static | > 1333 | > 1333 |
| `[tapmatrix~ 8 32]` `q15`, static | 569 | 363 |
| `[tapmatrix~ 8 32]` `q15`, all 256 taps moving | | ~1000 (board load 0.92–0.96: at the limit) |

The PIE kernels alone: static 3.7 cycles per tap and sample, moving 11.2 (10.0 in 8-sample chunks without a step). The float FIR: ~37.

**i.MX6ULL** (one Cortex-A7 at 792 MHz with NEON, Debian 10, Pd 0.56-2 built for it; 2026-11-22). CPU time of `pd -batch` per 10 s of audio, 48 kHz, 64-sample blocks; the WFS patches are the same bench patches (`c4x16` etc.) with Pd's objects:

| patch | generic armv7 build | built with `-mcpu=cortex-a7 -mfpu=neon-vfpv4 -O3 -ffast-math` |
|---|---|---|
| WFS `c1x8` / `c2x16` / `c4x16`, Pd objects | 7.9 / 28.7 / 58.2 % | |
| `[tapmatrix~ 4 16]` static | 11.9 % | 7.1 % |
| same, all 64 taps moving | 35.6 % | 36.3 % |
| `[tapmatrix~ 8 32]` static | 48.5 % | 28.1 % |
| same, all 256 taps moving | 147 % | 150 % |

The static FIR loop costs ~31 cycles per tap and sample in the generic build and ~18 with the NEON flags (GCC vectorises it only with `-ffast-math`, as NEON floats are not fully IEEE). The moving loop (~92 cycles per tap and sample) does not vectorise. So 8×32 with moving sources needs a NEON engine (float, intrinsics, the structure of the PIE kernels: 4 output samples per vector). Not written yet (open).

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
- **Other architectures.** The float engine runs everywhere. The `q15` C kernels are integer only: the natural choice for the ESP32-C3 (no FPU); not enabled or measured there. On the ESP32-S3, its `ee.*` SIMD would replace PIE (not ported).
- **Q15 quality.** ~75 dB. Q31 data (two 16-bit passes) would double the kernels' share.
- **The WFS prefilter** (`biquad~` ×4, `lop~` per source, ~20 µs per source) stays in Pd objects.

## Source layout

| file | what |
|---|---|
| `tapmatrix~.c` | the object: state, messages, the block loop (which taps move, which engine) |
| `arch/platform.h` | timer for `stats` and scratch memory: ESP32, Windows, POSIX |
| `arch/scalar.h` | the portable kernels in `t_sample` (float or double): coefficients, static FIR, moving tap |
| `arch/neon.h` | the same with NEON (armv7, arm64), single precision |
| `arch/q15.h` | the `q15` engine: its data layout, the C kernels (hosts), the declarations of the PIE kernels, a moving tap's preparation |
| `esp32/tapmatrix_pie.S` | the PIE kernels (ESP32-P4) |
| `tapmatrix~.pd`, `tapmatrix.*~.pd` | the abstraction fallback |

The kernels are `static` functions in headers, so the build stays one translation unit. A new SIMD architecture is one more header like `arch/neon.h`, included in `tapmatrix~.c` and called where the NEON kernels are.

## Tests

`tests/msgtest.py` runs Pd in batch mode (`PD=/path/to/pd python3 tests/msgtest.py`, needs numpy): `row`
and `rowl` must give bit-identical output to the same cells set by `tap`; `jump` is checked sample by sample
against its model (a linear ramp as input, exact for the 4-point interpolation); `get` and `clear <in>`.
`TM_PATH=<folder>` tests another `tapmatrix~`, e.g. the fallback alone (a folder with only the `.pd` files).

`tests/` also runs two patches with [libpd](https://github.com/libpd/libpd) and loads the built external as Pd does: `tm.pd` (float engine) and `tmq.pd` (q15 engine, source at −6 dBFS). Each compares a static and a moving tap with Pd's own `delwrite~` → `line~` → `delread4~` → `*~` chain and with an ideal double-precision computation (`ana.py`, needs numpy).

```sh
make PDINCLUDEDIR=/path/to/libpd/pure-data/src
make test PDINCLUDEDIR=/path/to/libpd/pure-data/src LIBPD_DIR=/path/to/libpd   # a built libpd
```

## CI and releases

`.gitlab-ci.yml` builds the external on Debian trixie for Linux (amd64, arm64, armv7), Windows (amd64, i386) and, with a macOS runner (`MACOS_RUNNER_TAG`), macOS (`d_fat`: x86_64 + arm64). It runs the tests, and for a version tag (`v1.0` or `1.0`) packages a binary and a sources `.dek` with deken and uploads them to puredata.info. Set `DEKEN_USERNAME` and `DEKEN_PASSWORD` as masked, protected CI variables. Each architecture has its own extension (`l_amd64`, `l_arm64`, `l_arm`, `m_amd64`, `m_i386`, `d_fat`; double precision: `linux-amd64-64.so`), so one package holds all of them, together with the abstraction fallback. The Linux binaries must still load on older systems: `tests/check_glibc.sh` fails the build if one needs a glibc newer than `GLIBC_MAX` (2.28; today they need 2.17). On armhf, trixie defaults to a 64-bit `time_t`, which would make `clock_gettime` require glibc 2.34: `tapmatrix~.c` undefines `_TIME_BITS` there (only the `stats` timer uses time types).

## Origin: pdesp

`tapmatrix~` was written for pdesp (Pure Data on the ESP32-P4), to run a Wave Field Synthesis renderer of 8 sources × 32 speakers on the chip. There it is linked into the firmware and has the `q15` engine with PIE SIMD kernels: `esp32/tapmatrix_pie.S`, built only for the ESP32-P4 (on desktop systems the `q15` engine uses the same arithmetic in C, for testing). pdesp keeps a copy of `tapmatrix~.c` (`components/pd_objects/tapmatrix_tilde.c`): change both together. Since the split into `arch/`, the copy needs the `arch/` headers next to it (or this repository as a component).

## License

BSD 3-Clause, see [LICENSE.txt](LICENSE.txt).
