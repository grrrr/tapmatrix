# tapmatrix~: a matrix of fractional delays for Pd

`[tapmatrix~ [-float|-fixed] [-oversample] <ins> <outs> <maxdelay-ms> [ramp-ms]]` takes one multichannel signal with `<ins>` channels and gives one with `<outs>` channels. Every input → output cell is a tap with its own delay and gain. Each output is the sum of its taps.
Delays are read with Pd's 4-point interpolation, the same as `delread4~`.

What it is for: Wave Field Synthesis (sources × speakers), multi-tap delays, delay-based panning, matrix mixing with delays.
It replaces, per cell, the chain `delwrite~` → `line~` → `delread4~` → `line~` → `*~` → `+~`: in float 2.5× faster than that chain, and more accurate while delays ramp.

Needs Pd ≥ 0.56 (multichannel signals).
Help patch: [tapmatrix~-help.pd](tapmatrix~-help.pd).

## Install

From Pd: *Help → Find externals*, search `tapmatrix~` (once a release is uploaded to [deken](https://deken.puredata.info)).
Or build it yourself:

```sh
git clone --recurse-submodules <this repository>
make PDINCLUDEDIR=/path/to/pure-data/src        # Pd >= 0.56 headers
make install PDLIBDIR=~/Documents/Pd/externals
```

The build uses [pd-lib-builder](https://github.com/pure-data/pd-lib-builder) (a submodule): Linux, macOS, Windows (MinGW).
For double-precision Pd (Pd64) add `floatsize=64 extension=<system>-<cpu>-64.<ext>` (e.g. `linux-amd64-64.so`): everything then computes in `double`, the NEON kernels are left out.

### Fallback without the binary

`tapmatrix~.pd` is an abstraction with the same arguments, inlets, outlets and messages (`tap`, `row`, `rowl`, `jump`, `get`, `ramp`, `clear`; not the flags, `engine`, `check`, `stats`, `fixedshift`), built from vanilla objects: per cell `delread4~`, two `line~` and `*~` (helpers `tapmatrix.in~`, `tapmatrix.out~`, `tapmatrix.tap~`).
It gives the same output as the external within float rounding, with the same latency, but costs several times as much.
It is installed next to the binary: in one folder Pd tries the binary first, so the abstraction only runs where no binary loads (another platform, Pd64 without a 64-bit build).
A patch that should prefer an installed binary over a fallback copy elsewhere on its path declares `[declare -stdlib tapmatrix~]`: Pd searches binaries and abstractions folder by folder, so an abstraction in the patch's folder would otherwise win.

## Creation and messages

`[tapmatrix~ [-float|-fixed] [-oversample] <ins> <outs> <maxdelay-ms> [ramp-ms]]`

- `-float` (the default): computes in Pd's sample type (32-bit float; `double` in a Pd64 build).
- `-fixed`: computes in 16-bit fixed point. On the ESP32-P4 with its PIE SIMD unit, several times faster than float; on other platforms the same arithmetic in plain C, slow, for testing only. Signals must stay within ±1.
- `-oversample`: reads the delay lines as if they were sampled at twice the rate (a halfband interpolator of 48 taps per input, then the same 4-point interpolation). The 4-point interpolation alone loses up to 8.4 dB at 20 kHz (48 kHz, fraction 0.5) and modulates the level of high frequencies while a delay moves; with `-oversample` it is within 0.53 dB up to 20 kHz. Float engine only. Delays start at 25 samples instead of 1 (the interpolator's half length); the delays themselves stay exact, no latency is added. Costs per input, not per tap: +40–65 % for `[tapmatrix~ 8 32]` (see Speed). The abstraction fallback has no flags.
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
| `engine float` / `engine fixed` | switch the engine while running (the same as the creation flag) |
| `check <blocks>` | **accuracy test**, `fixed` only. For that many blocks the object also computes everything in float and compares the two. Then the right outlet sends `check <snr-db> <peak-error>`: signal-to-error ratio of fixed against float in dB, and the largest difference in sample values. Costs about as much again while it runs. |
| `stats <blocks>` | **time measurement**. Averaged over that many blocks, the right outlet sends `stats <moving-fixed> <moving-float> <us-total> <us-inputs> <us-moving-prep> <us-moving-kernel> <us-static-kernel>`: per block, how many moving taps were computed in fixed and in float, then microseconds for the whole object, for writing the inputs into the delay lines, for preparing the moving taps, for the moving-tap computation and for the static-tap computation (the last three: fixed). Measured with the cycle counter on the ESP32, with the system clock elsewhere. |
| `fixedshift <n>` | the accumulator shift of the PIE kernels (15; a test switch) |

The help patch shows both diagnostics with labelled number boxes.

Limits: delays from 1 sample to `<maxdelay-ms>`. In `fixed`, the signals must stay within ±1 (they saturate).
A tap whose delay changes faster than 1/7 sample per sample (a radial speed of c/7 ≈ 49 m/s; in WFS: a position jump with a short ramp) takes the scalar float path in that block; the SIMD float kernels have the same limit with AVX2 (8 lanes) and 1/3 with NEON (4 lanes).
The limit is not a setting: it is where the kernels' window of 5 samples per lane stops being enough (see "Source layout").

## How it computes

Per block of n samples:

1. **Delay lines.** One ring buffer per input, a power of 2 long, in PSRAM (`getbytes`). The last n + 3 samples are mirrored in front of the ring. A block's reads are then contiguous, also where they cross the ring's end: the mirror runs on into the ring's start. In `fixed` a second ring holds the inputs as int16.
2. **A tap that doesn't move in this block** is a 4-tap FIR over the delayed input: the 4-point coefficients for its fraction, times its gain. They are computed once when the tap changes and then cached. The result is added straight into its output.
3. **A moving tap** ramps delay and gain per sample. Its read position is a 32.32 integer (exact for the float increment, without per-sample float→int conversion). Each block's delay and gain are computed from the ramp's start and end, not added up block by block.
4. **`-oversample`.** A second ring per input holds the half-sample values, computed when the input arrives by a Kaiser-windowed halfband (2 × 24 taps, β = 9: passband error and images below −91 dB up to 20 kHz at 48 kHz; `TAPMATRIX_OS_K` at compile time sets the half length). The 2× signal u is then u[2k] = x[k] (the input ring) and u[2k+1] = the half-sample ring, and the 4-point interpolation reads u. A static tap's 4 points alternate between the two rings, so it is two 2-tap FIRs on contiguous memory (`fir2x2`); a moving tap's window points of all lanes are, per point, consecutive values of one ring, so the SIMD moving kernels work as before with 2·dd (limit |dd| < 1/14 with AVX2, 1/6 with NEON).
5. **`fixed`**: the static taps of an output are summed by one PIE kernel (`tapmatrix_fixed_sum`): 8 samples per vector, per tap and point one broadcast coefficient × one unaligned load × `esp.vmulas.s16.qacc`, all taps of the output in one accumulator, one saturating store. The moving taps of an output are summed by a second kernel (`tapmatrix_fixed_moving`): per 8 samples it computes each lane's fraction q, the coefficients b = q + Q/2, c = (1−q) + P/2 − Q/2, a = −(P+Q)/6, d = −(2P−Q)/6 (P = q(1−q), Q = Pq), multiplies them by the gain ramp and accumulates them over a 5-point window. Lane masks pick points 0..3 or 1..4 where the read position steps by a sample within the 8. The C side only prepares, per 8 samples, a window pointer, the fraction offset and the gain. The lane offsets and the gain ramp stay the same for a whole ramp and are computed once, at its first block.

## Source layout

| file | what |
|---|---|
| `tapmatrix~.c` | the object: state, messages, the block loop (which taps move, which engine) |
| `arch/platform.h` | timer for `stats` and scratch memory: ESP32, Windows, POSIX |
| `arch/scalar.h` | the portable kernels in `t_sample` (float or double): coefficients, static FIR, moving tap; `-oversample`: the halfband, `fir2x2`, `moving_os` |
| `arch/neon.h` | the same with NEON (armv7, arm64), single precision, 4 lanes; explains the moving kernel |
| `arch/avx2.h` | the same with AVX2 + FMA (x86), single precision, 8 lanes, chosen at run time |
| `arch/fixed.h` | the `fixed` engine: its data layout, the C kernels (hosts), the declarations of the PIE kernels, a moving tap's preparation |
| `arch/fixed_pie.S` | the PIE kernels (ESP32-P4), with a guide to the PIE instructions they use |
| `tapmatrix~.pd`, `tapmatrix.*~.pd` | the abstraction fallback |

The kernels are `static` functions in headers, so the build stays one translation unit. A new SIMD architecture is one more header like `arch/neon.h`, included in `tapmatrix~.c` and called where the NEON kernels are. The PIE kernels are assembly (no intrinsics for PIE in GCC); `arch/fixed.h` has the same arithmetic in C, which runs on hosts as the `fixed` engine's reference.

## Tests

`tests/msgtest.py` runs Pd in batch mode (`PD=/path/to/pd python3 tests/msgtest.py`, needs numpy): `row`
and `rowl` must give bit-identical output to the same cells set by `tap`; `jump` is checked sample by sample
against its model (a linear ramp as input, exact for the 4-point interpolation); `get` and `clear <in>`.
`TM_PATH=<folder>` tests another `tapmatrix~`, e.g. the fallback alone (a folder with only the `.pd` files).

`tests/ostest.py` (Pd batch, like `msgtest.py`; the external only): `-oversample` against the plain interpolation, frequency responses and the delay of static taps, the level of a 15 kHz sine in a moving tap.

`tests/kerneltest.c` checks the SIMD float kernels of the build machine (NEON or AVX2) against the scalar ones, 20 000 random static and moving taps, also at the ends of the ring and at the kernels' |dd| limits; it needs only Pd's headers (`make -C tests kerneltest PDINCLUDEDIR=…`).

`tests/` also runs two patches with [libpd](https://github.com/libpd/libpd) and loads the built external as Pd does: `tm.pd` (float engine) and `tmfixed.pd` (fixed engine, source at −6 dBFS). Each compares a static and a moving tap with Pd's own `delwrite~` → `line~` → `delread4~` → `*~` chain and with an ideal double-precision computation (`ana.py`, needs numpy).

```sh
make PDINCLUDEDIR=/path/to/libpd/pure-data/src
make test PDINCLUDEDIR=/path/to/libpd/pure-data/src LIBPD_DIR=/path/to/libpd   # a built libpd
```

## License

GNU General Public License 3, see [LICENSE.txt](LICENSE.txt).
