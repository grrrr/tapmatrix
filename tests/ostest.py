#!/usr/bin/env python3
"""-oversample against the plain 4-point interpolation (Pd batch, needs numpy and the external:
the abstraction fallback has no -oversample).

    PD=/path/to/pd python3 tests/ostest.py      (TM_PATH=<dir>: where tapmatrix~ is)

* static taps: frequency response from the impulse response, for delay fractions 0, 0.25, 0.5;
  the delay (group delay at 1 kHz) must be the one asked for
* a moving tap: a 15 kHz sine through a delay ramp of 20 samples; the plain interpolation
  modulates its amplitude with the fraction, -oversample keeps it flat
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msgtest import OUT, SR, P, run  # noqa: E402

FRACS = (0.0, 0.25, 0.5)
D0 = 40                             # samples
IMP = int(0.1 * SR)                 # the impulse's sample (at 100 ms, a block boundary)


def static_case(os_flag):
    p = P()
    sg = p.obj("sig~ 0")            # an impulse: a step, differentiated (one sample of 1)
    imp = p.obj("rzero~ 1")
    p.c(sg, 0, imp, 0)
    t = p.obj(f"tapmatrix~ {os_flag} 1 {len(FRACS)} 100".replace("  ", " "))
    p.c(imp, 0, t, 0)
    r = p.obj("r ctl")
    p.c(r, 0, t, 0)
    so = p.obj(f"snake~ out {len(FRACS)}")
    p.c(t, 0, so, 0)
    taps = "; ".join(f"ctl tap 0 {k} {(D0 + f) / SR * 1000!r} 1 0" for k, f in enumerate(FRACS))
    sched = [(10, "; " + taps), (100, "; imp 1")]
    ri = p.obj("r imp")
    p.c(ri, 0, sg, 0)
    x, out = run(p, f"os-static{'-os' if os_flag else ''}", len(FRACS), 12000, sched,
                 lambda tws: [p.c(so, k, tws[k], 0) for k in range(len(FRACS))])
    return x


def response(x, k):
    """magnitude (dB) at 1, 12, 16, 20 kHz and the delay at 1 kHz (samples) of channel k"""
    y = x[:, k]
    i0 = int(np.argmax(np.abs(y))) - 64
    seg = y[i0:i0 + 256]
    seg = seg / max(np.sum(seg), 1e-12)                 # DC gain 1 (the impulse's height is arbitrary)
    H = np.fft.rfft(seg, 4096)
    fr = np.fft.rfftfreq(4096, 1 / SR)
    mags = [20 * np.log10(abs(H[np.argmin(abs(fr - f))])) for f in (1000, 12000, 16000, 20000)]
    k1 = np.argmin(abs(fr - 1000))
    ph = np.unwrap(np.angle(H[:k1 + 2]))
    gd = -(ph[k1 + 1] - ph[k1 - 1]) / (2 * np.pi * (fr[k1 + 1] - fr[k1 - 1])) * SR + i0 - IMP
    return mags, gd


def moving_case(os_flag):
    p = P()
    osc = p.obj("osc~ 15000")
    t = p.obj(f"tapmatrix~ {os_flag} 1 1 100".replace("  ", " "))
    p.c(osc, 0, t, 0)
    r = p.obj("r ctl")
    p.c(r, 0, t, 0)
    sched = [(10, f"; ctl tap 0 0 {D0 / SR * 1000!r} 1 0"),
             (100, f"; ctl tap 0 0 {(D0 + 20) / SR * 1000!r} 1 500")]
    x, _ = run(p, f"os-moving{'-os' if os_flag else ''}", 1, 36000, sched, lambda tws: p.c(t, 0, tws[0], 0))
    y = x[int(0.15 * SR):int(0.55 * SR), 0]               # inside the ramp
    # the amplitude per 64 samples (a 15 kHz period is 3.2 samples: 20 periods)
    env = np.array([np.sqrt(2 * np.mean(y[i:i + 64] ** 2)) for i in range(0, len(y) - 64, 64)])
    return 20 * np.log10(env.max() / env.min()), 20 * np.log10(env.mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    res = {}
    for flag in ("", "-oversample"):
        x = static_case(flag)
        res[flag] = [response(x, k) for k in range(len(FRACS))]
    for k, f in enumerate(FRACS):
        (mp, gp), (mo, go) = res[""][k], res["-oversample"][k]
        flat = all(abs(m) < 0.6 for m in mo)
        dly = abs(go - (D0 + f)) < 0.01 and abs(gp - (D0 + f)) < 0.01
        ok &= flat and dly
        print(f"{'PASS' if flat and dly else 'FAIL'} static, fraction {f}: dB at 1/12/16/20 kHz plain "
              + " ".join(f"{m:+.2f}" for m in mp) + ", -oversample " + " ".join(f"{m:+.2f}" for m in mo)
              + f"; delay {gp:.3f} / {go:.3f} samples (want {D0 + f})")
    rp, ap = moving_case("")
    ro, ao = moving_case("-oversample")
    mok = ro < 0.2 and abs(ao) < 0.6
    ok &= mok
    print(f"{'PASS' if mok else 'FAIL'} moving 15 kHz sine, delay ramp {D0} -> {D0 + 20} samples: amplitude "
          f"ripple plain {rp:.2f} dB (mean {ap:+.2f} dB), -oversample {ro:.2f} dB (mean {ao:+.2f} dB)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
