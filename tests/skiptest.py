#!/usr/bin/env python3
"""Silent moving taps are skipped (Pd batch, needs numpy and the external: the abstraction
fallback has no `skip`).

    PD=/path/to/pd python3 tests/skiptest.py    (TM_PATH=<dir>: where tapmatrix~ is)

A moving tap whose gain is 0 at both ends of a block is not computed (`skip 1`, the default);
it only goes on to the next block as the moving kernel would. Here two [tapmatrix~ 2 4] get the
same input and the same rows, one with `skip 0` (every moving tap computed: the reference).
The rows come every 20 ms with a 20 ms ramp (a moving WFS source: every cell ramps all the time):
first all gains on, then the cells 1..3 of both inputs ramp to 0 and stay there while their
delays go on changing (a silent tap jumps to its new delay: its ramp moves the gain 0), then they ramp up again; cell 0 stays on throughout, one cell moves
fast enough for the scalar path (|dd| > 1/3 sample per sample). The outputs must be
bit-identical, for the float engine, -fixed and -oversample. `stats` during the silent part
shows that the skip happened: fewer moving taps computed.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msgtest import OUT, SR, P, errors, run, source  # noqa: E402

FRAMES = int(0.8 * SR)
UPDATE = 20                         # ms between rows, also their ramp


def rows():
    """[(ms, [[(delay-ms, gain)] * 4] * 2)]: the rows of both inputs over time"""
    out = []
    for k, ms in enumerate(range(10, 760, UPDATE)):
        on = ms < 200 or ms >= 520  # cells 1..3: on, then silent, then on again
        vals = []
        for i in range(2):
            cells = []
            for o in range(4):
                d = 5 + 3 * o + 2 * i + 0.37 * k * (1 if o % 2 else -0.6) + 0.5 * np.sin(0.3 * k + o)
                if o == 3 and i == 1:
                    d = 10 + (9 if k % 2 else 0)   # 9 ms in 20 ms: > 1/3 sample per sample
                g = 0.5 - 0.1 * o + 0.2 * i if (o == 0 or on) else 0
                cells.append((round(d, 4), round(g, 4)))
            vals.append(cells)
        out.append((ms, vals))
    return out


def case(flags):
    p = P()
    sn = source(p)
    tms, sns, prs = [], [], []
    for k in range(2):
        t = p.obj(f"tapmatrix~ {flags} 2 4 100".replace("  ", " "))
        p.c(sn, 0, t, 0)
        r = p.obj(f"r ctl{k}")
        p.c(r, 0, t, 0)
        s = p.obj("snake~ out 4")
        p.c(t, 0, s, 0)
        pr = p.obj(f"print STATS{k}")
        p.c(t, 1, pr, 0)
        tms.append(t)
        sns.append(s)
    sched = [(1, "; ctl1 skip 0")]
    for ms, vals in rows():
        msg = "; ".join(f"ctl{k} rowl {i} {UPDATE} " + " ".join(f"{d} {g}" for d, g in vals[i])
                        for k in range(2) for i in range(2))
        sched.append((ms, "; " + msg))
    sched.append((260, "; ctl0 stats 150; ctl1 stats 150"))   # within the silent part (200..520 ms)

    def conn(tws):
        for k in range(2):
            for o in range(4):
                p.c(sns[k], o, tws[4 * k + o], 0)
    name = "skip" + flags.replace("-", "_")
    x, out = run(p, name, 8, FRAMES, sched, conn)
    stats = {}
    for ln in out:
        for k in range(2):
            if ln.startswith(f"STATS{k}:"):
                w = [v for v in ln.split()[1:] if v not in ("list", "stats")]
                stats[k] = [float(v) for v in w]
    a, b = x[:, 0:4], x[:, 4:8]
    silent = slice(int(0.25 * SR), int(0.5 * SR))
    mv = [s[0] + s[1] for s in (stats.get(0, [0, 0]), stats.get(1, [0, 0]))]
    ok = (np.array_equal(a, b) and np.abs(a).max() > 0.1 and np.abs(a[silent, 1:]).max() == 0
          and np.abs(a[int(0.6 * SR):, 1:]).max() > 0.1
          and len(stats) == 2 and mv[0] < mv[1] and not errors(out))
    print(f"{'PASS' if ok else 'FAIL'} skip {flags or '-float'}: max |skip - computed| {np.abs(a - b).max():.2e} "
          f"(peak {np.abs(a).max():.2f}); moving taps computed per block {mv[0]:.1f} / {mv[1]:.1f} without skip",
          errors(out)[:2])
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for flags in ("", "-fixed", "-oversample"):
        ok &= case(flags)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
