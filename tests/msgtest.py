#!/usr/bin/env python3
"""Pd batch tests of the row messages: row, rowl, jump, get, clear <in>.

    PD=/path/to/pd python3 tests/msgtest.py     (needs numpy; the external built in this folder;
                                                 TM_PATH=<dir>: another tapmatrix~, e.g. an abstraction)

* row and rowl give bit-identical output to the same cells set with tap (static and ramping).
* jump: with a linear ramp as input (exact for the 4-point interpolation) the output is
  y[n] = g[n] * (x[n] - d[n]): gains fade to 0 over xfade with the old delays, the delays jump at the
  turning point, gains fade in over xfade. A row sent during the fade-out sets the targets after it.
* get: after a ramp the current values equal the targets.
* clear <in>: only that input's row goes silent.
* first-out: row and jump with a first output set the cells from there on (and jump keeps the others).
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PD = os.environ.get("PD", "pd")
TM_PATH = os.environ.get("TM_PATH", ROOT)  # where tapmatrix~ is found (the external here, or an abstraction)
TM_TOL = 1e-5  # jump: |y - model| relative to the peak
SR, N = 48000, 64
OUT = os.path.join(HERE, "out")


def esc(t):
    return t.replace("$", "\\$").replace(",", " \\,").replace(";", " \\;")


class P:
    """minimal patch writer"""

    def __init__(self):
        self.recs, self.n = [], 0

    def add(self, kind, text):
        self.recs.append(f"#X {kind} 10 {10 + 20 * self.n} {esc(text)};")
        self.n += 1
        return self.n - 1

    def obj(self, t):
        return self.add("obj", t)

    def msg(self, t):
        return self.add("msg", t)

    def c(self, a, o, b, i):
        self.recs.append(f"#X connect {a} {o} {b} {i};")

    def save(self, path):
        with open(path, "w") as f:
            f.write("#N canvas 0 0 800 600 12;\n" + "\n".join(self.recs) + "\n")


def read_wav(path):
    data = open(path, "rb").read()
    pos, fmt, raw = 12, None, b""
    while pos < len(data) - 8:
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body if 0 < size <= len(data) - pos - 8 else data[pos + 8:]
            break
        pos += 8 + size + (size & 1)
    nch = fmt[1]
    n = len(raw) // (4 * nch)
    return np.frombuffer(raw[:n * 4 * nch], dtype="<f4").reshape(-1, nch).astype(float)


def run(p, name, nch, frames, sched, conn, stderr=False):
    """sched: [(ms, message box text)]; conn(tws) connects the signals to record to the tabwrite~
    objects (one per channel; `frames` samples from the first block) -> (signals, Pd output lines)"""
    lb = p.obj("loadbang")
    p.c(lb, 0, p.msg("; pd dsp 1"), 0)
    tws = []
    for k in range(nch):
        p.obj(f"array define {name}-{k} {frames}")
        tws.append(p.obj(f"tabwrite~ {name}-{k}"))
        p.c(lb, 0, tws[-1], 0)
    conn(tws)
    for ms, text in sched:
        d = p.obj(f"del {ms}")
        p.c(lb, 0, d, 0)
        p.c(d, 0, p.msg(text), 0)
    dq = p.obj(f"del {frames / SR * 1000 + 50}")
    p.c(lb, 0, dq, 0)
    wav = os.path.join(OUT, f"{name}.wav")
    arrs = " ".join(f"{name}-{k}" for k in range(nch))
    w = p.msg(f"write -wave -bytes 4 {wav} {arrs}")
    p.c(dq, 0, w, 0)
    sf = p.obj("soundfiler")
    p.c(w, 0, sf, 0)
    q = p.msg("; pd quit")
    p.c(sf, 0, q, 0)
    path = os.path.join(OUT, f"{name}.pd")
    p.save(path)
    r = subprocess.run([PD, "-nogui", "-noaudio", "-batch", "-r", str(SR), "-stderr", "-path", TM_PATH, path],
                       capture_output=True, text=True, timeout=60)
    return read_wav(wav), (r.stdout + r.stderr).splitlines()


def errors(lines):
    return [ln for ln in lines if "error" in ln]


def source(p, ramp=False):
    """2-channel input: noise and a sine, or two linear ramps x[n] = n+1, 2(n+1)"""
    if ramp:
        a = p.obj("sig~ 1")
        b = p.obj("sig~ 2")
        ra, rb = p.obj("rpole~ 1"), p.obj("rpole~ 1")
        p.c(a, 0, ra, 0)
        p.c(b, 0, rb, 0)
        s0, s1 = ra, rb
    else:
        s0, s1 = p.obj("noise~"), p.obj("osc~ 1234")
    sn = p.obj("snake~ in 2")
    p.c(s0, 0, sn, 0)
    p.c(s1, 0, sn, 1)
    return sn


def test_row():
    """two matrices, the same cells: one by tap, one by row (arrays) and one by rowl"""
    p = P()
    sn = source(p)
    tms = []
    for k in range(3):
        t = p.obj("tapmatrix~ 2 4 100")
        p.c(sn, 0, t, 0)
        r = p.obj(f"r ctl{k}")
        p.c(r, 0, t, 0)
        tms.append(t)
    for a in ("d0", "g0", "d1", "g1"):
        p.obj(f"array define {a} 4")
    v = [[(3.3, 0.5), (7.25, -0.2), (12.0, 1.0), (1.5, 0.3)], [(20.1, 0.7), (0.02, 0.4), (5.5, 0.0), (9.9, -1.0)]]
    v2 = [[(13.3, 0.1), (2.25, 0.6), (2.0, -0.5), (31.5, 0.9)], [(1.1, 0.2), (8.0, 0.8), (15.5, 0.3), (0.5, 0.5)]]

    def taps(vals, ramp):
        return "; ".join(f"ctl0 tap {i} {o} {d} {g} {ramp}" for i in range(2) for o, (d, g) in enumerate(vals[i]))

    def arrays(vals):
        return "; ".join(f"d{i} 0 {' '.join(str(d) for d, _ in vals[i])}; g{i} 0 {' '.join(str(g) for _, g in vals[i])}"
                         for i in range(2))

    def rows(vals, ramp):
        return "; ".join(f"ctl1 row {i} {ramp} d{i} g{i}" for i in range(2))

    def rowls(vals, ramp):
        return "; ".join(f"ctl2 rowl {i} {ramp} " + " ".join(f"{d} {g}" for d, g in vals[i]) for i in range(2))
    sched = [(10, "; " + taps(v, 0) + "; " + arrays(v) + "; " + rows(v, 0) + "; " + rowls(v, 0)),
             (200, "; " + taps(v2, 150) + "; " + arrays(v2) + "; " + rows(v2, 150) + "; " + rowls(v2, 150))]
    sns = [p.obj("snake~ out 4") for _ in range(3)]
    for t, s in zip(tms, sns):
        p.c(t, 0, s, 0)
    def conn(tws):
        for k in range(3):
            for o in range(4):
                p.c(sns[k], o, tws[4 * k + o], 0)
    x, out = run(p, "row", 12, 24000, sched, conn)
    errs = errors(out)
    a, b, c = x[:, 0:4], x[:, 4:8], x[:, 8:12]
    ok = np.array_equal(a, b) and np.array_equal(a, c) and np.abs(a).max() > 0.1
    print(f"{'PASS' if ok else 'FAIL'} row / rowl = tap: max |row - tap| {np.abs(a - b).max():.2e}, "
          f"|rowl - tap| {np.abs(a - c).max():.2e} (peak {np.abs(a).max():.2f})", errs[:2])
    return ok


def build(name, sched, nch_out=2):
    """one tapmatrix~ 2 x nch_out, linear ramps as input; arrays d, g, d2, g2, gd, gg (size nch_out)"""
    p = P()
    sn = source(p, ramp=True)
    t = p.obj(f"tapmatrix~ 2 {nch_out} 100")
    p.c(sn, 0, t, 0)
    r = p.obj("r ctl")
    p.c(r, 0, t, 0)
    for a in ("d", "g", "d2", "g2", "gd", "gg"):
        p.obj(f"array define {a} {nch_out}")
    so = p.obj(f"snake~ out {nch_out}")
    p.c(t, 0, so, 0)
    x, out = run(p, name, nch_out, 12000, sched, lambda tws: [p.c(so, o, tws[o], 0) for o in range(nch_out)])
    return x, errors(out)


def model(blocks, x, events):
    """per-block tap state machine for one cell: events = {block: (kind, d, g, b)}; kind 'set'
    (ramp over b blocks; silent: jump) -> per-sample y = g * (x - d)"""
    d = g = 0.0
    nb = left = 0
    df = dt = gf = gt = 0.0
    y = np.zeros(blocks * N)
    for k in range(blocks):
        if k in events:
            for kind, dd, gg, b in events[k]:
                if g == 0:
                    d = dd
                df, gf, dt, gt = d, g, dd, gg
                if b < 1:
                    d, g, left = dd, gg, 0
                else:
                    nb = left = b
        n0 = k * N
        if left:
            d0, g0 = d, g
            left -= 1
            u = (nb - left) / nb
            d = df + (dt - df) * u if left else dt
            g = gf + (gt - gf) * u if left else gt
            i = np.arange(N)
            dd_ = d0 + (d - d0) * i / N
            gg_ = g0 + (g - g0) * i / N
        else:
            dd_, gg_ = np.full(N, d), np.full(N, g)
        y[n0:n0 + N] = gg_ * (x[n0:n0 + N] - dd_)
    return y


def test_jump():
    """row at 10 ms (static), jump at 100 ms (xfade 20 ms), a row 5 ms into the dip changes the target"""
    ms = 1000 / SR
    xb = round(20 / 1000 * SR / N)  # 15 blocks
    sched = [(10, "; d 0 5 9; g 0 0.5 0.8; ctl row 0 0 d g"),
             (100, "; d2 0 21.5 3.25; g2 0 0.9 0.3; ctl jump 0 20 d2 g2"),
             (105, "; d2 0 30.25 4.5; g2 0 0.6 1; ctl row 0 50 d2 g2")]
    x, errs = build("jump", sched)
    n = np.arange(len(x)) + 1.0  # input 0: x[n] = n + 1 (rpole~ of 1); first sample 1
    blocks = len(x) // N
    # the block each message lands in: from the output itself (first deviation)
    y0 = x[:, 0]
    k_row = int(np.argmax(np.abs(y0) > 0)) // N
    events = {k_row: [("set", 5 / ms, 0.5, 0)]}
    pre = model(blocks, n, events)
    k_jump = int(np.argmax(np.abs(y0[:blocks * N] - pre) > 1e-3)) // N
    events[k_jump] = [("set", 5 / ms, 0.0, xb)]
    events[k_jump + xb] = [("set", 30.25 / ms, 0.6, xb)]   # the row during the dip wins
    want = model(blocks, n, events)
    err = np.abs(y0[:blocks * N] - want)
    ok = err.max() < TM_TOL * np.abs(want).max() + 1e-3
    print(f"{'PASS' if ok else 'FAIL'} jump: blocks row {k_row}, jump {k_jump} (+{xb} fade), "
          f"max |y - model| {err.max():.2e} of peak {np.abs(want).max():.0f}", errs[:2])
    return ok


def test_get_clear():
    """get after a ramp = targets (ms); clear 0 silences only input 0's row"""
    p = P()
    sn = source(p)
    t = p.obj("tapmatrix~ 2 3 100")
    p.c(sn, 0, t, 0)
    r = p.obj("r ctl")
    p.c(r, 0, t, 0)
    for a in ("d", "g", "gd", "gg"):
        p.obj(f"array define {a} 3")
    pr = p.obj("print GOT")
    rq = p.obj("r report")
    for a in ("gd", "gg"):
        ga = p.obj(f"array get {a}")
        lp = p.obj(f"list prepend {a}")
        p.c(rq, 0, ga, 0)
        p.c(ga, 0, lp, 0)
        p.c(lp, 0, pr, 0)
    so = p.obj("snake~ out 3")
    p.c(t, 0, so, 0)
    sched = [(10, "; d 0 4 8.5 12.25; g 0 0.25 0.5 -0.75; ctl row 0 100 d g; ctl row 1 0 d g"),
             (200, "; ctl get 0 gd gg; report bang"),
             (250, "; ctl clear 0")]
    x, out = run(p, "getclear", 3, 18000, sched, lambda tws: [p.c(so, o, tws[o], 0) for o in range(3)])
    got = {}
    for ln in out:
        if ln.startswith("GOT:"):
            w = [v for v in ln.split()[1:] if v != "list"]
            got[w[0]] = [float(v) for v in w[1:]]
    ok_get = (np.allclose(got.get("gd", []), [4, 8.5, 12.25], atol=1e-3)
              and np.allclose(got.get("gg", []), [0.25, 0.5, -0.75]))
    # after clear 0 (at 250 ms) only input 1 (the sine) remains: compare with a run without input 0
    tail = x[int(0.3 * SR):]
    sine_only = np.abs(np.fft.rfft(tail[:, 0] * np.hanning(len(tail))))
    peak = np.argmax(sine_only) * SR / len(tail)
    ok_clear = abs(peak - 1234) <= SR / len(tail) and np.abs(tail).max() > 0.1  # (FFT bin width)
    # and nothing of input 0 (noise) left: the sine's harmonics-free spectrum, energy near 1234 Hz
    spec = sine_only ** 2
    near = spec[max(int(np.argmax(sine_only)) - 3, 0):int(np.argmax(sine_only)) + 4].sum()
    ok_clear &= near / spec.sum() > 0.99
    print(f"{'PASS' if ok_get else 'FAIL'} get: {got}")
    print(f"{'PASS' if ok_clear else 'FAIL'} clear 0: input 1 remains (spectral peak {peak:.0f} Hz)")
    return ok_get and ok_clear


def test_first():
    """row ... 2 and jump ... 1 (first-out) = the same cells set with tap"""
    p = P()
    sn = source(p)
    tms = []
    for k in range(2):
        t = p.obj("tapmatrix~ 2 4 100")
        p.c(sn, 0, t, 0)
        r = p.obj(f"r ctl{k}")
        p.c(r, 0, t, 0)
        tms.append(t)
    for a in ("d", "g", "d2", "g2"):
        p.obj(f"array define {a} 2")
    sched = [(10, "; d 0 3.3 7.25; g 0 0.5 -0.2; ctl1 row 0 0 d g 2"
                  "; ctl0 tap 0 2 3.3 0.5 0; ctl0 tap 0 3 7.25 -0.2 0"),
             (100, "; d2 0 12 1.5; g2 0 0.9 0.3; ctl1 jump 0 20 d2 g2 1"
                   "; ctl0 tap 0 1 12 0.9 0; ctl0 tap 0 2 1.5 0.3 0")]
    sns = [p.obj("snake~ out 4") for _ in range(2)]
    for t, s in zip(tms, sns):
        p.c(t, 0, s, 0)

    def conn(tws):
        for k in range(2):
            for o in range(4):
                p.c(sns[k], o, tws[4 * k + o], 0)
    x, out = run(p, "first", 8, 12000, sched, conn)
    errs = errors(out)
    a, b = x[:, 0:4], x[:, 4:8]
    pre = slice(int(0.03 * SR), int(0.09 * SR))   # after the row, before the jump
    post = slice(int(0.16 * SR), None)            # after the jump's fade-in
    d_pre, d_post = np.abs(a[pre] - b[pre]).max(), np.abs(a[post] - b[post]).max()
    ok = d_pre < 1e-6 and d_post < 1e-6 and np.abs(b[pre, 2:]).max() > 0.1 and np.abs(b[post, 1:]).max() > 0.1
    print(f"{'PASS' if ok else 'FAIL'} first-out: max |row/jump - tap| {d_pre:.2e} / {d_post:.2e}", errs[:2])
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = test_row() & test_jump() & test_get_clear() & test_first()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
