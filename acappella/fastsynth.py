"""Compiled versions of the renderer's per-sample work (needs numba).

Singer.render() spends most of its time in a few dozen whole-song numpy
passes at the audio rate: turning the control tracks into pitch, glottal
phase and voice quality, reading the glottal wavetables, mixing in breath
noise, and running the formant cascade; render_score's mixing makes a few
more.  These kernels do the same arithmetic in as few passes as they can,
most of them on every core at once (parallel() shares the cores out among
the voices being sung), so a voice renders many times faster and needs a
fraction of the memory.  Random numbers are still drawn by numpy, in the
same order, so a render is the same as without numba to within rounding
(a few units in the last place).

Without numba, synth.py and render.py use their numpy code and this
module isn't loaded.
"""
from __future__ import annotations

import math
import threading

import numba
import numpy as np
from numba import njit, prange

HARMONIC_LIMITS = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
STEP = 4096           # samples per glottal-table choice (and wobble re-anchoring)
MIX_CHUNK = 65536     # samples per piece of the mix, when the cores share it out

# How many voices are being sung right now, across render.py's worker
# processes (a multiprocessing.Value), so each takes its share of the cores.
singing = None
_TURN = threading.Lock()


def parallel(kernel, *args):
    """Run one of the parallel kernels (glottal_phase, excitation, cascade,
    level_powers) on this voice's share of the cores: twice an even share,
    since the voices are rarely all in a parallel kernel at the same moment.
    numba's thread pool runs one parallel kernel at a time, so voices in one
    process take turns."""
    with _TURN:
        n = max(1, singing.value) if singing is not None else 1
        cores = numba.config.NUMBA_NUM_THREADS
        numba.set_num_threads(min(cores, max(1, 2 * cores // n)))
        return kernel(*args)


def table_stack(tabs, q_levels):
    """synth's {quality: {harmonic limit: table}} as one float32 array."""
    return np.ascontiguousarray(
        np.stack([np.stack([tabs[q][h] for h in HARMONIC_LIMITS]) for q in range(q_levels)]),
        dtype=np.float32)


@njit(cache=True, nogil=True, inline="always")
def _lerp(arr, pos):
    """np.interp(pos, arange(len(arr)), arr) for one value."""
    n = arr.shape[0]
    if pos <= 0.0:
        return arr[0]
    if pos >= n - 1:
        return arr[n - 1]
    j = int(pos)
    return (arr[j + 1] - arr[j]) * (pos - j) + arr[j]


@njit(cache=True, nogil=True, inline="always")
def _lf(pts, i, n, step):
    """Sample i of np.interp(np.linspace(0, k - 1, n), arange(k), pts);
    step = (k - 1) / (n - 1)."""
    if i == n - 1:
        return pts[pts.shape[0] - 1]
    return _lerp(pts, i * step)


@njit(cache=True, nogil=True, inline="always")
def _at(arr, j, d):
    """np.interp at control position j + d (0 <= d < 1, j inside arr)."""
    return (arr[j + 1] - arr[j]) * d + arr[j]


@njit(cache=True, nogil=True, parallel=True)
def glottal_phase(n_samples, sr, cr, midi, vib, cents_c, vrate, u1, u2, p12, p40, p60,
                  phase_out, fmax_out):
    """The pitch, sample by sample (Singer.render's sample-rate section up
    to the glottal phase): fills phase_out with the glottis's running phase
    (in cycles, not yet wrapped to 0..1) and fmax_out with the highest pitch
    in each STEP-sample chunk.

    Two running sums tie each sample to the one before (the vibrato's phase
    and the glottis's), so the song goes in stretches: within one, what
    feeds a sum is worked out for all its samples at once on every core,
    the sum runs through in order, and so on."""
    c_rate = 2.0 * math.pi * 0.23
    big = 64 * STEP                       # samples per stretch
    n_ctrl = midi.shape[0]
    s12 = (p12.shape[0] - 1.0) / (n_samples - 1.0)
    s40 = (p40.shape[0] - 1.0) / (n_samples - 1.0)
    s60 = (p60.shape[0] - 1.0) / (n_samples - 1.0)
    ln2_12 = math.log(2.0) / 12.0
    inv_sr = 1.0 / sr
    dphi = 2.0 * math.pi * inv_sr
    rot_c = math.cos(c_rate * inv_sr)     # one sample of the 0.23 Hz wobble
    rot_s = math.sin(c_rate * inv_sr)
    inc = np.empty(big)                   # what each sample adds to a running sum
    acc = np.empty(big)
    vph = 0.0
    ph = 0.0
    for b0 in range(0, n_samples, big):
        b1 = min(n_samples, b0 + big)
        nch = (b1 - b0 + STEP - 1) // STEP
        # 1. the vibrato rate (with its slow wobble, exact at each chunk start)
        for c in prange(nch):
            s = b0 + c * STEP
            e = min(b1, s + STEP)
            w_s = math.sin(c_rate * (s / sr) + u1)
            w_c = math.cos(c_rate * (s / sr) + u1)
            for i in range(s, e):
                j, d = _ctrl_pos(i, sr, cr, n_ctrl)
                rate = _at(vrate, j, d) * (1 + 0.04 * w_s)
                w_s, w_c = w_s * rot_c + w_c * rot_s, w_c * rot_c - w_s * rot_s
                inc[i - b0] = rate * dphi
        # 2. the vibrato's phase, in order
        for i in range(b0, b1):
            vph += inc[i - b0]
            acc[i - b0] = vph + u2
        # 3. the pitch, as what it adds to the glottal phase per sample
        for c in prange(nch):
            s = b0 + c * STEP
            e = min(b1, s + STEP)
            fm = -1e300
            for i in range(s, e):
                j, d = _ctrl_pos(i, sr, cr, n_ctrl)
                drift = _lf(p12, i, n_samples, s12) * 5.0 + _lf(p40, i, n_samples, s40) * 1.5
                jitter = _lf(p60, i, n_samples, s60) * 3.0
                cents = (_at(vib, j, d) * math.sin(acc[i - b0]) + _at(cents_c, j, d)
                         + drift + jitter)
                f0 = 440.0 * math.exp((_at(midi, j, d) - 69 + cents / 100.0) * ln2_12)
                inc[i - b0] = f0 * inv_sr
                if f0 > fm:
                    fm = f0
            fmax_out[s // STEP] = fm
        # 4. the glottis's phase, in order
        for i in range(b0, b1):
            ph += inc[i - b0]
            phase_out[i] = ph


@njit(cache=True, nogil=True, parallel=True)
def excitation(n_samples, sr, cr, av_c, ah_c, open_c, lvl_c, qoff_c, p30, noise, breathiness,
               tables, close_phase, block, fmax, x_out, gain_out, active_out):
    """The glottal source and noise, sample by sample (the rest of
    Singer.render's sample-rate section up to the cascade), for every
    STEP-sample chunk at once.  x_out comes in holding glottal_phase's
    phase and goes out holding the excitation; also fills gain_out
    (10 ** (level / 20)) and active_out (which BLOCK-sample blocks have any
    sound)."""
    q_levels = tables.shape[0]
    table_n = tables.shape[2]
    n_ctrl = av_c.shape[0]
    s30 = (p30.shape[0] - 1.0) / (n_samples - 1.0)
    ln10_20 = math.log(10.0) / 20.0
    nblk = active_out.shape[0]
    for b in range(nblk):
        active_out[b] = False
    for c in prange((n_samples + STEP - 1) // STEP):
        s = c * STEP
        e = min(n_samples, s + STEP)
        qb = np.empty(e - s)
        qmin = 1e300
        qmax = -1e300
        for i in range(s, e):
            j, d = _ctrl_pos(i, sr, cr, n_ctrl)
            av = _at(av_c, j, d)
            ah = _at(ah_c, j, d)
            lvl = _at(lvl_c, j, d)
            q = min(max((lvl + 22) / 26, 0.0), 1.0)
            q = min(max(q - 0.45 * (1.0 - _at(open_c, j, d)) + _at(qoff_c, j, d), 0.0), 1.0)
            qb[i - s] = q
            gain_out[i] = math.exp(lvl * ln10_20)
            qp = q * (q_levels - 1)
            if qp < qmin:
                qmin = qp
            if qp > qmax:
                qmax = qp
            blk = i // block
            if blk < nblk and abs(av) + abs(ah) > 1e-5:
                active_out[blk] = True
        # the richest table whose harmonics stay under 0.45 * sr here
        hmax = int(0.45 * sr / max(fmax[c], 30.0))
        hi_idx = 11                     # 1 harmonic, if nothing richer fits
        lim = 2048
        for k in range(12):
            if lim <= hmax:
                hi_idx = k
                break
            lim //= 2
        lo_q = int(math.floor(qmin))
        hi_q = min(q_levels - 1, int(math.ceil(qmax)))
        for i in range(s, e):
            j, d = _ctrl_pos(i, sr, cr, n_ctrl)
            ph = x_out[i]
            phase = ph - math.floor(ph)     # ph % 1.0 (the same, exactly: ph >= 0)
            pos = phase * table_n
            i0 = int(pos) % table_n
            frac = pos - math.floor(pos)
            i1 = (i0 + 1) % table_n
            q = qb[i - s]
            qp = q * (q_levels - 1)
            glot = 0.0
            for qi in range(lo_q, hi_q + 1):
                w = min(max(1.0 - abs(qp - qi), 0.0), 1.0)
                tab = tables[qi, hi_idx]
                glot += w * (np.float64(tab[i0]) * (1 - frac) + np.float64(tab[i1]) * frac)
            shimmer = 1 + 0.04 * _lf(p30, i, n_samples, s30)
            nz = noise[i]
            om = (1.0 if (phase > close_phase - 0.6) and (phase < close_phase) else 0.0) * 0.7 + 0.3
            breath = nz * om * (breathiness * (1.4 - 0.8 * q))
            x_out[i] = _at(av_c, j, d) * (glot * shimmer + breath) + _at(ah_c, j, d) * nz * 2.0


@njit(cache=True, nogil=True, inline="always")
def _ctrl_pos(i, sr, cr, n_ctrl):
    """Sample i's position on the control tracks, as (frame, fraction)."""
    fpos = i / sr * cr
    if fpos >= n_ctrl - 1:
        return n_ctrl - 2, 1.0
    j = int(fpos)
    return j, fpos - j


@njit(cache=True, nogil=True, inline="always")
def _reson(f, bw, T):
    """A formant resonator's coefficients (A, B, C) at f Hz, bandwidth bw."""
    C = -math.exp(-2 * math.pi * bw * T)
    Bc = 2 * math.exp(-math.pi * bw * T) * math.cos(2 * math.pi * f * T)
    return 1 - Bc - C, Bc, C


@njit(cache=True, nogil=True, parallel=True)
def cascade(x, F, B, NZ, sr, cr, ring, block, y):
    """The formant cascade: per block of `block` samples, 7 biquads in turn
    (scipy's direct form II transposed): the nasal pole and zero, then the 5
    formants, tuned to the control tracks F, B and NZ at the block's middle.
    Blocks that don't ring are silent and the filter is at rest after them,
    so the runs of ringing blocks between them are filtered all at once."""
    nb = ring.shape[0]
    T = 1.0 / sr
    fcap = 0.45 * sr
    first = np.empty(nb, np.int64)
    last = np.empty(nb, np.int64)
    runs = 0
    for b in range(nb):
        if ring[b]:
            if b == 0 or not ring[b - 1]:
                first[runs] = b
                runs += 1
            last[runs - 1] = b + 1
    pA, pB, pC = _reson(270.0, 100.0, T)        # the nasal pole never moves
    for r in prange(runs):
        c = np.empty((7, 6))
        z0 = np.zeros(7)
        z1 = np.zeros(7)
        for b in range(first[r], last[r]):
            mid = (b * block + block / 2) / sr * cr
            c[0, 0], c[0, 1], c[0, 2], c[0, 3], c[0, 4], c[0, 5] = pA, 0.0, 0.0, 1.0, -pB, -pC
            A, Bc, C = _reson(_lerp(NZ, mid), 100.0, T)
            c[1, 0], c[1, 1], c[1, 2], c[1, 3], c[1, 4], c[1, 5] = 1 / A, -Bc / A, -C / A, 1.0, 0.0, 0.0
            for k in range(5):
                A, Bc, C = _reson(min(_lerp(F[k], mid), fcap), _lerp(B[k], mid), T)
                c[2 + k, 0], c[2 + k, 1], c[2 + k, 2] = A, 0.0, 0.0
                c[2 + k, 3], c[2 + k, 4], c[2 + k, 5] = 1.0, -Bc, -C
            for i in range(b * block, b * block + block):
                v = x[i]
                for k in range(7):
                    out = c[k, 0] * v + z0[k]
                    z0[k] = c[k, 1] * v - c[k, 4] * out + z1[k]
                    z1[k] = c[k, 2] * v - c[k, 5] * out
                    v = out
                y[i] = v


@njit(cache=True, nogil=True)
def finish(y, sos, fric, fric_gain, breath, gain, out):
    """The end of Singer.render in one pass: y through the presence shelf
    (scipy's sosfilt, from rest), plus the consonants (fric * fric_gain) and
    breaths, times the level, as float32 into out.  y is used up."""
    n = y.shape[0]
    nsec = sos.shape[0]
    for k in range(nsec):                 # section by section: the same sums
        b0, b1, b2, a1, a2 = sos[k, 0], sos[k, 1], sos[k, 2], sos[k, 4], sos[k, 5]
        z0 = 0.0
        z1 = 0.0
        if k < nsec - 1:
            for i in range(n):
                v = y[i]
                o = b0 * v + z0
                z0 = b1 * v - a1 * o + z1
                z1 = b2 * v - a2 * o
                y[i] = o
        else:
            for i in range(n):
                v = y[i]
                o = b0 * v + z0
                z0 = b1 * v - a1 * o + z1
                z1 = b2 * v - a2 * o
                o = o + fric[i] * fric_gain
                o = o + breath[i]
                out[i] = np.float32(o * gain[i])


@njit(cache=True, nogil=True, parallel=True)
def var_smooth_rows(X, W, passes):
    """var_smooth on every row of X at once."""
    out = np.empty_like(X)
    for r in prange(X.shape[0]):
        out[r] = var_smooth(X[r], W, passes)
    return out


@njit(cache=True, nogil=True)
def var_smooth(x, W, passes):
    """synth._var_smooth: a centred moving average of half-width W[i]
    (frames), `passes` times over (the same sums, in the same order)."""
    n = x.shape[0]
    pad = 0
    for i in range(n):
        pad = max(pad, W[i])
    pad += 1
    m = n + 2 * pad
    C = np.empty(m + 1)
    cur = x.copy()
    for _ in range(passes):
        first = cur[0]
        lastv = cur[n - 1]
        C[0] = 0.0
        C[1] = first                      # np.cumsum starts at the first value itself
        for k in range(1, m):
            v = first if k < pad else (cur[k - pad] if k < pad + n else lastv)
            C[k + 1] = C[k] + v
        for i in range(n):
            lo = i + pad - W[i]
            hi = i + pad + W[i] + 1
            cur[i] = (C[hi] - C[lo]) / np.float64(hi - lo)
    return cur


@njit(cache=True, nogil=True, parallel=True)
def level_powers(f0s, F, B, NZ, amps2, grid, T):
    """Singer._level_comp's sums, frame by frame: the power of the voice's
    harmonics through the vocal tract (Ph) and of a smooth spectrum with the
    same envelope (Pc)."""
    nf = f0s.shape[0]
    L = amps2.shape[0]
    Ph = np.empty(nf)
    Pc = np.empty(nf)
    dg = grid[1] - grid[0]
    for i in prange(nf):
        sA = np.empty(7)
        sB = np.empty(7)
        sC = np.empty(7)
        for k in range(7):
            if k == 0:
                f, bw = 270.0, 100.0
            elif k == 1:
                f, bw = NZ[i], 100.0
            else:
                f, bw = F[k - 2, i], B[k - 2, i]
            C = -math.exp(-2 * math.pi * bw * T)
            Bc = 2 * math.exp(-math.pi * bw * T) * math.cos(2 * math.pi * f * T)
            sA[k] = 1 - Bc - C
            sB[k] = Bc
            sC[k] = C
        ph = 0.0
        for h in range(1, 65):
            fh = f0s[i] * h
            if fh < 5000:
                ph += amps2[min(h - 1, L - 1)] * _mag2(fh, T, sA, sB, sC)
        pc = 0.0
        for g in range(grid.shape[0]):
            kk = grid[g] / f0s[i]
            if kk <= 1.0:
                ag = amps2[0]
            elif kk >= L:
                ag = amps2[L - 1]
            else:
                j = int(kk) - 1
                ag = (amps2[j + 1] - amps2[j]) * (kk - (j + 1)) + amps2[j]
            pc += ag * _mag2(grid[g], T, sA, sB, sC)
        Ph[i] = ph
        Pc[i] = pc * dg / f0s[i]
    return Ph, Pc


@njit(cache=True, nogil=True, inline="always")
def _mag2(freq, T, sA, sB, sC):
    w = 2 * math.pi * freq * T
    zr, zi = math.cos(w), -math.sin(w)            # z1 = e^(-iw)
    z2r, z2i = zr * zr - zi * zi, 2 * zr * zi     # z2 = z1 * z1
    out = 1.0
    for k in range(7):
        re = 1 - sB[k] * zr - sC[k] * z2r
        im = -sB[k] * zi - sC[k] * z2i
        d = re * re + im * im
        a2 = sA[k] * sA[k]
        out = out * (d / a2 if k == 1 else a2 / d)    # section 1 is the nasal zero
    return out


@njit(cache=True, nogil=True, parallel=True)
def place(mix, ys, scale, lg, rg, root2):
    """mix += pan(ys * scale): render_score's place(), the same arithmetic
    in one pass with no temporaries (scale comes in ys's own precision, as
    numpy multiplies a float32 track by a Python float)."""
    n = min(ys.shape[0], mix.shape[0])
    for c in prange((n + MIX_CHUNK - 1) // MIX_CHUNK):
        for i in range(c * MIX_CHUNK, min(n, c * MIX_CHUNK + MIX_CHUNK)):
            v = np.float64(ys[i] * scale)
            mix[i, 0] += v * lg * root2
            mix[i, 1] += v * rg * root2


@njit(cache=True, nogil=True, parallel=True)
def part_loudness(voices, gain, sr, cr, sos):
    """render_score's part calibration: the part's voices summed (in
    float32, as numpy sums float32 tracks), the dynamics curve divided back
    out (on every core), then RMS after the high-pass (in order), over the
    samples that sound."""
    n = voices.shape[1]
    nv = voices.shape[0]
    n_ctrl = gain.shape[0]
    x = np.empty(n)
    for c in prange((n + MIX_CHUNK - 1) // MIX_CHUNK):
        for i in range(c * MIX_CHUNK, min(n, c * MIX_CHUNK + MIX_CHUNK)):
            s = voices[0, i]
            for v in range(1, nv):
                s = np.float32(s + voices[v, i])
            fpos = i / sr * cr
            if fpos >= n_ctrl - 1:
                g = gain[n_ctrl - 1]
            else:
                j = int(fpos)
                g = (gain[j + 1] - gain[j]) * (fpos - j) + gain[j]
            x[i] = np.float64(s) / 10 ** (g / 20.0)
    b0, b1, b2, a1, a2 = sos[0], sos[1], sos[2], sos[4], sos[5]
    z0 = 0.0
    z1 = 0.0
    total = 0.0
    count = 0
    for i in range(n):
        v = x[i]
        out = b0 * v + z0
        z0 = b1 * v - a1 * out + z1
        z1 = b2 * v - a2 * out
        if abs(out) > 1e-4:
            total += out * out
            count += 1
    if count == 0:
        return 1e-9
    return math.sqrt(total / count)


@njit(cache=True, nogil=True, parallel=True)
def mono_blocks(mix, blocks, m):
    """The mono mix (mix.mean(axis=1)) cut into rows of m samples, the rest
    of each row of `blocks` left as it is (zero)."""
    n = mix.shape[0]
    for k in prange(blocks.shape[0]):
        for i in range(k * m, min(n, k * m + m)):
            blocks[k, i - k * m] = (mix[i, 0] + mix[i, 1]) / 2.0


@njit(cache=True, nogil=True, parallel=True)
def times(spec, room, out):
    """out = spec * room (every block's spectrum through the room)."""
    for k in prange(spec.shape[0]):
        for f in range(spec.shape[1]):
            out[k, f] = spec[k, f] * room[f]


@njit(cache=True, nogil=True, parallel=True)
def add_wet_blocks(mix, c, wet, m, amount):
    """mix[:, c] += the reverb * amount * 0.9, the reverb being the rows of
    `wet` (each block of m samples through the room) overlapped and added:
    row k starts at sample k * m and rings on into the next rows."""
    n = mix.shape[0]
    rows, width = wet.shape
    for k in prange(rows):
        r0 = max(0, k - (width - 1) // m)          # the earliest row still ringing here
        for i in range(k * m, min(n, k * m + m)):
            v = 0.0
            for r in range(r0, k + 1):
                j = i - r * m
                if j < width:
                    v += wet[r, j]
            mix[i, c] += v * amount * 0.9


@njit(cache=True, nogil=True, parallel=True)
def finite(ys):
    """Whether every sample of ys is a number (not NaN or inf)."""
    n = ys.shape[0]
    chunks = (n + MIX_CHUNK - 1) // MIX_CHUNK
    bad = np.zeros(chunks, np.bool_)
    for c in prange(chunks):
        for i in range(c * MIX_CHUNK, min(n, c * MIX_CHUNK + MIX_CHUNK)):
            if not math.isfinite(ys[i]):
                bad[c] = True
                break
    for c in range(chunks):
        if bad[c]:
            return False
    return True


@njit(cache=True, nogil=True, parallel=True)
def peak(mix):
    """max(mix.max(), -mix.min()), or NaN if any sample is NaN."""
    n = mix.shape[0]
    chunks = (n + MIX_CHUNK - 1) // MIX_CHUNK
    top = np.zeros(chunks)
    nan = np.zeros(chunks, np.bool_)
    for c in prange(chunks):
        t = 0.0
        for i in range(c * MIX_CHUNK, min(n, c * MIX_CHUNK + MIX_CHUNK)):
            for ch in range(2):
                v = abs(mix[i, ch])
                if v > t:
                    t = v
                elif v != v:
                    nan[c] = True
        top[c] = t
    t = 0.0
    for c in range(chunks):
        if nan[c]:
            return np.nan
        t = max(t, top[c])
    return t


@njit(cache=True, nogil=True, parallel=True)
def to_pcm(mix, gain, out):
    """16-bit samples of mix * gain, clipped to -1..1 (numpy's arithmetic:
    scale, clip, * 32767, truncate).  Returns False if any sample isn't a
    number, so the caller can clean up."""
    n = mix.shape[0]
    chunks = (n + MIX_CHUNK - 1) // MIX_CHUNK
    bad = np.zeros(chunks, np.bool_)
    for k in prange(chunks):
        for i in range(k * MIX_CHUNK, min(n, k * MIX_CHUNK + MIX_CHUNK)):
            for c in range(2):
                v = mix[i, c] * gain
                if not math.isfinite(v):
                    bad[k] = True
                    v = 0.0
                v = min(max(v, -1.0), 1.0) * 32767
                out[i, c] = np.int16(int(v))
    for k in range(chunks):
        if bad[k]:
            return False
    return True
