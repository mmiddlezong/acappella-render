"""Command-line entry point: score in, MP3 (or WAV) out."""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import butter, sosfilt

from . import cache
from .perc import REF as PART_REF
from .perc import render_hits
from .perform import plan, voice_kind
from .score import load_score
from .synth import CR, SR, Singer

PREROLL = 0.35
TAIL = 2.5
LANG_NAMES = {"en": "English", "ja": "Japanese romaji/kana", "zh": "Chinese",
              "ko": "Korean (Hangul)", "ko-rom": "Korean (romanized)"}


def _parse_map(items, cast=str):
    out = {}
    for it in items or []:
        for part in it.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip().lower()] = cast(v.strip())
    return out


def _match(name: str, keys) -> str | None:
    n = name.lower()
    for k in keys:
        if n == k or n.startswith(k):
            return k
    return None


def _neutral_syllable(text: str):
    """Phonemes for the neutral syllable (e.g. "ah", "doo", "loo")."""
    from .phonemes import Phonemizer
    return Phonemizer("en").word([text])[0]


def _neutralize(units, syl):
    """Same notes, rhythm, ties, slurs and accents, but every lyric word
    replaced by one neutral syllable (melismas stay melismas; vocables like
    doo/dm are left alone)."""
    import copy
    onset, nucleus, coda = syl
    out = []
    for u in units:
        v = copy.deepcopy(u)
        if not u.word:
            out.append(v)
            continue
        # a slurred syllable change (legato_in) becomes a plain re-sung vowel
        v.onset = [] if u.legato_in else list(onset)
        v.nucleus, v.coda = list(nucleus), list(coda)
        for n in v.notes:
            n.nucleus = None
        out.append(v)
    return out


@lru_cache(maxsize=4)
def _reverb_ir(rt60=1.3, length=2.2, seed=7):
    rng = np.random.default_rng(seed)
    L = int(length * SR)
    t = np.arange(L) / SR
    env = np.exp(-6.91 * t / rt60)
    ir = np.zeros((L, 2))
    for ch in range(2):
        nz = rng.standard_normal(L) * env
        # darker as it decays
        lo = sosfilt(butter(1, 2500, fs=SR, output="sos"), nz)
        w = np.clip(t / 0.6, 0, 1)
        ir[:, ch] = (1 - w) * nz + w * lo
        # a few early reflections
        for d, g in ((0.011, 0.6), (0.017, 0.45), (0.023 + 0.004 * ch, 0.4), (0.031, 0.3)):
            ir[int(d * SR), ch] += g * 8
    pre = int(0.018 * SR)
    ir = np.vstack([np.zeros((pre, 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    return ir


@lru_cache(maxsize=4)
def _reverb_spectrum(L, dtype=np.float64):
    """The room's two channels, transformed for a convolution of length L."""
    ir = _reverb_ir().astype(dtype)
    return [rfft(ir[:, c], L, workers=-1) for c in range(2)]


def _pan(i, n, name, kind):
    nm = name.lower()
    if "voice" in nm or "solo" in nm or "lead" in nm or kind == "bass" or "bass" in nm:
        return 0.0
    if n <= 1:
        return 0.0
    return -0.6 + 1.2 * i / (n - 1)


_HIGHPASS = butter(2, 180, btype="high", fs=SR, output="sos")
REVERB_BLOCK = 2 ** 18      # samples of the mix per block of the reverb


def _loudness(y):
    """Rough loudness: RMS after a high-pass (low notes count for less)."""
    hp = sosfilt(_HIGHPASS, y)
    voiced = np.abs(hp) > 1e-4
    if not voiced.any():
        return 1e-9
    return float(np.sqrt(np.mean(hp[voiced] ** 2)))


def _write_audio(path: Path, stereo: np.ndarray, bitrate=192, finite=False, scratch=False):
    """Write -1..1 stereo as 16-bit audio.  finite: no NaN/inf to clean up;
    scratch: stereo may be used up (converted in place)."""
    if not finite:
        stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)
    elif not scratch:
        stereo = stereo.copy()
    np.clip(stereo, -1, 1, out=stereo)
    stereo *= 32767
    _write_pcm(path, stereo.astype(np.int16), bitrate)


def _write_pcm(path: Path, pcm: np.ndarray, bitrate=192):
    if path.suffix.lower() == ".wav":
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        return
    try:
        import lameenc  # type: ignore
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate)
        enc.set_in_sample_rate(SR)
        enc.set_channels(2)
        enc.set_quality(2)
        data = enc.encode(pcm.tobytes()) + enc.flush()
        path.write_bytes(data)
    except ImportError:
        import shutil
        import subprocess
        import tempfile
        if not shutil.which("ffmpeg"):
            raise SystemExit("MP3 needs 'pip install lameenc' (or ffmpeg). Use -o file.wav instead.")
        from scipy.io import wavfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wavfile.write(tmp.name, SR, pcm)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-b:a",
                        f"{bitrate}k", str(path)], check=True)


def _render_voice(job, key=None, part_gain=None):
    """Sing one singer's track, or a part's percussion track (job["hits"]);
    runs in a worker process.  Returns (track, air warnings, loudness):
    with a cache key, the track goes into the cache and comes back as None
    (so it doesn't have to be copied back to the caller); with the part's
    dynamics curve (a part of just this voice), loudness is the part's
    calibration level, else None."""
    ys, air = _render_track(job)
    ref = None
    if part_gain is not None and _finite(ys):
        ref = _part_loudness([ys], part_gain)
    if key and cache.store(key, ys, air):
        return None, air, ref
    return ys, air, ref


def _finite(ys):
    from . import synth
    if synth._fast is not None:
        return synth._fast.parallel(synth._fast.finite, ys)
    return bool(np.isfinite(ys.sum(dtype=np.float64)))


def _part_loudness(voices, gain):
    """A part's loudness with the dynamics curve divided back out (render
    _score calibrates every part to the same level at 'f')."""
    from . import synth
    fast = synth._fast
    if fast is not None:
        return float(fast.parallel(fast.part_loudness, np.stack(voices),
                                   np.ascontiguousarray(gain, dtype=np.float64),
                                   float(SR), CR, _HIGHPASS[0]))
    y = np.sum(voices, axis=0)
    g = np.interp(np.arange(len(y)) / SR * CR, np.arange(len(gain)), gain)
    return _loudness(y / 10 ** (g / 20.0))


def _render_track(job):
    """(track, air warnings) for one job."""
    if "hits" in job:
        return render_hits(job["hits"], job["gain"], job["N"], seed=job["seed"],
                           humanize=job["humanize"], until=job["until"]), []
    from . import synth
    singing = synth._fast.singing if synth._fast is not None else None
    if singing is not None:
        with singing.get_lock():
            singing.value += 1
    try:
        singer = Singer(job["kind"], job["total"], **job["opts"])
        singer.sing(job["units"], gain_db=job["gain"])
        ys = singer.render(job["gain"])[:job["N"]]
    finally:
        if singing is not None:
            with singing.get_lock():
                singing.value -= 1
    return ys, [tuple(w) for w in singer.air_warnings]


_POOL = None          # worker processes kept between renders (by the render server)


def _warm_up(_i=0):
    """Sing a tiny phrase so this process has loaded the compiled kernels
    and built the wavetables before a real voice arrives."""
    from .synth import SungNote, Unit
    singer = Singer("tenor", 1.0, seed=1)
    singer.sing([Unit(onset=["d"], nucleus=["u"], coda=["t"],
                      notes=[SungNote(t0=0.2, t1=0.6, midi=60.0)])], gain_db=np.zeros(singer.n))
    ys = singer.render(np.zeros(singer.n))
    from . import synth
    fast = synth._fast
    if fast is not None:
        # and the mixing kernels, as render_score calls them
        mix = np.zeros((len(ys), 2))
        cached = ys.copy()
        cached.flags.writeable = False              # as voices come from the cache
        for track in (ys, cached):
            fast.parallel(fast.place, mix, track, np.float32(1.0), 0.7, 0.7, 1.4)
            fast.parallel(fast.finite, track)
        perc = ys.astype(float)
        fast.parallel(fast.place, mix, perc, 1.0, 0.7, 0.7, 1.4)
        perc.flags.writeable = False
        fast.parallel(fast.place, mix, perc, 1.0, 0.7, 0.7, 1.4)
        _part_loudness([ys], np.zeros(singer.n))
        blocks = np.zeros((2, len(ys)), dtype=np.float32)
        fast.parallel(fast.mono_blocks, mix, blocks, len(ys) // 2)
        fast.parallel(fast.add_wet_blocks, mix, 0, blocks, len(ys) // 2, 0.25)
        spec = np.zeros((2, 8), dtype=np.complex64)
        fast.parallel(fast.times, spec, spec[0], np.empty_like(spec))
        fast.parallel(fast.to_pcm, mix, 0.5 / (fast.parallel(fast.peak, mix) + 1e-9),
                      np.empty((len(ys), 2), dtype=np.int16))
    return os.getpid()


def _join_pool(singing):
    """A worker process starting: it shares with the others how many voices
    are being sung, so each voice's kernels use its share of the cores, and
    it goes when the process that started it does (however that ends)."""
    import threading
    from . import synth
    if synth._fast is not None:
        synth._fast.singing = singing
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def keep_pool(workers=None, wait=True):
    """Keep warmed-up worker processes for every later render in this
    process.  wait=False returns at once: the warming up goes on meanwhile,
    and voices sent to the pool before it's done start right after."""
    global _POOL
    import multiprocessing
    import threading
    if workers is None:
        # a voice's kernels share the cores out among themselves: more
        # processes than half the cores only cost memory
        workers = max(2, min((os.cpu_count() or 2) // 2, 8))
    _POOL = ProcessPoolExecutor(max_workers=workers, initializer=_join_pool,
                                initargs=(multiprocessing.Value("i", 0),))
    warming = [_POOL.submit(_warm_up, k) for k in range(workers)]

    def warm_here():
        _warm_up()
        # the room's impulse response, for every mix from now on
        width = next_fast_len(REVERB_BLOCK + _reverb_ir().shape[0] - 1, real=True)
        _reverb_spectrum(width, np.float32)

    if wait:
        warm_here()
        for f in warming:
            f.result()
    else:
        threading.Thread(target=warm_here, daemon=True).start()


def close_pool():
    """Stop the worker processes now (even mid-render)."""
    global _POOL
    if _POOL is None:
        return
    for proc in list((getattr(_POOL, "_processes", None) or {}).values()):
        proc.kill()
    _POOL.shutdown(wait=False, cancel_futures=True)
    _POOL = None


def _render_voices(jobs, use_cache=True, workers=None, progress=None, part_gains=None):
    """Every singer's track, in order: from the cache when nothing that goes
    into it changed, otherwise rendered, several at once.  progress(done,
    to_sing, unchanged) is called as voices finish.  part_gains[i], if not
    None, is the dynamics curve of a part sung by jobs[i] alone: its
    loudness is measured as soon as it's sung (and goes into the cache, or
    into the list returned last).
    Returns (tracks, voices unchanged, cache keys, loudnesses)."""
    keys = [cache.voice_key(j) for j in jobs] if use_cache else [None] * len(jobs)
    part_gains = part_gains or [None] * len(jobs)
    louds = [None] * len(jobs)
    out = [cache.load(k) if k else None for k in keys]
    todo = [i for i, r in enumerate(out) if r is None]
    todo.sort(key=lambda i: "hits" in jobs[i])        # voices first: they take longest
    is_voice = ["hits" not in j for j in jobs]
    voices_todo = sum(is_voice[i] for i in todo)
    if not workers:
        # a voice needs about 130 bytes per sample while it renders: run as
        # many at once as fit in half the memory
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            ram = 8 * 1024 ** 3
        from . import synth
        per_voice = (60 if synth._fast is not None else 130) * max((j["N"] for j in jobs), default=1)
        workers = max(1, min(os.cpu_count() or 1, int(0.5 * ram / per_voice)))
    workers = min(len(todo), workers)
    unchanged = sum(is_voice) - voices_todo
    sung = 0

    def finished(i, result):
        nonlocal sung
        ys, air, ref = result
        if ys is None:                              # the worker put it in the cache
            ys, air = cache.load(keys[i]) or _render_track(jobs[i])
        out[i] = (ys, air)
        if ref is not None:
            louds[i] = ref
            if keys[i]:
                cache.store_value(cache.part_key([keys[i]], part_gains[i]), ref)
        if is_voice[i]:
            sung += 1
            if progress:
                progress(sung, voices_todo, unchanged)

    if progress:
        progress(0, voices_todo, unchanged)
    if _POOL is not None and todo:
        running = {_POOL.submit(_render_voice, jobs[i], keys[i], part_gains[i]): i for i in todo}
        for f in as_completed(running):
            finished(running[f], f.result())
    elif workers > 1:
        # with the compiled synthesizer a voice's heavy lifting releases the
        # interpreter lock, so threads in this process do: no process to
        # start, nothing to copy back.  Otherwise separate processes.
        from . import synth
        pool = ThreadPoolExecutor if synth._fast is not None else ProcessPoolExecutor
        with pool(max_workers=workers) as ex:
            running = {ex.submit(_render_voice, jobs[i], None, part_gains[i]): i for i in todo}
            for f in as_completed(running):
                finished(running[f], f.result())
    else:
        for i in todo:
            finished(i, _render_voice(jobs[i], None, part_gains[i]))
    if use_cache:
        if _POOL is None:
            for i in todo:
                cache.store(keys[i], *out[i])
        cache.prune()
    return out, unchanged, keys, louds


def _place(mix, ys, pan, scale=1.0):
    """mix += ys * scale, panned (equal power)."""
    from . import synth
    fast = synth._fast
    lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    root2 = np.sqrt(2)
    if fast is not None:
        fast.parallel(fast.place, mix, ys, ys.dtype.type(scale), lg, rg, root2)
    else:
        ys = ys * scale
        mix[:len(ys), 0] += ys * lg * root2
        mix[:len(ys), 1] += ys * rg * root2


def _add_reverb(mix, reverb):
    """The room, added to the stereo mix in place: the mono mix through
    each channel's impulse response."""
    if reverb <= 0:
        return
    from . import synth
    fast = synth._fast
    N = mix.shape[0]
    if fast is not None:
        # a block of the song at a time (the forward transforms shared by
        # both channels, all the blocks at once on every core), the blocks'
        # tails overlapping the blocks after them.  In single precision,
        # which is still far finer than the 16-bit file, and quicker: the
        # transforms are as fast as memory.
        m = REVERB_BLOCK
        width = next_fast_len(m + _reverb_ir().shape[0] - 1, real=True)
        blocks = np.zeros(((N + m - 1) // m, width), dtype=np.float32)
        fast.parallel(fast.mono_blocks, mix, blocks, m)
        spec = rfft(blocks, axis=1, workers=-1)
        del blocks
        heard = np.empty_like(spec)
        for c, room in enumerate(_reverb_spectrum(width, np.float32)):
            fast.parallel(fast.times, spec, room, heard)
            wet = irfft(heard, width, axis=1, workers=-1, overwrite_x=True)
            fast.parallel(fast.add_wet_blocks, mix, c, wet, m, float(reverb))
    else:
        # the same, the whole song as one block
        L = next_fast_len(N + _reverb_ir().shape[0] - 1, real=True)
        spec = rfft(mix.mean(axis=1), L, workers=-1)
        for c, room in enumerate(_reverb_spectrum(L)):
            wet = irfft(spec * room, L, workers=-1)[:N]
            mix[:, c] += wet * reverb * 0.9


def _part_track(tracks, N, reverb, use_cache=True):
    """(float32 frames, cache key) of one part's stereo track: its tracks
    [(track, pan, scale, cache key)] placed, with their own reverb.  From
    the cache if every track it's made of is unchanged."""
    import hashlib
    import pickle
    keyed = use_cache and all(k for _y, _p, _s, k in tracks)
    if keyed:
        h = hashlib.sha256(b"part track" + cache.code_fingerprint())
        h.update(pickle.dumps((N, float(reverb), [(k, float(p), float(sc)) for _y, p, sc, k in tracks]),
                              protocol=5))
        key = h.hexdigest()
        got = cache.load(key)
        if got is not None and got[0].shape == (N, 2):
            return got[0], key
    else:
        key = os.urandom(16).hex()
    buf = np.zeros((N, 2))
    for ys, pan, scale, _k in tracks:
        _place(buf, ys, pan, scale)
    _add_reverb(buf, reverb)
    ys = buf.astype(np.float32)
    del buf
    cache.store(key, ys, [])
    return ys, key


def render_score(path, out=None, bpm=None, tempo_scale=1.0, lang="auto", voices=None,
                 gains=None, focus=None, only=None, mute=None, stems=False, reverb=0.25,
                 vibrato=1.0, measures=None, report=False, quiet=False, singers=2,
                 humanize=1.0, breaths=1.0, attack=1.0, neutral=None, solo=None,
                 use_cache=True, workers=None, progress=None, part_tracks=None):
    """progress(text), if given, hears what the render is doing.

    part_tracks, if given (a dict), also gets each MuseScore part's own
    stereo track, so a player can leave parts out (MuseScore's mute and
    solo): "parts" is a list of (part id, .npy file of float32 frames),
    reverb included, which add up to the mix; "gain" scales them the way
    the mix file is scaled.  They're kept in the voice cache, so a part
    that didn't change isn't mixed again."""
    def say(text):
        if progress:
            progress(text)

    t_start = time.time()
    path = Path(path)
    say("Reading the score")
    score = load_score(path)
    tm = score.time_map(bpm_override=bpm, tempo_scale=tempo_scale)

    # excerpt window
    win0, win1 = F(0), score.end
    if measures:
        a, _, b = measures.partition("-")
        a = int(a)
        b = int(b) if b else a
        starts = [(no, pos, ln) for no, pos, ln in score.measures]
        sel = [x for x in starts if a <= x[0] <= b]
        if not sel:
            raise SystemExit(f"No measures {measures} in this score")
        win0 = sel[0][1]
        last = [x for x in sel if x[0] == b] or sel
        win1 = last[-1][1] + last[-1][2]
    offset = PREROLL - tm.sec(win0)
    total = tm.sec(win1) - tm.sec(win0) + PREROLL + TAIL
    n_frames = int(total * CR) + 2

    lines, warnings, lang_used = plan(score, tm, offset, lang, n_frames)

    voices = _parse_map(voices)
    gains = _parse_map(gains, float)
    focus = [f.lower() for f in (focus or [])]
    only = [f.lower() for f in (only or [])]
    solo = [f.lower() for f in (solo or [])]
    neutral_syl = _neutral_syllable(neutral) if neutral else None
    solo_names = []
    air_warn = []
    mute = [f.lower() for f in (mute or [])]

    staff_names = []
    for st, *_ in lines:
        if st.name not in staff_names:
            staff_names.append(st.name)

    N = int(total * SR) + 1024
    mix = np.zeros((N, 2))
    stem_sigs = {}
    rendered = []
    parts = []
    for k, (st, name, units, gain) in enumerate(lines):
        nm = st.name.lower()
        if only and not _match(nm, only):
            continue
        if mute and _match(nm, mute):
            continue
        units = [u for u in units if u.t_on >= PREROLL - 0.05 and u.t_on < total - TAIL + 0.01]
        hits = [u for u in units if u.perc]
        units = [u for u in units if not u.perc]
        if not units and not hits:
            continue
        kind = voice_kind(st, None)
        key = _match(nm, voices.keys())
        if key:
            kind = voices[key]
        if solo:
            is_solo = bool(_match(nm, solo))
        else:
            is_solo = st.instrument == "voice" or any(w in nm for w in ("solo", "lead", "voice"))
        if neutral_syl:
            n_words = sum(1 for u in units if u.word)
            if n_words:
                units = _neutralize(units, neutral_syl)
                lines[k] = (st, name, sorted(units + hits, key=lambda u: u.t_on),
                            gain)                        # so --report shows it too
                solo_names.append(f"{name} ({n_words})")
        vib = (16.0 if is_solo else 6.0) * vibrato
        seed = zlib.crc32(name.encode()) & 0xFFFF
        # backing parts are sung by a small section: each singer is a little
        # different (timing, tuning, vibrato, vocal tract), which is most of
        # what keeps a held chord from sounding like an organ
        n_sing = (1 if is_solo else max(1, int(singers))) if units else 0
        jobs = []
        for s in range(n_sing):
            spread = 0.0 if n_sing == 1 else (2.0 * s / (n_sing - 1) - 1.0)    # -1..1
            opts = dict(seed=seed + 7919 * s,
                        vibrato_cents=vib * (1.0 + 0.15 * spread),
                        breathiness=0.08 if is_solo else 0.06,
                        humanize=humanize,
                        breaths=breaths * (1.0 if is_solo else 0.6) / np.sqrt(n_sing),
                        attack=attack, scoop_cents=15.0 if is_solo else 10.0,
                        formant_shift=1.0 + 0.0125 * spread,
                        detune_cents=3.0 * spread)
            opts = {k2: float(v) if isinstance(v, np.floating) else v for k2, v in opts.items()}
            jobs.append((spread, dict(kind=kind, total=total, N=N, units=units, gain=gain,
                                      opts=opts)))
        perc_job = None
        if hits:
            # vocal percussion (x-noteheads): one track per part, not per singer
            until = []
            for h in hits:
                nxt = next((u for u in units if u.t_on > h.t_on + 0.005), None)
                # silent again just before the next sung syllable's consonant
                until.append(None if nxt is None else
                             nxt.t_on - sum(Singer.cons_dur(c) for c in nxt.onset) - 0.012)
            perc_job = dict(hits=hits, gain=gain, N=N, seed=seed + 31, humanize=humanize,
                            until=until)
        parts.append((st, name, kind, is_solo, units, hits, gain, seed, jobs, perc_job))

    def sung_so_far(done, to_sing, unchanged):
        if not to_sing:
            left = "nothing to sing"
        elif done:
            left = f"{done} of {to_sing} voices sung"
        else:
            left = f"Singing {to_sing} voice" + ("s" if to_sing > 1 else "")
        say(left + (f", {unchanged} unchanged" if unchanged else ""))

    voice_jobs = [j for p in parts for _s, j in p[-2]]
    perc_jobs = [p[-1] for p in parts if p[-1] is not None]
    # a part of one voice has its loudness measured as soon as it's sung
    part_gains = [p[6] if len(p[-2]) == 1 else None for p in parts for _j in p[-2]]
    sung, reused, voice_keys, louds = _render_voices(
        voice_jobs + perc_jobs, use_cache, workers, sung_so_far,
        part_gains + [None] * len(perc_jobs))
    say("Mixing")
    perc_tracks = iter(ys for ys, _air in sung[len(voice_jobs):])
    perc_keys = iter(voice_keys[len(voice_jobs):])
    sung = iter(sung[:len(voice_jobs)])
    voice_keys = iter(voice_keys)
    louds = iter(louds)
    from . import synth
    fast = synth._fast
    # with part_tracks, what goes into each part's track: (track, pan,
    # scale, its cache key), by part
    placed = {}
    for st, name, kind, is_solo, units, hits, gain, seed, jobs, perc_job in parts:
        nm = st.name.lower()
        voices_out = []
        keys = []
        measured = None
        for s, (spread, _job) in enumerate(jobs):
            ys, air = next(sung)
            keys.append(next(voice_keys))
            measured = next(louds)
            if s == 0:
                for w in air:
                    air_warn.append((name, gain) + tuple(w))
            if not _finite(ys):                              # a NaN or inf somewhere
                bad = ~np.isfinite(ys)
                # never let one bad sample ruin the file
                ys = np.where(bad, 0.0, ys)
                warnings.append(f"{name}: {int(bad.sum())} invalid samples silenced (please report)")
            voices_out.append((spread, ys))
        n_sing = len(jobs)
        part_db = gains.get(_match(nm, gains.keys()) or "", 0.0)
        if is_solo:
            part_db += 2.0
        if focus:
            part_db += 4.0 if _match(nm, focus) else -9.0
        pan0 = _pan(staff_names.index(st.name), len(staff_names), st.name, kind)
        stem = stem_sigs.setdefault(st.name, np.zeros(N)) if stems else None

        def place(ys, pan, scale=1.0, key=None):
            if part_tracks is not None:
                placed.setdefault(st.part_id or st.name, []).append((ys, pan, scale, key))
            else:
                _place(mix, ys, pan, scale)

        if voices_out:
            # every part is calibrated to the same loudness at 'f' (0 dB); the
            # dynamics curve already applied inside render() sets the real level
            pkey = cache.part_key(keys, gain) if use_cache and all(keys) else None
            ref = measured if len(jobs) == 1 else None      # (only ever of a clean voice)
            if ref is None and pkey:
                ref = cache.load_value(pkey)
            y = None
            if ref is None:
                ref = _part_loudness([v for _s, v in voices_out], gain)
                if pkey:
                    cache.store_value(pkey, ref)
            scale = PART_REF / ref * 10 ** (part_db / 20.0)
            for (spread, ys), key in zip(voices_out, keys):
                place(ys, float(np.clip(pan0 + 0.12 * spread, -1.0, 1.0)), scale, key)
            if stems:
                if y is None:
                    y = np.sum([v for _s, v in voices_out], axis=0)
                y = y * scale
                stem[:len(y)] += y
        if perc_job is not None:
            yp = next(perc_tracks)
            place(yp, pan0, 10 ** (part_db / 20.0), next(perc_keys))
            if stems:
                stem += yp * 10 ** (part_db / 20.0)
        rendered.append((name, kind, len(units), n_sing, len(hits)))

    if part_tracks is not None:
        part_tracks["parts"] = []
        for part_id, tracks in placed.items():
            ys = _part_track(tracks, N, reverb, use_cache)
            part_tracks["parts"].append((part_id, cache.track_path(ys[1])))
            mix += ys[0]
    else:
        _add_reverb(mix, reverb)
    if fast is not None:
        peak = fast.parallel(fast.peak, mix) + 1e-9
    else:
        peak = max(float(mix.max()), -float(mix.min())) + 1e-9
    gain_norm = 0.89 / peak
    if part_tracks is not None:
        part_tracks["gain"] = gain_norm

    out = Path(out) if out else path.with_suffix(".mp3")
    pcm = None
    if fast is not None:
        pcm = np.empty((N, 2), dtype=np.int16)
        if not fast.parallel(fast.to_pcm, mix, gain_norm, pcm):
            pcm = None
    if pcm is None:
        mix *= gain_norm
        _write_audio(out, mix, finite=bool(np.isfinite(peak)), scratch=True)
    else:
        _write_pcm(out, pcm)
    if stems:
        for nm, sig in stem_sigs.items():
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in nm).strip()
            st_out = out.with_name(f"{out.stem} - {safe}{out.suffix}")
            _write_audio(st_out, np.stack([sig, sig], axis=1) * gain_norm)

    if not quiet:
        n_voices = sum(len(p[-2]) for p in parts)
        again = f", {reused} of {n_voices} voices unchanged" if reused else ""
        print(f"{score.title}: {total - PREROLL - TAIL:.1f}s of music, "
              f"rendered in {time.time() - t_start:.1f}s{again} -> {out}")
        saved = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        print(f"  score file: {path.resolve()} (last saved {saved})")
        print(f"  lyrics read as: {LANG_NAMES.get(lang_used, lang_used)}")
        for name, kind, nu, ns, nh in rendered:
            who = f"  x{ns} singers" if ns > 1 else ""
            perc = f"  + {nh} percussion hits" if nh else ""
            print(f"  {name:22s} {kind:14s} {nu:4d} syllables{who}{perc}")
        if air_warn:
            mstarts = [(no, tm.sec(pos) + offset) for no, pos, _ln in score.measures]

            def where(t):
                no = mstarts[0][0]
                for n2, t2 in mstarts:
                    if t2 <= t + 1e-6:
                        no = n2
                return no

            from .perform import DYN_DB
            for name, gain, t_start, t_out, short in air_warn:
                lvl = float(gain[min(len(gain) - 1, int(t_out * CR))])
                dyn = min(("pp", "p", "mp", "mf", "f", "ff"), key=lambda k: abs(DYN_DB[k] - lvl))
                print(f"  breath: {name} runs out of air in m.{where(t_out)} and is short about "
                      f"{short:.1f}s of singing at {dyn} before it can refill (last real breath "
                      f"m.{where(t_start)}); add a breath mark or a rest")
        if neutral_syl:
            if solo_names:
                print(f"  lyric words sung on '{neutral}': " + ", ".join(solo_names))
            else:
                print("  note: --neutral found no lyric words to replace")
        seen = set()
        for w in warnings:
            if w not in seen:
                seen.add(w)
                print("  note:", w)
    if report:
        _report(lines, tm)
    return out


def _report(lines, tm):
    """Print what each part sings, for checking syllables."""
    for st, name, units, _g in lines:
        print(f"\n== {name}")
        row = []
        for u in [u for u in units if u.t_on >= PREROLL - 0.05][:60]:
            syl = "".join(u.onset) + "·" + "".join(u.nucleus) + "·" + "".join(u.coda)
            if u.perc:
                syl = f"[{u.perc}]"
            elif len(u.notes) > 1:
                syl += f"~{len(u.notes)}"
            row.append(f"{u.t_on - PREROLL:6.2f}s {syl}")
        for i in range(0, len(row), 6):
            print("   " + " | ".join(row[i:i + 6]))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="render.py",
        description="Render an a cappella score (.mscz/.mscx/.musicxml/.mxl) to MP3 with "
                    "the syllables, dynamics and cutoffs sung as written.")
    p.add_argument("score")
    p.add_argument("-o", "--out", help="output file (.mp3 or .wav); default: next to the score")
    p.add_argument("--measures", help="render only these measures, e.g. 17-32")
    p.add_argument("--bpm", type=float, help="override the starting tempo (quarter = BPM)")
    p.add_argument("--tempo-scale", type=float, default=1.0, help="e.g. 0.8 for 80%% speed")
    p.add_argument("--lang", default="auto", choices=["auto", "en", "ja", "zh", "ko"],
                   help="how to read Latin-letter lyrics (auto-detected by default); "
                        "ko reads romanized Korean")
    p.add_argument("--neutral", nargs="?", const="ah", metavar="SYLLABLE",
                   help="sing every lyric word on one neutral syllable, in every part; "
                        "vocables (doo, dm...) are kept (default 'ah'; e.g. --neutral loo)")
    p.add_argument("--solo", action="append", metavar="PART",
                   help="which part(s) are the soloist (default: parts named Voice/Solo/Lead)")
    p.add_argument("--voice", action="append", metavar="PART=TYPE",
                   help="set a part's voice: soprano, mezzo-soprano, alto, tenor, baritone, bass")
    p.add_argument("--gain", action="append", metavar="PART=DB", help="adjust a part's level")
    p.add_argument("--focus", action="append", metavar="PART",
                   help="practice mix: this part up, the others down")
    p.add_argument("--only", action="append", metavar="PART", help="render only these parts")
    p.add_argument("--mute", action="append", metavar="PART", help="leave these parts out")
    p.add_argument("--stems", action="store_true", help="also write one file per part")
    p.add_argument("--reverb", type=float, default=0.25, help="0 = dry, 1 = very wet")
    p.add_argument("--vibrato", type=float, default=1.0, help="vibrato amount (0 = straight tone)")
    p.add_argument("--singers", type=int, default=2,
                   help="singers on each backing part (default 2; 1 = one voice per part)")
    p.add_argument("--humanize", type=float, default=1.0,
                   help="timing/tuning scatter (0 = exactly on the grid and in tune)")
    p.add_argument("--breaths", type=float, default=1.0,
                   help="breaths before phrases (0 = none, 2 = very audible)")
    p.add_argument("--attack", type=float, default=1.0,
                   help="attack softness (0.5 = crisper, 2 = very soft)")
    p.add_argument("--report", action="store_true", help="print the syllables each part sings")
    p.add_argument("--no-cache", action="store_true",
                   help="sing every voice again instead of reusing the ones that didn't change")
    p.add_argument("--jobs", type=int, help="voices rendered at once (default: one per CPU core)")
    a = p.parse_args(argv)
    render_score(a.score, out=a.out, bpm=a.bpm, tempo_scale=a.tempo_scale, lang=a.lang,
                 voices=a.voice, gains=a.gain, focus=a.focus, only=a.only, mute=a.mute,
                 stems=a.stems, reverb=a.reverb, vibrato=a.vibrato, measures=a.measures,
                 report=a.report, singers=a.singers, humanize=a.humanize, breaths=a.breaths,
                 attack=a.attack, neutral=a.neutral, solo=a.solo, use_cache=not a.no_cache,
                 workers=a.jobs)


if __name__ == "__main__":
    main()
