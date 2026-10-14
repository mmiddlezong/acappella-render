"""Command-line entry point: score in, MP3 (or WAV) out."""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.signal import butter, oaconvolve, sosfilt

from .perform import plan, voice_kind
from .score import TimeMap, load_score
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


def _pan(i, n, name, kind):
    nm = name.lower()
    if "voice" in nm or "solo" in nm or "lead" in nm or kind == "bass" or "bass" in nm:
        return 0.0
    if n <= 1:
        return 0.0
    return -0.6 + 1.2 * i / (n - 1)


def _loudness(y):
    """Rough loudness: RMS after a high-pass (low notes count for less)."""
    hp = sosfilt(butter(2, 180, btype="high", fs=SR, output="sos"), y)
    voiced = np.abs(hp) > 1e-4
    if not voiced.any():
        return 1e-9
    return float(np.sqrt(np.mean(hp[voiced] ** 2)))


def _write_audio(path: Path, stereo: np.ndarray, bitrate=192):
    if path.suffix.lower() == ".wav":
        from scipy.io import wavfile
        wavfile.write(str(path), SR, (np.clip(stereo, -1, 1) * 32767).astype(np.int16))
        return
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
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


def render_score(path, out=None, bpm=None, tempo_scale=1.0, lang="auto", voices=None,
                 gains=None, focus=None, only=None, mute=None, stems=False, reverb=0.25,
                 vibrato=1.0, measures=None, report=False, quiet=False, singers=2,
                 humanize=1.0, breaths=1.0, attack=1.0):
    t_start = time.time()
    path = Path(path)
    score = load_score(path)
    tm = TimeMap(score.tempos, score.fermatas, bpm_override=bpm, tempo_scale=tempo_scale)

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
    mute = [f.lower() for f in (mute or [])]

    staff_names = []
    for st, *_ in lines:
        if st.name not in staff_names:
            staff_names.append(st.name)

    N = int(total * SR) + 1024
    mix = np.zeros((N, 2))
    stem_sigs = {}
    rendered = []
    for k, (st, name, units, gain) in enumerate(lines):
        nm = st.name.lower()
        if only and not _match(nm, only):
            continue
        if mute and _match(nm, mute):
            continue
        units = [u for u in units if u.t_on >= PREROLL - 0.05 and u.t_on < total - TAIL + 0.01]
        if not units:
            continue
        kind = voice_kind(st, None)
        key = _match(nm, voices.keys())
        if key:
            kind = voices[key]
        is_solo = st.instrument == "voice" or any(w in nm for w in ("solo", "lead", "voice"))
        vib = (28.0 if is_solo else 12.0) * vibrato
        seed = zlib.crc32(name.encode()) & 0xFFFF
        # backing parts are sung by a small section: each singer is a little
        # different (timing, tuning, vibrato, vocal tract), which is most of
        # what keeps a held chord from sounding like an organ
        n_sing = 1 if is_solo else max(1, int(singers))
        voices_out = []
        for s in range(n_sing):
            spread = 0.0 if n_sing == 1 else (2.0 * s / (n_sing - 1) - 1.0)    # -1..1
            singer = Singer(kind, total, seed=seed + 7919 * s,
                            vibrato_cents=vib * (1.0 + 0.15 * spread),
                            breathiness=0.08 if is_solo else 0.06,
                            humanize=humanize,
                            breaths=breaths * (1.0 if is_solo else 0.6) / np.sqrt(n_sing),
                            attack=attack, scoop_cents=30.0 if is_solo else 18.0,
                            formant_shift=1.0 + 0.0125 * spread,
                            detune_cents=3.0 * spread)
            singer.sing(units, gain_db=gain)
            voices_out.append((spread, singer.render(gain)[:N]))
        y = np.sum([v for _s, v in voices_out], axis=0)
        # every part is calibrated to the same loudness at 'f' (0 dB); the
        # dynamics curve already applied inside render() sets the real level
        g = np.interp(np.arange(len(y)) / SR * CR, np.arange(len(gain)), gain)
        ref = _loudness(y / 10 ** (g / 20.0))
        part_db = gains.get(_match(nm, gains.keys()) or "", 0.0)
        if is_solo:
            part_db += 2.0
        if focus:
            part_db += 4.0 if _match(nm, focus) else -9.0
        scale = 0.12 / ref * 10 ** (part_db / 20.0)
        pan0 = _pan(staff_names.index(st.name), len(staff_names), st.name, kind)
        for spread, ys in voices_out:
            ys = ys * scale
            pan = float(np.clip(pan0 + 0.12 * spread, -1.0, 1.0))
            lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            mix[:len(ys), 0] += ys * lg * np.sqrt(2)
            mix[:len(ys), 1] += ys * rg * np.sqrt(2)
        y = y * scale
        stem_sigs.setdefault(st.name, np.zeros(N))[:len(y)] += y
        rendered.append((name, kind, len(units), n_sing))

    if reverb > 0:
        ir = _reverb_ir()
        mono = mix.mean(axis=1)
        wet = np.stack([oaconvolve(mono, ir[:, c])[:N] for c in range(2)], axis=1)
        mix = mix + wet * reverb * 0.9
    peak = np.max(np.abs(mix)) + 1e-9
    gain_norm = 0.89 / peak
    mix *= gain_norm

    out = Path(out) if out else path.with_suffix(".mp3")
    _write_audio(out, mix)
    if stems:
        for nm, sig in stem_sigs.items():
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in nm).strip()
            st_out = out.with_name(f"{out.stem} - {safe}{out.suffix}")
            _write_audio(st_out, np.stack([sig, sig], axis=1) * gain_norm)

    if not quiet:
        print(f"{score.title}: {total - PREROLL - TAIL:.1f}s of music, "
              f"rendered in {time.time() - t_start:.1f}s -> {out}")
        saved = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
        print(f"  score file: {path.resolve()} (last saved {saved})")
        print(f"  lyrics read as: {LANG_NAMES.get(lang_used, lang_used)}")
        for name, kind, nu, ns in rendered:
            who = f"  x{ns} singers" if ns > 1 else ""
            print(f"  {name:22s} {kind:14s} {nu:4d} syllables{who}")
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
            if len(u.notes) > 1:
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
    a = p.parse_args(argv)
    render_score(a.score, out=a.out, bpm=a.bpm, tempo_scale=a.tempo_scale, lang=a.lang,
                 voices=a.voice, gains=a.gain, focus=a.focus, only=a.only, mute=a.mute,
                 stems=a.stems, reverb=a.reverb, vibrato=a.vibrato, measures=a.measures,
                 report=a.report, singers=a.singers, humanize=a.humanize, breaths=a.breaths,
                 attack=a.attack)


if __name__ == "__main__":
    main()
