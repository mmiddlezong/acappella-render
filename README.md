# acappella-render

Turns a MuseScore arrangement into an MP3 where every part actually **sings
its syllables**, follows the **dynamics and hairpins**, and **cuts off where
the notes and rests say**. It's a formant synthesizer, so it sounds like a
clear synthetic choir, not real singers. The point is to hear what your
syllable choices, dynamics and cutoffs do.

## Setup (once)

Python 3.9 or newer.

```
pip install numpy scipy cmudict lameenc
pip install pypinyin      # optional: Chinese-character lyrics
pip install music21       # optional: MusicXML input
```

## Use

```
python render.py "Zankyosanka.mscz"                    # -> Zankyosanka.mp3 next to it
python render.py song.mscz --measures 17-32            # just a section (fast for checking edits)
python render.py song.mscz --focus Alto                # practice mix: alto up, others down
python render.py song.mscz --only Tenor --only Bass    # just some parts
python render.py song.mscz --stems                     # also one MP3 per part
python render.py song.mscz --report                    # print what each part sings
```

Other options: `--bpm 128`, `--tempo-scale 0.8`, `--gain "Bass=+3"`,
`--voice "Voice=tenor"` (the solo part's voice type is guessed from its
range), `--reverb 0` (dry), `--vibrato 0` (straight tone),
`--lang ja|en|zh`, `-o out.wav`. Part names match by prefix and ignore case.

Input: `.mscz` / `.mscx` directly (MuseScore 3 and 4), or MusicXML.

## How it reads your score

**Syllables**
- A cappella vocables use their a cappella sound: *doo, dah, dn, dm, bm, jn,
  hoo, woh, loo, doot, dat, bahm, ding*. "dah" is "ah", not the
  dictionary's "duh".
- English words use the CMU pronouncing dictionary. Your phonetic splits
  work: "wi-ll", "bu-t", "beau-ti-fu-l" sing the consonant on the note it's
  under. "chi-i-ild" holds the vowel and closes to "ld" on the last note.
  "long-ger" sings as "lon-ger".
- Japanese romaji and kana are read as Japanese, auto-detected per score:
  *te* is "teh", *u* is unrounded, *rou* is a long o, *kasu* devoices to
  "kas", and several morae under one note share it. Kanji can't be read,
  so use kana or romaji.
- Chinese characters are read through pinyin.
- A lone "d" or "t" is a short ghost hit. X-noteheads are short, soft ghost
  notes.
- "(sim.)" repeats the previous bar's syllable pattern beat for beat until
  the next lyric.
- A note with no lyric continues the previous syllable (melisma). After a
  rest it re-sings it, and the run prints a note about that.

**Timing and cutoffs**
- Each vowel lands on the beat. Onset consonants come just before it, taking
  time from the previous note (at most half of it).
- Sound stops at the written end of the note. A final stop consonant
  ("doot", "dat") releases on the cutoff.
- Staccato sings half the length, staccatissimo 30%. Breath marks shorten
  the note before them. Fermatas stretch time. Ties merge. Slurs and
  melismas are legato.

**Dynamics**
- ppp to fff are about 5–6 dB apart, and voice quality follows: soft is
  breathier, loud is brighter.
- A hairpin ramps to the dynamic written at its end. A hairpin with no
  ending dynamic moves about one step and holds until the next marking,
  never drifting far from the last written one. sfz/fp work. Accents and
  marcato add punch.
- Every part is calibrated to the same loudness at a given dynamic, so the
  balance you hear comes from your markings. The solo part gets +2 dB.

## Limits
- It sounds synthetic. Judge syllables, rhythm, dynamics and balance, not
  tone.
- Percussion staves are skipped. Repeats and voltas are followed, but
  D.S./D.C./Coda jumps aren't. Only verse 1 lyrics are used.
