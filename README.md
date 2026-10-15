# acappella-render

Turns a MuseScore arrangement into an MP3 where every part actually **sings
its syllables**, follows the **dynamics and hairpins**, and **cuts off where
the notes and rests say**. It's a formant synthesizer, so it still sounds
like a synthetic choir rather than real singers, but it phrases like a
singer: attacks swell in instead of switching on, pitch glides between
notes, and each backing part is sung by a small section. The point is to
hear what your syllable choices, dynamics and cutoffs do.

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
python render.py song.mscz --neutral                   # every lyric word sung on "ah"
python render.py song.mscz --neutral loo               # ...or on any syllable
```

`--neutral` swaps every lyric word (pinyin, Hangul, kana, English words), in
whichever part sings it, for one neutral syllable, keeping notes, rhythm,
slurs and dynamics. Vocables (doo, dm, dn, duh, oh, bap, 음…) are left as
written. Syllables that are both (da, ba, wo, 다…) count as words only inside
a phrase that has other words in it.

`--solo PART` (repeatable) says which part is the soloist when it isn't
named Voice/Solo/Lead: one singer, more vibrato, +2 dB.

Other options: `--bpm 128`, `--tempo-scale 0.8`, `--gain "Bass=+3"`,
`--voice "Voice=tenor"` (the solo part's voice type is guessed from its
range), `--reverb 0` (dry), `--vibrato 0` (straight tone; 2 = twice as much),
`--lang ja|en|zh|ko`, `-o out.wav`. Part names match by prefix and ignore case.
`--bpm` and `--tempo-scale` work like MuseScore's playback speed: every
tempo scales, pauses don't.

How human it sounds:

```
--singers 2      singers on each backing part (default 2; 1 = one voice per part, 3+ = choir)
--attack 1       attack softness (0.5 = crisper, 2 = very soft)
--humanize 1     timing/tuning scatter (0 = exactly on the grid and in tune)
--breaths 1      breaths before phrases (0 = none)
```

Input: `.mscz` / `.mscx` directly (MuseScore 3 and 4), or MusicXML.

**Re-rendering is quick.** Every singer's track is kept (in
`~/Library/Caches/acappella-render/`, up to 3 GB) under a fingerprint of
everything that goes into it, so rendering again after an edit only sings
the parts that changed, and changing only the mix (`--gain`, `--focus`,
`--reverb`) sings nothing again. The result is the same, sample for sample,
as a fresh render. The voices that are sung are rendered several at once
(as many as fit in memory). A 2½-minute song with 9 voices renders in about
9 s, about 5 s after changing one part, and 2 s with nothing changed.
`--no-cache` sings everything again; `--jobs N` sets how many voices render at
once.

## Playing the render inside MuseScore (macOS, MuseScore 4)

A plugin renders the open score with one button, and then MuseScore's own
playback (Play, the moving cursor, starting from a selected note, loops)
plays the render instead of MuseScore's sounds.

```
musescore/install.sh
```

This builds and installs the "Acappella Render" VST3 effect (it fetches the
MIT-licensed VST 3 SDK into `musescore/build/` the first time; needs CMake
and the Xcode command line tools) and the "Acappella Render" MuseScore
plugin. Then:

1. Quit and reopen MuseScore.
2. Home > Plugins: enable Acappella Render.
3. Plugins > Acappella Render > Render. It saves the score, renders it
   (about 20 s for a 3-minute song) and shows the renderer's notes.
4. Once per score: in the Mixer, add Acappella Render to the Master
   channel's Audio FX. Press Play.

Render again after editing; the new render takes over even mid-playback.
Switch the effect off in the Mixer to hear MuseScore's sounds again. While
playback is stopped MuseScore's sound passes through, so clicking notes
still sounds. "Keep MuseScore's own sound underneath" in the dialog mixes it
in quietly during playback too (handy with the metronome).

The render is timed exactly the way MuseScore times playback (see Tempo
below), so it stays with the cursor. Keep the playback speed at 100% and repeats on:
the render doesn't follow MuseScore's speed slider, and D.S./D.C./Coda jumps
aren't rendered, so playback drifts after the first jump.
`musescore/install.sh --uninstall` removes the effect and the plugin. Renders
are kept in `~/Library/Application Support/AcappellaRender/` (the last 3).

## How it reads your score

**Syllables**
- A cappella vocables use their a cappella sound: *doo, dah, dn, dm, bm, jn,
  hoo, woh, loo, doot, dat, bap, bahm, ding*. "dah", "bap" and "dat" get the
  round, dark "ah" backing singers use, not the dictionary's "duh" or the
  "a" of "cat", and open vocables are sung a little warmer than lyrics so
  they blend.
- English words use the CMU pronouncing dictionary. Your phonetic splits
  work: "wi-ll", "bu-t", "beau-ti-fu-l" sing the consonant on the note it's
  under. "chi-i-ild" holds the vowel and closes to "ld" on the last note.
  "long-ger" sings as "lon-ger".
- Japanese romaji and kana are read as Japanese, auto-detected per score:
  *te* is "teh", *u* is unrounded, *rou* is a long o, *kasu* devoices to
  "kas", and several morae under one note share it. Kanji can't be read,
  so use kana or romaji.
- Korean Hangul is read as Korean, auto-detected. Enter it the usual way,
  one syllable per note; pronunciation carries across notes the way it's
  sung: 음악 is "eu-mak", 없어 "eop-sseo", 좋아 "jo-a", 같이 "ga-chi",
  학교 has a tense k, 할 거야 / 기다릴게 tense the k. Final consonants are
  held, not popped. A rest or breath mark ends the run. 음, 흠, 응 on their
  own are hums; inside a word (마음, 처음) they're sung as written. Latin
  vocables (doo, dm) in a Korean score stay vocables.
- Romanized Korean (*saranghae*, *neo-reul*, *bo-go si-peo*) is detected
  too, or force it with `--lang ko`.
- Chinese characters are read through pinyin.
- A lone "d" or "t" is a short ghost hit.
- X-noteheads are vocal percussion, made with the mouth the way a
  beatboxer would. With no lyric they're a snare (a "pf"/"psh"); a lyric
  picks the sound: *b, boom, kick* = kick, *ts, tss, ch* = hi-hat,
  *k, pf, psh* = snare. The hit follows the part's dynamics and accents.
  The singer can't make a hit and sing at once, so the hit's tail stops
  before the next sung note's consonant, no breath is taken during it,
  and the next lyric-less note is re-sung rather than slurred through.
  An x-notehead with any other lyric is a short, soft ghost note on that
  syllable.
- "(sim.)" repeats the previous bar's syllable pattern beat for beat until
  the next lyric.
- A note with no lyric continues the previous syllable (melisma). After a
  rest it re-sings it, and the run prints a note about that.

**Timing and cutoffs**
- Each vowel lands on the beat. Onset consonants come just before it, taking
  time from the previous note (at most half of it). A vowel out of silence
  starts swelling a hair before the beat so it arrives on it.
- Sound stops at the written end of the note, with a short fade into the
  cutoff rather than a hard stop. A final stop consonant ("doot", "dat")
  releases on the cutoff.
- Singers are a few milliseconds early or late and a few cents off, like
  people. `--humanize 0` puts everything exactly on the grid.
- Staccato sings half the length, staccatissimo 30%. Breath marks shorten
  the note before them. Ties merge. Slurs and melismas are legato.

**Tempo** (MuseScore files are timed the way MuseScore 4 plays them, to
within a fraction of a millisecond, so the MP3 lines up with MuseScore's
playback of the same score)
- Tempo marks, including "a tempo" and "tempo primo" ones. Before the first
  tempo mark it's 120 BPM, as in MuseScore; a pickup takes the tempo marked
  in the bar after it.
- rit., rall., accel. and the other tempo-change lines change the tempo by
  their factor (rit./rall. 0.75, accel. 1.33, stringendo 1.5, ..., or the one
  you set) along their easing curve, in the same steps MuseScore uses. The
  new tempo holds after the line until the next tempo mark ("a tempo").
- A fermata stretches the time to the next note in any part by its type's
  factor (normal ×2, short ×1.5, very short ×1.25, long ×3, very long ×4)
  or the time stretch you set. "Play" turned off in Properties is respected.
- A breath mark or caesura with a pause (Properties > Pause) adds that
  much silence; the note before it stops at the mark. Section breaks pause
  3 s (or what they're set to) before the next bar.
- Swing, from the score's swing setting or a "Swing"/"Straight" text (a
  system text sets every part, a staff text just its own): a note on the
  off-beat of the swing unit comes in later by the swing ratio and still
  ends on time, the note before it is held to meet it, and nothing swings
  where the beat is subdivided or in tuplets.

**Dynamics**
- ppp to fff are about 5–6 dB apart, and voice quality follows: soft is
  breathier, loud is brighter. Higher notes come out a little brighter and
  louder, lower ones a little darker.
- A hairpin ramps to the dynamic written at its end. A hairpin with no
  ending dynamic moves about one step and holds until the next marking,
  never drifting far from the last written one. sfz/fp work. Accents and
  marcato add punch (a faster, harder attack).
- Every part is calibrated to the same loudness at a given dynamic, so the
  balance you hear comes from your markings. The solo part gets +2 dB.

## How it sings
- **Attacks** swell in. A vowel out of silence gets a soft onset led by a
  little air; out of a consonant the vowel rises over 30–60 ms; the tone
  starts darker and brightens as it opens. Tense consonants (Korean) and
  accents get quicker attacks. `--attack` scales all of it.
- **Consonants** sit under the vowel: aspiration is short and quiet, and the
  tongue and lips move quickly into a consonant and more slowly out of it.
  A d/b/g starting a syllable is a voiced closure that springs open with a
  click: crisp and percussive on backing vocables (doo, dah, dn), lighter
  in lyric words. Consonants are scaled to each voice, so a soprano's "d"
  is as clear against her vowel as a bass's.
- **Pitch** moves like a voice: each note is approached with a little lag
  and overshoot, most of the move happens during the consonant so the vowel
  arrives in tune, phrases start a touch flat and settle, and long releases
  sag slightly.
- **Vibrato** is light: backing parts are nearly straight tone and the solo
  has a gentle vibrato. It only comes in on notes long enough to hold
  (about 0.6 s or more), starts late and fades in. `--vibrato 2` doubles it,
  `--vibrato 0` turns it off.
- **Sections**: each backing part is two singers, slightly different in
  timing, tuning, vibrato and voice, panned a little apart. The solo part is
  always one singer. Percussion hits are one per part, not one per singer.
- **Breaths** come from a running model of each singer's air: it drains
  while they sing (faster when loud) and refills in the gaps. Before a
  phrase they take in what the stretch ahead needs plus a reserve, topping
  up more in long rests. A big refill squeezed into a short gap is a quick,
  loud, hissy gasp; a top-up in a long rest is slow and quiet; a singer
  running low snatches a catch breath in a gap that would otherwise go
  unused. `--breaths 2` makes them louder, `--breaths 0` turns them off.

## Limits
- It sounds synthetic. Judge syllables, rhythm, dynamics and balance, not
  tone.
- Percussion (drum) staves are skipped; write vocal percussion with
  x-noteheads on the singer's own staff. Repeats and voltas are followed, but
  D.S./D.C./Coda jumps aren't. Only verse 1 lyrics are used.
- Korean lyrics don't mark word boundaries, so sound changes that depend on
  them are approximated (못 잊어, 꽃잎 and the common lyric words are
  handled; rarer compounds may be read a little too literally).
- A full song with two singers per backing part takes about twice as long
  to render as with `--singers 1`, and a voice needs about 1 GB of memory
  per 2½ minutes of music while it renders.
