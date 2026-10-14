"""Turn lyric syllables into phonemes.

A syllable is returned as (onset, nucleus, coda): three lists of phoneme
symbols.  The nucleus is one vowel, two vowels (a diphthong: the first is
held, the second is the off-glide), or a nasal/liquid for hummed syllables
like "dn", "mm", "bm".

Lookup order for Latin-script tokens:
  1. exceptions (common English words the a cappella grammar would misread)
  2. tokens with no vowel letters -> hum syllables ("dn", "dm", "jn", "mm")
  3. the a cappella spelling grammar ("doo", "dah", "woh", "bahm", "dit"...)
  4. the CMU pronouncing dictionary (real English words)
  5. letter-by-letter fallback rules
Chinese characters go through pypinyin, kana through a built-in table,
Hangul through its own spelling (with Korean sound changes applied across
the sung line: see korean_sounds).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------- inventory
VOWELS = {
    "i", "I", "e", "E", "ae", "a", "O", "o", "U", "u", "V", "@", "3", "y", "1", "M",
    "7",                                   # Korean eo
}
NASALS = {"m", "n", "N"}
LIQUIDS = {"l", "r", "R"}
GLIDES = {"w", "j"}
STOPS = {"p", "b", "t", "d", "k", "g",
         "p'", "t'", "k'", "ph", "th", "kh",   # Korean tense / aspirated
         "p_}", "t_}", "k_}"}                  # Korean unreleased finals
FRICATIVES = {"f", "v", "T", "D", "s", "z", "S", "Z", "C", "h", "x", "s'", "S'"}
AFFRICATES = {"tS", "dZ", "ts", "dz", "tS'", "tSh"}
CONSONANTS = NASALS | LIQUIDS | GLIDES | STOPS | FRICATIVES | AFFRICATES
SONORANT_LIKE = NASALS | LIQUIDS

Syllable = tuple  # (onset list, nucleus list, coda list)


class Multi(list):
    """Several syllables written under one note ("zaya", "kaerou"):
    the note is shared between them."""


# ------------------------------------------------- a cappella spelling grammar
# Longest spellings first.  Value is the phoneme list for the onset.
_ONSETS = [
    ("sch", ["S"]), ("wh", ["w"]), ("sh", ["S"]), ("ch", ["tS"]), ("th", ["D"]), ("zh", ["Z"]),
    ("dj", ["dZ"]), ("tw", ["t", "w"]), ("kw", ["k", "w"]), ("bw", ["b", "w"]),
    ("dw", ["d", "w"]), ("gw", ["g", "w"]), ("bl", ["b", "l"]), ("pl", ["p", "l"]),
    ("dr", ["d", "r"]), ("br", ["b", "r"]), ("tr", ["t", "r"]), ("ts", ["ts"]),
    ("dz", ["dz"]), ("ny", ["n", "j"]), ("dy", ["d", "j"]), ("by", ["b", "j"]),
    ("b", ["b"]), ("d", ["d"]), ("g", ["g"]), ("p", ["p"]), ("t", ["t"]),
    ("k", ["k"]), ("c", ["k"]), ("q", ["k"]), ("j", ["dZ"]), ("m", ["m"]),
    ("n", ["n"]), ("l", ["l"]), ("r", ["r"]), ("w", ["w"]), ("y", ["j"]),
    ("h", ["h"]), ("f", ["f"]), ("v", ["v"]), ("s", ["s"]), ("z", ["z"]),
    ("", []),
]
# (spelling, nucleus, allowed position) position: "any", "open" (no coda) or
# "closed" (needs a coda)
_NUCLEI = [
    ("ooh", ["u"], "any"), ("oo", ["u"], "any"), ("ou", ["u"], "open"),
    ("oh", ["o"], "any"), ("oa", ["o"], "any"), ("ow", ["a", "U"], "any"),
    ("aah", ["a"], "any"), ("ah", ["a"], "any"), ("aa", ["a"], "any"),
    ("aw", ["O"], "any"), ("au", ["O"], "any"),
    ("ay", ["e", "I"], "any"), ("ai", ["a", "I"], "any"), ("ei", ["e", "I"], "any"),
    ("eh", ["E"], "any"), ("ee", ["i"], "any"), ("ea", ["i"], "any"),
    ("ih", ["I"], "any"), ("uh", ["V"], "any"), ("oi", ["O", "I"], "any"),
    ("oy", ["O", "I"], "any"), ("igh", ["a", "I"], "any"),
    ("a", ["a"], "open"), ("a", ["ae"], "closed"),
    ("e", ["i"], "open"), ("e", ["E"], "closed"),
    ("i", ["i"], "open"), ("i", ["I"], "closed"),
    ("o", ["o"], "open"), ("o", ["a"], "closed"),
    ("u", ["u"], "open"), ("u", ["V"], "closed"),
    ("y", ["a", "I"], "open"),
]
_CODAS = [
    ("ng", ["N"]), ("nk", ["N", "k"]), ("mp", ["m", "p"]), ("nt", ["n", "t"]),
    ("nd", ["n", "d"]), ("ts", ["t", "s"]), ("st", ["s", "t"]), ("sh", ["S"]),
    ("ch", ["tS"]), ("ck", ["k"]),
    ("m", ["m"]), ("n", ["n"]), ("t", ["t"]), ("p", ["p"]), ("k", ["k"]),
    ("d", ["d"]), ("b", ["b"]), ("g", ["g"]), ("s", ["s"]), ("z", ["z"]),
    ("l", ["l"]), ("x", ["k", "s"]), ("f", ["f"]),
    ("", []),
]

# Words where the grammar above would give the wrong reading.
_EXCEPTIONS: dict[str, Syllable] = {
    "a": ([], ["@"], []), "the": (["D"], ["@"], []), "to": (["t"], ["u"], []),
    "do": (["d"], ["u"], []), "who": (["h"], ["u"], []), "two": (["t"], ["u"], []),
    "you": (["j"], ["u"], []), "i": ([], ["a", "I"], []), "my": (["m"], ["a", "I"], []),
    "by": (["b"], ["a", "I"], []), "fly": (["f", "l"], ["a", "I"], []),
    "so": (["s"], ["o"], []), "no": (["n"], ["o"], []), "go": (["g"], ["o"], []),
    "love": (["l"], ["V"], ["v"]), "one": (["w"], ["V"], ["n"]),
    "come": (["k"], ["V"], ["m"]), "some": (["s"], ["V"], ["m"]),
    "we": (["w"], ["i"], []), "me": (["m"], ["i"], []), "be": (["b"], ["i"], []),
    "he": (["h"], ["i"], []), "she": (["S"], ["i"], []),
}


def _norm(token: str) -> str:
    t = unicodedata.normalize("NFKC", token).strip().lower()
    t = re.sub(r"[\"'“”‘’`.,!?;:()\[\]{}…\-–—_~*]", "", t)
    return t


def _parse_acappella(tok: str) -> Syllable | None:
    """Parse tokens like doo, dah, woh, bahm, dit, jow, hoo, loo, doom."""
    for o_sp, o_ph in _ONSETS:
        if not tok.startswith(o_sp):
            continue
        rest = tok[len(o_sp):]
        for n_sp, n_ph, pos in _NUCLEI:
            if not rest.startswith(n_sp):
                continue
            tail = rest[len(n_sp):]
            # 'h' after a vowel only lengthens it ("doh", "dah", "bahm")
            if tail.startswith("h") and n_sp not in ("ooh", "oh", "ah", "aah", "eh", "ih", "uh"):
                tail = tail[1:]
            for c_sp, c_ph in _CODAS:
                if tail != c_sp:
                    continue
                closed = bool(c_sp)
                if pos == "open" and closed:
                    continue
                if pos == "closed" and not closed:
                    continue
                return (list(o_ph), list(n_ph), list(c_ph))
    return None


_HUM_ONSETS = [("dj", ["dZ"]), ("j", ["dZ"]), ("d", ["d"]), ("b", ["b"]),
               ("g", ["g"]), ("t", ["t"]), ("p", ["p"]), ("k", ["k"]),
               ("h", ["h"]), ("", [])]


def _parse_hum(tok: str) -> Syllable | None:
    """Vowel-less tokens: dn, dm, bm, jn, mm, hmm, ng, dng, n, m."""
    for o_sp, o_ph in _HUM_ONSETS:
        if not tok.startswith(o_sp):
            continue
        rest = tok[len(o_sp):]
        if re.fullmatch(r"m+", rest):
            return (list(o_ph), ["m"], [])
        if re.fullmatch(r"n+", rest):
            return (list(o_ph), ["n"], [])
        if re.fullmatch(r"n+g", rest) or rest == "ng":
            return (list(o_ph), ["N"], [])
    return None


# ---------------------------------------------------------- CMU dictionary
_ARPA = {
    "AA": ["a"], "AE": ["ae"], "AH": ["V"], "AO": ["O"], "AW": ["a", "U"],
    "AY": ["a", "I"], "EH": ["E"], "ER": ["3"], "EY": ["e", "I"], "IH": ["I"],
    "IY": ["i"], "OW": ["o"], "OY": ["O", "I"], "UH": ["U"], "UW": ["u"],
    "B": ["b"], "CH": ["tS"], "D": ["d"], "DH": ["D"], "F": ["f"], "G": ["g"],
    "HH": ["h"], "JH": ["dZ"], "K": ["k"], "L": ["l"], "M": ["m"], "N": ["n"],
    "NG": ["N"], "P": ["p"], "R": ["r"], "S": ["s"], "SH": ["S"], "T": ["t"],
    "TH": ["T"], "V": ["v"], "W": ["w"], "Y": ["j"], "Z": ["z"], "ZH": ["Z"],
}
_LEGAL_ONSETS = {
    ("s", "p"), ("s", "t"), ("s", "k"), ("p", "l"), ("p", "r"), ("b", "l"),
    ("b", "r"), ("t", "r"), ("d", "r"), ("k", "l"), ("k", "r"), ("g", "l"),
    ("g", "r"), ("f", "l"), ("f", "r"), ("s", "l"), ("s", "m"), ("s", "n"),
    ("s", "w"), ("t", "w"), ("k", "w"), ("T", "r"), ("S", "r"), ("p", "j"),
    ("b", "j"), ("k", "j"), ("f", "j"), ("m", "j"), ("h", "j"),
    ("s", "t", "r"), ("s", "p", "r"), ("s", "k", "r"), ("s", "p", "l"),
}
_cmu = None


def _cmu_dict():
    global _cmu
    if _cmu is None:
        try:
            import cmudict  # type: ignore
            _cmu = cmudict.dict()
        except Exception:
            _cmu = {}
    return _cmu


def _cmu_syllables(word: str) -> list[Syllable] | None:
    d = _cmu_dict()
    prons = d.get(word)
    if not prons:
        return None
    pron = prons[0]
    phones: list[tuple[list[str], bool]] = []
    for p in pron:
        base = re.sub(r"\d", "", p)
        is_v = p[-1].isdigit()
        if base == "AH" and p.endswith("0"):
            phones.append((["@"], True))
        elif base in _ARPA:
            phones.append((_ARPA[base], is_v))
    vidx = [i for i, (_, v) in enumerate(phones) if v]
    if not vidx:
        # syllabic consonant word like "mm" / "hmm"
        cons = [p for ph, _ in phones for p in ph]
        nas = [c for c in cons if c in NASALS]
        if nas:
            k = cons.index(nas[0])
            return [(cons[:k], [nas[0]], cons[k + 1:])]
        return None
    sylls: list[Syllable] = []
    onset_start = 0
    for n, vi in enumerate(vidx):
        nxt = vidx[n + 1] if n + 1 < len(vidx) else None
        if nxt is None:
            coda_end = len(phones)
            next_onset = len(phones)
        else:
            cluster = list(range(vi + 1, nxt))
            # maximal legal onset for the next syllable
            k = len(cluster)
            take = 0
            for t in range(min(3, k), 0, -1):
                cl = tuple(phones[i][0][0] for i in cluster[k - t:])
                if t == 1 or cl in _LEGAL_ONSETS:
                    take = t
                    break
            next_onset = vi + 1 + (k - take)
            coda_end = next_onset
        onset = [p for ph, _ in phones[onset_start:vi] for p in ph]
        nucleus = list(phones[vi][0])
        coda = [p for ph, _ in phones[vi + 1:coda_end] for p in ph]
        sylls.append((onset, nucleus, coda))
        onset_start = next_onset
    return sylls


# --------------------------------------------- letter/phoneme alignment
_SPELL = {
    "AA": "a o ah aw au al ea", "AE": "a ai", "AH": "u a o e i ou oo io ia", "AO": "o aw au a ou oa al",
    "AW": "ow ou", "AY": "i y igh ie ai ye ei uy", "EH": "e ea a ai ie", "ER": "er ir ur or ear ar r our",
    "EY": "a ay ai ey ei ea", "IH": "i y e ee ie u", "IY": "e ee ea y i ie ey ei",
    "OW": "o ow oa oe ou ough eau", "OY": "oy oi", "UH": "oo u ou", "UW": "oo u ew ue o ou ui eu",
    "B": "b bb", "CH": "ch tch t", "D": "d dd ed", "DH": "th", "F": "f ff ph gh", "G": "g gg gu gh",
    "HH": "h wh", "JH": "j g dg ge", "K": "k c ck ch q cc", "L": "l ll", "M": "m mm mb",
    "N": "n nn kn gn", "NG": "ng n", "P": "p pp", "R": "r rr wr", "S": "s ss c ce sc",
    "SH": "sh ti ci s ch ss", "T": "t tt ed th", "TH": "th", "V": "v f", "W": "w wh u o",
    "Y": "y i j", "Z": "z s zz ss x", "ZH": "s g si",
}
_SPELL = {k: set(v.split()) for k, v in _SPELL.items()}
_ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def _align_letters(letters, phones, owner=None):
    """Monotonic alignment: each phoneme takes 1-4 letters (or shares an
    'x'), letters may be silent.  Returns, for each phoneme, the index of
    its first letter."""
    L, P = len(letters), len(phones)
    INF = 1e9
    cost = [[INF] * (P + 1) for _ in range(L + 1)]
    back = [[None] * (P + 1) for _ in range(L + 1)]
    cost[0][0] = 0.0

    def c_match(seg, ph):
        if seg in _SPELL.get(ph, ()):
            return 0.0
        vl = all(ch in "aeiouy" for ch in seg)
        if (ph in _ARPA_VOWELS) == vl:
            return 1.5 + 0.5 * (len(seg) - 1)
        return 4.0

    for i in range(L + 1):
        for j in range(P + 1):
            c = cost[i][j]
            if c >= INF:
                continue
            if i < L:      # silent letter
                sc = 0.6 if letters[i] in "aeiouyhw" else 1.6
                if c + sc < cost[i + 1][j]:
                    cost[i + 1][j] = c + sc
                    back[i + 1][j] = (i, j, None)
            if j < P:
                for k in range(1, 5):
                    if i + k > L:
                        break
                    cross = owner is not None and owner[i] != owner[i + k - 1]
                    nc = c + c_match("".join(letters[i:i + k]), phones[j]) + 0.01 * i + (2.0 if cross else 0.0)
                    if nc < cost[i + k][j + 1]:
                        cost[i + k][j + 1] = nc
                        back[i + k][j + 1] = (i, j, i)
                if i < L and letters[i] == "x" and j + 1 < P and phones[j] == "K" and phones[j + 1] == "S":
                    if c < cost[i + 1][j + 2]:
                        cost[i + 1][j + 2] = c
                        back[i + 1][j + 2] = (i, j, "x")
                if c + 5 < cost[i][j + 1]:      # phoneme with no letter
                    cost[i][j + 1] = c + 5
                    back[i][j + 1] = (i, j, max(0, i - 1))
    if cost[L][P] >= INF:
        return None
    first = [0] * P
    i, j = L, P
    while i > 0 or j > 0:
        pi, pj, li = back[i][j]
        if li == "x":
            first[pj] = first[pj + 1] = pi
        elif li is not None:
            first[pj] = li
        i, j = pi, pj
    return [min(x, L - 1) for x in first]


# ------------------------------------------------------------ fallback rules
_FALLBACK_LETTERS = [
    ("tion", ["S", "@", "n"]), ("igh", ["a", "I"]), ("ough", ["o"]), ("ll", ["l"]),
    ("sh", ["S"]), ("ch", ["tS"]), ("th", ["T"]), ("ph", ["f"]), ("ng", ["N"]),
    ("ck", ["k"]), ("qu", ["k", "w"]), ("wh", ["w"]),
    ("ee", ["i"]), ("ea", ["i"]), ("oo", ["u"]), ("ou", ["a", "U"]),
    ("ow", ["a", "U"]), ("ai", ["e", "I"]), ("ay", ["e", "I"]), ("oa", ["o"]),
    ("oi", ["O", "I"]), ("oy", ["O", "I"]), ("au", ["O"]), ("aw", ["O"]),
    ("ie", ["i"]), ("ue", ["u"]),
    ("a", ["ae"]), ("e", ["E"]), ("i", ["I"]), ("o", ["a"]), ("u", ["V"]),
    ("y", ["i"]), ("b", ["b"]), ("c", ["k"]), ("d", ["d"]), ("f", ["f"]),
    ("g", ["g"]), ("h", ["h"]), ("j", ["dZ"]), ("k", ["k"]), ("l", ["l"]),
    ("m", ["m"]), ("n", ["n"]), ("p", ["p"]), ("q", ["k"]), ("r", ["r"]),
    ("s", ["s"]), ("t", ["t"]), ("v", ["v"]), ("w", ["w"]), ("x", ["k", "s"]),
    ("z", ["z"]),
]


def _fallback_phones(tok: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tok):
        for sp, ph in _FALLBACK_LETTERS:
            if tok.startswith(sp, i):
                out.extend(ph)
                i += len(sp)
                break
        else:
            i += 1
    # silent final e: "make", "time"
    if tok.endswith("e") and len(tok) > 2 and out and out[-1] == "E":
        out.pop()
    return out


def _phones_to_syllable(phones: list[str]) -> Syllable:
    vi = [k for k, p in enumerate(phones) if p in VOWELS]
    if not vi:
        nas = [k for k, p in enumerate(phones) if p in NASALS | LIQUIDS]
        if nas:
            k = nas[0]
            return (phones[:k], [phones[k]], phones[k + 1:])
        return (phones, ["@"], [])
    a, b = vi[0], vi[-1]
    nucleus = phones[a:b + 1]
    nucleus = [p for p in nucleus if p in VOWELS][:2]
    return (phones[:a], nucleus, phones[b + 1:])


# ----------------------------------------------------------------- Chinese
_PY_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g",
                "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
_PY_INIT_PH = {
    "b": ["b"], "p": ["p"], "m": ["m"], "f": ["f"], "d": ["d"], "t": ["t"],
    "n": ["n"], "l": ["l"], "g": ["g"], "k": ["k"], "h": ["x"], "j": ["dZ"],
    "q": ["tS"], "x": ["C"], "zh": ["dZ"], "ch": ["tS"], "sh": ["S"], "r": ["r"],
    "z": ["dz"], "c": ["ts"], "s": ["s"], "y": [], "w": [],
}
# final -> (glide onset, nucleus, coda)
_PY_FINALS = {
    "a": ([], ["a"], []), "o": (["w"], ["o"], []), "e": ([], ["V"], []),
    "ai": ([], ["a", "I"], []), "ei": ([], ["e", "I"], []), "ao": ([], ["a", "U"], []),
    "ou": ([], ["o", "U"], []), "an": ([], ["a"], ["n"]), "en": ([], ["@"], ["n"]),
    "ang": ([], ["a"], ["N"]), "eng": ([], ["@"], ["N"]), "ong": ([], ["U"], ["N"]),
    "er": ([], ["3"], []), "i": ([], ["i"], []), "ia": (["j"], ["a"], []),
    "ie": (["j"], ["E"], []), "iao": (["j"], ["a", "U"], []), "iu": (["j"], ["o", "U"], []),
    "iou": (["j"], ["o", "U"], []), "ian": (["j"], ["E"], ["n"]), "in": ([], ["i"], ["n"]),
    "iang": (["j"], ["a"], ["N"]), "ing": ([], ["i"], ["N"]), "iong": (["j"], ["U"], ["N"]),
    "u": ([], ["u"], []), "ua": (["w"], ["a"], []), "uo": (["w"], ["o"], []),
    "uai": (["w"], ["a", "I"], []), "ui": (["w"], ["e", "I"], []), "uei": (["w"], ["e", "I"], []),
    "uan": (["w"], ["a"], ["n"]), "un": (["w"], ["@"], ["n"]), "uen": (["w"], ["@"], ["n"]),
    "uang": (["w"], ["a"], ["N"]), "ueng": (["w"], ["@"], ["N"]),
    "v": ([], ["y"], []), "ü": ([], ["y"], []), "ve": (["j"], ["E"], []),
    "üe": (["j"], ["E"], []), "van": (["j"], ["E"], ["n"]), "üan": (["j"], ["E"], ["n"]),
    "vn": ([], ["y"], ["n"]), "ün": ([], ["y"], ["n"]),
}


def pinyin_syllable(py: str) -> Syllable | None:
    py = py.lower().strip()
    py = re.sub(r"[1-5]$", "", py)
    if not py:
        return None
    init = ""
    for i in _PY_INITIALS:
        if py.startswith(i):
            init = i
            break
    fin = py[len(init):]
    onset = list(_PY_INIT_PH.get(init, []))
    # y-/w- spellings
    if init == "y":
        fin = {"i": "i", "in": "in", "ing": "ing", "u": "v", "ue": "ve", "uan": "van",
               "un": "vn"}.get(fin, "i" + fin if not fin.startswith("i") else fin)
    elif init == "w":
        fin = {"u": "u"}.get(fin, "u" + fin if not fin.startswith("u") else fin)
    elif init in ("j", "q", "x"):
        fin = {"u": "v", "ue": "ve", "uan": "van", "un": "vn"}.get(fin, fin)
    if fin == "i" and init in ("z", "c", "s"):
        return (onset, ["1"], [])
    if fin == "i" and init in ("zh", "ch", "sh", "r"):
        return (onset, ["3"], [])
    if fin not in _PY_FINALS:
        return None
    glide, nuc, coda = _PY_FINALS[fin]
    if init in ("b", "p", "m", "f") and fin == "o":
        glide = ["w"]
    return (onset + list(glide), list(nuc), list(coda))


def _chinese_syllable(ch: str) -> Syllable | None:
    try:
        from pypinyin import lazy_pinyin, Style  # type: ignore
    except Exception:
        return None
    py = lazy_pinyin(ch, style=Style.NORMAL, v_to_u=True)
    if not py:
        return None
    return pinyin_syllable(py[0])


# ---------------------------------------------------------------- Japanese
_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}
_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}


def kana_to_romaji(text: str) -> str:
    # katakana -> hiragana
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    out = ""
    i = 0
    while i < len(s):
        c = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if c == "っ":
            out += "Q"  # geminate marker
            i += 1
            continue
        if c == "ー":
            out += out[-1] if out and out[-1] in "aiueo" else ""
            i += 1
            continue
        r = _KANA.get(c)
        if r is None:
            i += 1
            continue
        if nxt in _SMALL_Y:
            v = _SMALL_Y[nxt]
            if r in ("shi", "chi", "ji"):
                r = r[:-1] + v
            else:
                r = r[:-1] + "y" + v
            i += 1
        out += r
        i += 1
    return out


_JA_ONSETS = [
    ("sh", ["S"]), ("ch", ["tS"]), ("ts", ["ts"]), ("ky", ["k", "j"]),
    ("gy", ["g", "j"]), ("ny", ["n", "j"]), ("hy", ["C"]), ("my", ["m", "j"]),
    ("ry", ["R", "j"]), ("by", ["b", "j"]), ("py", ["p", "j"]), ("j", ["dZ"]),
    ("k", ["k"]), ("g", ["g"]), ("s", ["s"]), ("z", ["dz"]), ("t", ["t"]),
    ("d", ["d"]), ("n", ["n"]), ("h", ["h"]), ("f", ["f"]), ("b", ["b"]),
    ("p", ["p"]), ("m", ["m"]), ("y", ["j"]), ("r", ["R"]), ("w", ["w"]),
    ("v", ["v"]), ("", []),
]
_JA_V = {"a": "a", "i": "i", "u": "M", "e": "E", "o": "o"}


def romaji_syllables(text: str) -> list[Syllable]:
    """Split a romaji string into morae.  'ou' and 'oo' become a long o;
    small-tsu / doubled consonants become a short closure on the previous
    syllable."""
    t = "".join(c if c == "Q" else c.lower() for c in text)
    sylls: list[Syllable] = []
    i = 0
    while i < len(t):
        c = t[i]
        if c == "Q" or (i + 1 < len(t) and c == t[i + 1] and c not in "aiueon"):
            if sylls:
                o, n, cd = sylls[-1]
                sylls[-1] = (o, n, cd + ["t"])
            i += 1
            continue
        if c == "n" and (i + 1 == len(t) or t[i + 1] not in "aiueoy"):
            sylls.append(([], ["n"], []))
            i += 1
            continue
        if c in "ou" and sylls and sylls[-1][1] == ["o"] and not sylls[-1][2] and c in "ou":
            # long o ("kou", "rou", "oo"): just extend the previous vowel
            if c == "u" or c == "o":
                i += 1
                continue
        for sp, ph in _JA_ONSETS:
            if t.startswith(sp, i) and i + len(sp) < len(t) and t[i + len(sp)] in _JA_V:
                v = t[i + len(sp)]
                sylls.append((list(ph), [_JA_V[v]], []))
                i += len(sp) + 1
                break
        else:
            i += 1
    return sylls


_ROMAJI_RE = re.compile(
    r"^(?:(?:ky|gy|ny|hy|my|ry|by|py|sh|ch|ts|[kgsztdnhbpmyrwfjv])?[aiueo]|n(?![aiueoy])|(?<=[aiueo])[kstpgdbcj](?=[kstpgdbcj]))+$")
_JA_MARKERS = {"shi", "tsu", "chi", "fu", "ji", "wo", "sho", "sha", "shu", "cho", "cha",
               "chu", "ryo", "ryu", "kyo", "kyu", "ju", "jo", "ja", "zu", "tte", "tta",
               "kko", "kou", "rou", "sou", "tou", "you", "hou", "gyo", "nyo", "zo", "ze",
               "nu", "ne", "mu", "ru", "su", "ku", "ke", "ki", "ri", "re"}
# vocable spellings that should never be read as romaji
_VOCABLE_HINT = re.compile(r"(oo|ah|oh|eh$|uh|aa|ee|ow$|aw$|ay$|[ptkdbgm]$|^[^aeiouy]+$)")


def is_romaji(tok: str) -> bool:
    return bool(tok) and bool(_ROMAJI_RE.match(tok))


# ------------------------------------------------------------------ Korean
# Jamo of a Hangul syllable, in Unicode order, spelled like Revised
# Romanization (initials as written at the start of a syllable).
_K_INI = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch",
          "k", "t", "p", "h"]
_K_MED = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u",
          "wo", "we", "wi", "yu", "eu", "ui", "i"]
_K_FIN = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp",
          "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]
_K_DOUBLE = {"gs": ("g", "s"), "nj": ("n", "j"), "nh": ("n", "h"), "lg": ("l", "g"),
             "lm": ("l", "m"), "lb": ("l", "b"), "ls": ("l", "s"), "lt": ("l", "t"),
             "lp": ("l", "p"), "lh": ("l", "h"), "bs": ("b", "s")}
# The seven sounds a final consonant can make: K T P (held, unreleased) and
# the sonorants N L M NG.
_K_CLASS = {"g": "K", "kk": "K", "k": "K", "gs": "K", "lg": "K",
            "n": "N", "nj": "N", "nh": "N",
            "d": "T", "t": "T", "s": "T", "ss": "T", "j": "T", "ch": "T", "h": "T",
            "l": "L", "lb": "L", "ls": "L", "lt": "L", "lh": "L",
            "m": "M", "lm": "M",
            "b": "P", "p": "P", "bs": "P", "lp": "P",
            "ng": "NG", "": ""}
_K_CODA_PH = {"K": "k_}", "T": "t_}", "P": "p_}", "N": "n", "L": "l", "M": "m", "NG": "N"}
_K_NASALIZE = {"K": "NG", "T": "N", "P": "M"}
_K_LENIS_OF = {"K": "g", "T": "d", "P": "b"}
_K_TENSE = {"g": "kk", "d": "tt", "b": "pp", "s": "ss", "j": "jj"}
_K_ASP = {"g": "k", "d": "t", "b": "p", "j": "ch"}
# a final stop (or double final) before ㅎ: what stays behind, and the
# aspirate the two make together (축하 -> 추카, 밝히다 -> 발키다, 앉히다 -> 안치다)
_K_H_ASP = {"g": ("", "k"), "kk": ("", "k"), "k": ("", "k"), "gs": ("", "k"),
            "lg": ("l", "k"), "d": ("", "t"), "t": ("", "t"), "s": ("", "t"),
            "ss": ("", "t"), "j": ("", "ch"), "ch": ("", "ch"), "b": ("", "p"),
            "p": ("", "p"), "bs": ("", "p"), "lb": ("l", "p"), "lt": ("l", "t"),
            "nj": ("n", "ch")}
_K_MED_PH = {
    "a": ([], ["a"]), "ae": ([], ["E"]), "ya": (["j"], ["a"]), "yae": (["j"], ["E"]),
    "eo": ([], ["7"]), "e": ([], ["e"]), "yeo": (["j"], ["7"]), "ye": (["j"], ["e"]),
    "o": ([], ["o"]), "wa": (["w"], ["a"]), "wae": (["w"], ["E"]), "oe": (["w"], ["e"]),
    "yo": (["j"], ["o"]), "u": ([], ["u"]), "wo": (["w"], ["7"]), "we": (["w"], ["e"]),
    "wi": (["w"], ["i"]), "yu": (["j"], ["u"]), "eu": ([], ["1"]), "ui": ([], ["1", "i"]),
    "i": ([], ["i"]),
}
_K_PALATAL_V = {"i", "ya", "yae", "yeo", "ye", "yo", "yu", "wi", "ui"}   # s -> sh before these
_K_IY_V = {"i", "ya", "yae", "yeo", "ye", "yo", "yu"}       # ㄴ is inserted before these
# vowel-less compatibility jamo used as hum syllables: ㅁ ㄴ ㅇ
_K_JAMO_HUM = {"ㅁ": "m", "ㄴ": "n", "ㅇ": "N"}

# Words that end before a following vowel-initial word, so their final is
# reduced before it links (못 와 -> 모돠, 첫인상 -> 처딘상, 헛웃음 -> 허두슴);
# 못 before i/y gets an inserted ㄴ (못 잊어 -> 몬니저).
_K_WORD_END = set("못첫헛몇")
# Pairs sung tense although the spelling doesn't show it: compounds and
# Sino-Korean words common in lyrics, and verbs whose stem ends in ㄴ/ㅁ.
_K_TENSE_PAIRS = set("""
별빛 달빛 눈빛 불빛 물빛 봄빛 손길 눈길 발길 물길 산길 밤길 길가 강가 눈가 물가 손가
봄바 밤바 물결 잠결 꿈결 람결 등불 눈동 발걸 발자 글자 눈짓 손짓 몸짓 손등 발등 손바
발바 물방 눈사 물소 술잔 길거 산새 절대 열정 실수 일상 일생 열심 갈증 발전 결정 실제
갈등 출신 활동 발달 열중 절정 몰두 일정 결심 발생 실종 안고 감고 품고 참고 참지 담고
남고 넘고 숨고
""".split())
# After the future/adnominal ending -(으)ㄹ the next consonant is tense:
# 할 거야 -> 할꺼야, 기다릴게 -> 기다릴께, 할 수 -> 할쑤, 할 줄 알아 -> 할쭐 알아
_K_L_DEP = set("거것게걸겁수데")
# verb/adjective stems that take -을 (잊을 수, 있을 거야, 좋을 텐데)
_K_EUL_STEMS = set("잊있없같좋많받먹찾웃믿닿놓넣앉맞잡씻벗읽싫않낫짓잃찮들걸")
# 의: word-initial in these (너의 의미), plain [이] inside these words
_K_UI_START = {"의미", "의지", "의자", "의사", "의심", "의문", "의무", "의견", "의식", "의도",
               "의존", "의리", "의외", "의논", "의상", "의욕", "의아", "의젓", "의원", "의학"}
_K_UI_INNER = {"거의", "주의", "예의", "정의", "동의", "회의", "합의", "강의", "논의", "협의",
               "결의", "모의", "호의", "성의", "편의", "임의", "의의", "창의", "건의", "토의",
               "고의", "열의", "문의", "본의", "선의", "악의"}


def _is_hangul(c: str) -> bool:
    return "가" <= c <= "힣"


def has_hangul(text: str) -> bool:
    return any(_is_hangul(c) or c in _K_JAMO_HUM for c in text)


def hangul_syllables(text: str) -> list[list[str]]:
    """Hangul syllables of text as [initial, medial, final] spellings."""
    out = []
    for ch in text:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out.append([_K_INI[o // 588], _K_MED[(o % 588) // 28], _K_FIN[o % 28]])
    return out


def _k_compose(ini: str, med: str, fin: str) -> str:
    try:
        return chr(0xAC00 + (_K_INI.index(ini) * 21 + _K_MED.index(med)) * 28 + _K_FIN.index(fin))
    except ValueError:
        return "?"


# Revised Romanization (plus a few common informal spellings)
_RR_INI = [("kk", "kk"), ("tt", "tt"), ("pp", "pp"), ("ss", "ss"), ("jj", "jj"), ("ch", "ch"),
           ("sh", "s"), ("g", "g"), ("k", "k"), ("n", "n"), ("d", "d"), ("t", "t"),
           ("r", "r"), ("l", "r"), ("m", "m"), ("b", "b"), ("p", "p"), ("s", "s"),
           ("j", "j"), ("h", "h"), ("", "")]
_RR_MED = [("yae", "yae"), ("yeo", "yeo"), ("wae", "wae"), ("ae", "ae"), ("ya", "ya"),
           ("eo", "eo"), ("ye", "ye"), ("wa", "wa"), ("oe", "oe"), ("yo", "yo"), ("wo", "wo"),
           ("we", "we"), ("wi", "wi"), ("yu", "yu"), ("eu", "eu"), ("ui", "ui"), ("a", "a"),
           ("e", "e"), ("o", "o"), ("u", "u"), ("i", "i")]
_RR_FIN = [("ng", "ng"), ("kk", "kk"), ("k", "g"), ("g", "g"), ("n", "n"), ("t", "d"),
           ("d", "d"), ("l", "l"), ("m", "m"), ("p", "b"), ("b", "b"), ("ss", "ss"),
           ("s", "s"), ("ch", "ch"), ("j", "j"), ("", "")]
# vocables that should never be read as romanized Korean.  In romanization
# an h or w is always followed by a vowel, so "dah", "uh", "aw" are vocables
# while "gomawo", "guhae", "ohae" are Korean.
_KO_VOCABLE = re.compile(r"(oo|ee|aa|(ah|oh|uh|eh)(?![aeiouwy])|aw(?![aeiouy])|ow$|^[^aeiouy]+$)")
_KO_VOCABLE_SET = {"dum", "dun", "bum", "bun", "um", "hum", "bop", "bob", "dit", "dat", "dut",
                   "dop", "dap", "dot", "doot"}
# common one-syllable English words that romanization would misread
_KO_ENGLISH = {"me", "we", "be", "he", "she"}


def _ko_vocable(t: str) -> bool:
    return t in _KO_VOCABLE_SET or bool(_KO_VOCABLE.search(t))


def _ko_english(t: str) -> bool:
    """An English word in romanized-Korean lyrics: a dictionary word that
    romanization would have to split into several syllables ("time",
    "you", "rain"), or a short word romanization misreads."""
    if t in _KO_ENGLISH:
        return True
    if t not in _cmu_dict():
        return False
    r = rr_syllables(t)
    return r is None or len(r) >= 2


def rr_syllables(tok: str) -> list[list[str]] | None:
    """Split a romanized Korean token ("saranghae", "neol", "kkum") into
    [initial, medial, final] spellings, or None if it isn't one.  Between
    vowels, consonants go to the following syllable where they can, except
    that ng stays a final (sarangi = 사랑이); "maeum" is 마음, not 매움."""
    tok = tok.lower()
    if not tok or not re.fullmatch(r"[a-z]+", tok):
        return None

    @lru_cache(maxsize=None)
    def best(i: int):
        if i == len(tok):
            return (0, ())
        found = None
        for isp, ini in _RR_INI:
            if not tok.startswith(isp, i):
                continue
            j = i + len(isp)
            for msp, med in _RR_MED:
                if not tok.startswith(msp, j):
                    continue
                k = j + len(msp)
                for fsp, fin in _RR_FIN:
                    if not tok.startswith(fsp, k):
                        continue
                    m = k + len(fsp)
                    rest = best(m)
                    if rest is None:
                        continue
                    cost = rest[0] + 1
                    if m < len(tok):
                        cost += 2 * (1 if fsp == "ng" else len(fsp))
                    if med in ("ae", "oe") and not fsp and tok.startswith("u", k):
                        cost += 1
                    if found is None or cost < found[0]:
                        found = (cost, ((ini, med, fin),) + rest[1])
        return found

    r = best(0)
    return [list(s) for s in r[1]] if r else None


def korean_sounds(sylls: list[list[str]], links: list[bool], hums: set | None = None) -> list:
    """Standard Korean pronunciation across a sung line.

    sylls: [initial, medial, final] spellings; links[i] is True when syllable
    i runs straight into i+1 (no rest or breath between).  Applies linking
    of finals into a following vowel (음악 -> 으막, 없어 -> 업써), h-deletion
    and aspiration (좋아 -> 조아, 좋다 -> 조타, 축하 -> 추카), nasalization
    (합니다 -> 함니다), ㄹ rules (신라 -> 실라, 종로 -> 종노), tensing after
    stops and after the -ㄹ ending (학교 -> 학꾜, 할 거야 -> 할꺼야),
    palatalization (같이 -> 가치) and the common lexical cases.  Returns one
    (onset, nucleus, coda) phoneme triple per syllable.  hums: indices of
    syllables sung as a hum (음, 흠, 응)."""
    S = [list(s) for s in sylls]
    n = len(S)
    chars = [_k_compose(*s) for s in S]
    orig_ini = [s[0] for s in S]
    orig_fin = [s[2] for s in S]
    hums = hums or set()
    sung = [bool(links[i]) for i in range(n)]          # sound continues into i+1
    links = [sung[i] and i not in hums and (i + 1) not in hums for i in range(n)]
    fincls = [""] * n

    def adnominal_l(i):
        c0, c1 = chars[i], chars[i + 1]
        if c0 == "를":
            return False
        if c0 == "을" and not (i > 0 and sung[i - 1] and
                              (chars[i - 1] in _K_EUL_STEMS or orig_fin[i - 1] == "ss")):
            return False
        if c1 in _K_L_DEP:
            return True
        if c1 == "줄" and i + 2 < n and chars[i + 2] in ("알", "몰"):
            return True
        return (c1 == "지" and c0 != "날" and i + 2 < n
                and chars[i + 2] in ("도", "는", "몰", "모", "라", "요", "만"))

    for i in range(n):
        F = S[i][2]
        if not (i + 1 < n and links[i]):
            fincls[i] = _K_CLASS.get(F, "")
            continue
        nx = S[i + 1]
        c0, c1 = chars[i], chars[i + 1]
        if nx[0] == "" and F:
            cls = _K_CLASS.get(F, "")
            if c0 == "못" and nx[1] in _K_IY_V:
                nx[0] = "n"                          # 못 잊어 -> 몬니저
            elif ((c0 in _K_WORD_END and not (c0 == "몇" and nx[1] == "i")) or c1 == "없") \
                    and cls in _K_LENIS_OF:
                # a word ends here: the final is reduced, then links
                S[i][2] = ""
                nx[0] = _K_LENIS_OF[cls]
                continue
            elif c1 == "잎":
                nx[0] = "n"                          # 꽃잎 -> 꼰닙, 나뭇잎 -> 나문닙
        I = nx[0]
        if I == "":
            # a final runs into a following vowel
            if F in ("", "ng"):
                pass
            elif F == "h":
                F = ""
            elif F in ("nh", "lh"):
                nx[0] = "n" if F == "nh" else "r"
                F = ""
            elif F in _K_DOUBLE:
                F, second = _K_DOUBLE[F]
                nx[0] = "ss" if second == "s" else second
            else:
                nx[0] = "r" if F == "l" else F
                F = ""
            # palatalization: 굳이 -> 구지, 같이 -> 가치, 붙여 -> 부처
            if orig_fin[i] in ("d", "t", "lt") and nx[1] in ("i", "yeo") and nx[0] in ("d", "t"):
                nx[0] = "j" if nx[0] == "d" else "ch"
            S[i][2] = F
            fincls[i] = _K_CLASS.get(F, "")
            continue
        # final + consonant
        if F in ("h", "nh", "lh") and (I in _K_ASP or I == "s"):
            nx[0] = _K_ASP.get(I, "ss")
            F = {"h": "", "nh": "n", "lh": "l"}[F]
        elif F == "h" and I == "n":
            F = "n"
        elif I == "h" and F in _K_H_ASP:
            keep, asp = _K_H_ASP[F]
            if asp in ("t", "ch"):
                # 닫히다/맞히다 -> 다치다/마치다, but 꽃 한 송이 -> 꼬탄
                asp = "ch" if nx[1] in ("i", "yeo") else "t"
            nx[0] = asp
            F = keep
        cls = _K_CLASS.get(F, "")
        if c0 == "밟" and F == "lb":
            cls = "P"                                # 밟다 -> 밥따, 밟는 -> 밤는
        I = nx[0]
        # verb stems ending in a double consonant: 읽고 -> 일꼬, 앉다 -> 안따,
        # 젊지 -> 점찌, 넓다 -> 널따, 밟다 -> 밥따
        if F == "lg" and I == "g":
            cls = "L"
            nx[0] = "kk"
        elif F in ("nj", "lm", "lb", "lt") and I in _K_TENSE and c0 not in ("삶", "앎", "여덟"):
            nx[0] = _K_TENSE[I]
        I = nx[0]
        if I == "r":
            if cls == "L":
                nx[0] = "l"
            elif cls == "N":
                cls, nx[0] = "L", "l"
            elif cls in ("M", "NG"):
                nx[0] = "n"
            elif cls in _K_NASALIZE:
                cls, nx[0] = _K_NASALIZE[cls], "n"
        elif I == "n" and cls == "L":
            nx[0] = "l"
        elif I in ("n", "m") and cls in _K_NASALIZE:
            cls = _K_NASALIZE[cls]
        elif I in _K_TENSE and (cls in _K_NASALIZE or (c0 + c1) in _K_TENSE_PAIRS
                                or (F == "l" and adnominal_l(i))):
            nx[0] = _K_TENSE[I]
        S[i][2] = F
        fincls[i] = cls

    out = []
    for i, (I, M, F) in enumerate(S):
        if i in hums:
            hum = {"m": "m", "ng": "N", "n": "n"}.get(F, "m")
            out.append((["h"] if I == "h" else [], [hum], []))
            continue
        # after a vowel, a sonorant or a hum, a lenis consonant is voiced
        after_voiced = (i > 0 and sung[i - 1]
                        and ((i - 1) in hums or fincls[i - 1] not in ("K", "T", "P")))
        glide, nuc = _K_MED_PH[M]
        glide, nuc = list(glide), list(nuc)
        if I in ("j", "jj", "ch") and glide == ["j"]:
            glide = []                      # 져 = 저
        if M == "ui":
            if orig_ini[i]:
                nuc = ["i"]                                     # 희망 -> 히망
            elif not (i > 0 and sung[i - 1]):
                pass                                            # 의사: [의]
            elif i + 1 < n and chars[i] + chars[i + 1] in _K_UI_START:
                pass                                            # 너의 의미
            elif i > 0 and chars[i - 1] + chars[i] in _K_UI_INNER:
                nuc = ["i"]                                     # 거의, 주의
            else:
                nuc = ["e"]                                     # 나의 -> 나에
        if I in ("g", "d", "b", "j"):
            onset = [({"g": "g", "d": "d", "b": "b", "j": "dZ"} if after_voiced
                      else {"g": "k", "d": "t", "b": "p", "j": "tS"})[I]]
        elif I in ("kk", "tt", "pp", "jj"):
            onset = [{"kk": "k'", "tt": "t'", "pp": "p'", "jj": "tS'"}[I]]
        elif I in ("k", "t", "p", "ch"):
            onset = [{"k": "kh", "t": "th", "p": "ph", "ch": "tSh"}[I]]
        elif I in ("s", "ss"):
            onset = [("S" if M in _K_PALATAL_V else "s") + ("'" if I == "ss" else "")]
        else:
            onset = {"": [], "h": ["h"], "n": ["n"], "m": ["m"], "r": ["R"], "l": ["l"]}[I]
        coda = [_K_CODA_PH[fincls[i]]] if fincls[i] else []
        out.append((onset + glide, nuc, coda))
    return out


def _korean_piece_sylls(piece: str, romanized: bool) -> tuple[list, bool]:
    """Syllable spellings for one lyric piece, and whether it is a hum
    (음, 흠, 응, 음음, 으음 ...)."""
    txt = piece.strip()
    if any(_is_hangul(c) for c in txt):
        sylls = hangul_syllables(txt)
    elif any(c in _K_JAMO_HUM for c in txt):
        fin = {"m": "m", "n": "n", "N": "ng"}[_K_JAMO_HUM[[c for c in txt if c in _K_JAMO_HUM][0]]]
        return [["", "eu", fin]], True
    elif romanized:
        sylls = rr_syllables(_norm(txt)) or []
    else:
        return [], False
    humlike = (bool(sylls) and all(s[1] == "eu" and s[0] in ("", "h") and s[2] in ("", "m", "ng")
                                   for s in sylls)
               and any(s[2] in ("m", "ng") for s in sylls))
    if humlike:
        last = [s for s in sylls if s[2] in ("m", "ng")][-1]
        return [[sylls[0][0], "eu", last[2]]], True
    return sylls, False


def looks_romanized_korean(tokens, words=None) -> bool:
    """Latin lyrics that are clearly romanized Korean: a good share of the
    words have the romanization vowels 'eo'/'eu' (neo, geu, eo-di, ...),
    which neither English nor Japanese romaji uses, and most of the words
    parse as Korean syllables.  words: the lyric words as lists of their
    hyphenated pieces; pieces of an English word ("peo-ple") don't count."""
    english = set()
    for w in words or []:
        if len(w) > 1 and _norm("".join(w)) in _cmu_dict():
            english.update(_norm(p) for p in w)
    toks = {_norm(t) for t in tokens if t}
    toks = {t for t in toks if t and re.fullmatch(r"[a-z]+", t) and t not in english}
    words = {t for t in toks if not _ko_vocable(t)}
    if not words:
        return False
    ko = {t for t in words if rr_syllables(t) is not None and not _ko_english(t)}
    marked = {t for t in ko if re.search(r"eo|eu", t)}
    return len(marked) >= 3 and len(marked) >= 0.1 * len(words) and len(ko) >= 0.6 * len(words)


_PINYIN_MARK = re.compile(r"^(x|q|zh|c[aeiou]|z[aeiou])|(iang|iong|uang|ueng|ian|iao|ong|eng|ui|iu|ie|uo|ua)$")


def looks_pinyin(tokens) -> bool:
    """Latin lyrics written in pinyin (xin, zhuang, qing, guang, shuo...)."""
    toks = {re.sub(r"[1-5]$", "", _norm(t)) for t in tokens if t}
    toks = {t for t in toks if t and re.fullmatch(r"[a-zü]+", t)}
    py = {t for t in toks if pinyin_syllable(t) is not None}
    marked = {t for t in py if _PINYIN_MARK.search(t) and t not in _JA_MARKERS}
    return len(marked) >= 5 and len(py) >= 0.5 * len(toks)


def detect_language(tokens, words=None) -> str:
    """'ko' for Hangul, 'ja' for kana or Latin lyrics that look like
    Japanese romaji, 'zh' for Chinese characters, 'ko-rom' for romanized
    Korean, else 'en'.  words (optional): lyric words as lists of pieces."""
    tokens = [t for t in tokens if t]
    toks = {_norm(t) for t in tokens}
    ja = {t for t in toks if t in _JA_MARKERS}
    if any(classify(t) == "hangul" for t in tokens):
        return "ko"
    if any(classify(t) == "kana" for t in tokens):
        return "ja"
    if any(classify(t) == "cjk" for t in tokens):
        return "zh"
    # romanized Korean shares many syllables with romaji (ri, mu, jo, ne...),
    # so it is tested first, on the vowels only it uses
    if looks_romanized_korean(tokens, words):
        return "ko-rom"
    # pinyin also shares syllables with romaji (shi, chi, ni, wo); it is told
    # apart by spellings romaji and English vocables never use
    if looks_pinyin(tokens):
        return "zh"
    return "ja" if len(ja) >= 5 else "en"


# ------------------------------------------------------------------- public
def _is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"


def _is_kana(c: str) -> bool:
    return "぀" <= c <= "ヿ"


def classify(text: str) -> str:
    if has_hangul(text):
        return "hangul"
    if any(_is_kana(c) for c in text):
        return "kana"
    if any(_is_cjk(c) for c in text):
        return "cjk"
    return "latin"


def has_vowel_letter(tok: str) -> bool:
    return bool(re.search(r"[aeiouy]", tok.lower()))


def consonant_piece(tok: str) -> list[str]:
    """Phonemes for a consonant-only lyric piece like 'l', 'll', 't', 'ng', 'nd'."""
    t = _norm(tok)
    out: list[str] = []
    i = 0
    while i < len(t):
        for sp, ph in (("ng", ["N"]), ("ll", ["l"]), ("ss", ["s"]), ("tt", ["t"]),
                       ("sh", ["S"]), ("ch", ["tS"]), ("th", ["T"]), ("ck", ["k"])):
            if t.startswith(sp, i):
                out += ph
                i += len(sp)
                break
        else:
            ph = dict(_FALLBACK_LETTERS).get(t[i])
            if ph and all(p in CONSONANTS for p in ph):
                out += ph
            i += 1
    return out


class Phonemizer:
    """Stateful so it can collect warnings and know the score's language."""

    def __init__(self, lang: str = "en"):
        self.lang = lang  # en | zh | ja | ko (Hangul) | ko-rom (romanized Korean)
        self.warnings: list[str] = []
        self._warned: set[str] = set()

    def _warn(self, msg: str):
        if msg not in self._warned:
            self._warned.add(msg)
            self.warnings.append(msg)

    # ----------------------------------------------------------- Korean
    def is_korean(self, text: str) -> bool:
        """Hangul always is; Latin is read as romanized Korean only in a
        romanized-Korean score, and never when it looks like a vocable."""
        if not text:
            return False
        if has_hangul(text):
            return True
        if self.lang == "ko-rom":
            t = _norm(text)
            return (bool(t) and not _ko_vocable(t) and not _ko_english(t)
                    and rr_syllables(t) is not None)
        return False

    def korean_line(self, pieces: list[str], links: list[bool]) -> list:
        """Phonemize a run of Korean lyric pieces (one per note).  links[k]
        says whether piece k runs straight into piece k+1; Korean sound
        changes apply across those joins, as they do when it is sung.
        Returns a Syllable (or Multi, for several syllables under one note)
        per piece."""
        romanized = self.lang == "ko-rom"
        parsed = []
        for p in pieces:
            sylls, hum = _korean_piece_sylls(p, romanized)
            if not sylls:
                self._warn(f"couldn't read Korean lyric '{p}'; sung on 'ah'")
                sylls, hum = [["", "a", ""]], False
            parsed.append((sylls, hum))
        # 음/흠/응 is a hum unless it is part of a word: run on from a
        # syllable (마음, 처음, 웃음), linked into a following vowel (음악), or,
        # in a line without other clear hums, run into a consonant (음식).
        n = len(pieces)
        cand = [h for _s, h in parsed]

        def open_vowel(k):        # 아, 오, 우, 이...: a vocable, not a word start
            sy = parsed[k][0]
            return not cand[k] and len(sy) == 1 and sy[0][0] == "" and sy[0][2] == ""

        alone_before = [k == 0 or not links[k - 1] or cand[k - 1] or open_vowel(k - 1)
                        for k in range(n)]
        into_vowel = [bool(links[k]) and k + 1 < n and not cand[k + 1]
                      and parsed[k + 1][0][0][0] == "" for k in range(n)]
        ends = [not links[k] or k + 1 >= n or cand[k + 1] for k in range(n)]
        backing = sum(1 for k in range(n) if cand[k] and alone_before[k] and ends[k]) >= 2
        hum_piece = [cand[k] and alone_before[k] and not into_vowel[k] and (backing or ends[k])
                     for k in range(n)]
        flat, owner, flat_links, hums = [], [], [], set()
        for k, (sylls, _h) in enumerate(parsed):
            for j, s in enumerate(sylls):
                if hum_piece[k]:
                    hums.add(len(flat))
                flat.append(s)
                owner.append(k)
                flat_links.append(True if j + 1 < len(sylls) else bool(links[k]))
        sounds = korean_sounds(flat, flat_links, hums)
        out: list = [[] for _ in pieces]
        for k, s in zip(owner, sounds):
            out[k].append(s)
        return [ss[0] if len(ss) == 1 else Multi(ss) for ss in out]

    def word(self, pieces: list[str]) -> list[Syllable]:
        """Phonemize one lyric word given as its syllable pieces
        (["to", "night"] for a hyphenated word, ["doo"] for a single one).
        Returns one syllable per piece.  A consonant-only piece after the
        first ('l' in 'fu-l', 't' in 'bu-t') comes back with an empty
        nucleus and the consonant(s) as coda."""
        n = len(pieces)
        joined = "".join(pieces)
        kind = classify(joined)

        if kind == "hangul" or (self.lang == "ko-rom" and all(self.is_korean(p) for p in pieces)):
            return self.korean_line(pieces, [True] * (n - 1) + [False])
        if kind in ("cjk", "kana"):
            return [self._east_asian_piece(p) for p in pieces]

        norm = [_norm(p) for p in pieces]
        cons_piece = [k > 0 and bool(t) and not has_vowel_letter(t) for k, t in enumerate(norm)]

        if self.lang == "ja":
            out = []
            for t, cp in zip(norm, cons_piece):
                if cp:
                    out.append(([], [], consonant_piece(t)))
                else:
                    out.append(self._ja_piece(t))
            return out

        if self.lang == "zh" and n > 1 and _norm(joined) in _cmu_dict():
            aligned = self._align_cmu(pieces, [_norm(p) for p in pieces],
                                      [k > 0 and not has_vowel_letter(_norm(p)) for k, p in enumerate(pieces)])
            if aligned:
                return aligned
        if self.lang == "zh":
            out = []
            for t, cp in zip(norm, cons_piece):
                if cp:
                    out.append(([], [], consonant_piece(t)))
                    continue
                s = pinyin_syllable(t)
                out.append(s or self._latin_piece(t))
            return out

        if n > 1:
            aligned = self._align_cmu(pieces, norm, cons_piece)
            if aligned:
                return aligned
        out = []
        for raw, t, cp in zip(pieces, norm, cons_piece):
            if cp:
                out.append(([], [], consonant_piece(t)))
            else:
                out.append(self._latin_piece(t, raw))
        return out

    def _align_cmu(self, pieces, norm, cons_piece):
        """Split the dictionary pronunciation of the whole word between the
        written pieces by aligning letters to phonemes."""
        raw = "".join(pieces).lower().replace("’", "'")
        key = re.sub(r"[^a-z']", "", raw)
        d = _cmu_dict()
        cands = [key, key.replace("'", "")]
        # arrangers double letters across the split ("long-ger") or repeat
        # vowels for a run ("chi-i-ild")
        parts = [re.sub(r"[^a-z']", "", p.lower()) for p in pieces]
        dedup = parts[0]
        for p in parts[1:]:
            dedup += p[1:] if dedup and p and p[0] == dedup[-1] else p
        cands += [dedup, re.sub(r"([aeiouy])\1+", r"\1", key), re.sub(r"(.)\1+", r"\1", key)]
        prons = None
        for c in cands:
            if c in d:
                prons = d[c]
                break
        if not prons:
            return None
        letters, owner = [], []
        for k, p in enumerate(pieces):
            for ch in p.lower():
                if "a" <= ch <= "z":
                    letters.append(ch)
                    owner.append(k)
        if not letters:
            return None
        phones = [re.sub(r"\d", "", x) for x in prons[0]]
        stress0 = [x.endswith("0") for x in prons[0]]
        path = _align_letters(letters, phones, owner)
        if path is None:
            return None
        per_piece: list[list[str]] = [[] for _ in pieces]
        for j, li in enumerate(path):
            base = phones[j]
            ph = ["@"] if (base == "AH" and stress0[j]) else _ARPA.get(base, [])
            per_piece[owner[li]].extend(ph)
        out = []
        for k, ph in enumerate(per_piece):
            t = norm[k]
            has_v = any(p in VOWELS for p in ph)
            prev_v = re.findall(r"[aeiouy]", norm[k - 1]) if k > 0 else []
            starts_vowel = bool(t) and t[0] in "aeiouy"
            only_y = bool(t) and not re.search(r"[aeiou]", t)
            if has_v:
                out.append(_phones_to_syllable(ph))
            elif k == 0:
                out.append(self._latin_piece(t, pieces[k]))
            elif not has_vowel_letter(t) or only_y or (starts_vowel and prev_v and t[0] == prev_v[-1]):
                # no vowel sound of its own: hold the vowel, then these consonants
                out.append(([], [], ph))
            else:
                out.append(self._latin_piece(t, pieces[k], warn=False))
        # singers carry a consonant over to the next note ("ham-mer" -> "ha-mer")
        for k in range(1, len(out)):
            o, n, c = out[k]
            po, pn, pc = out[k - 1]
            if n and not o and pc and norm[k] and norm[k][0] not in "aeiouy" and pn:
                out[k - 1] = (po, pn, pc[:-1])
                out[k] = ([pc[-1]], n, c)
        return out

    def _ja_piece(self, t: str) -> Syllable:
        if not t:
            return ([], ["@"], [])
        if len(t) > 2 and t[0] == t[1] and t[0] not in "aeiouyn":
            t = t[1:]          # "tte" after a small tsu: the closure is on the previous note
        if not has_vowel_letter(t):
            h = _parse_hum(t)
            if h:
                return h
        if is_romaji(t) and not _VOCABLE_HINT.search(t):
            s = romaji_syllables(t)
            if s:
                return _ja_group(s)
        o, n, c = self._latin_piece(t)
        # vocables in a Japanese-language score use the Japanese 'a' ("dat" = "daht")
        return (o, ["a" if v == "ae" else v for v in n], c)

    def _latin_piece(self, tok: str, raw: str | None = None, warn: bool = True) -> Syllable:
        if not tok:
            return ([], ["@"], [])
        if raw and "'" in raw.replace("’", "'"):
            cmu = _cmu_syllables(re.sub(r"[^a-z']", "", raw.lower().replace("’", "'")))
            if cmu:
                return _merge_syllables(cmu)
        if tok in _EXCEPTIONS:
            o, nu, c = _EXCEPTIONS[tok]
            return (list(o), list(nu), list(c))
        if not has_vowel_letter(tok):
            h = _parse_hum(tok)
            if h:
                return h
            # a lone obstruent like "d" or "t": a clipped ghost hit
            ph = consonant_piece(tok)
            if ph:
                return (ph, ["n"], ["*ghost*"])
        g = _parse_acappella(tok)
        in_cmu = tok in _cmu_dict()
        # Prefer the dictionary for real words, the grammar for vocables.
        vocable_like = bool(re.search(r"(oo|ah|oh|eh|uh|aa|ee|aw)", tok)) or len(tok) <= 3
        if g and (vocable_like or not in_cmu):
            return g
        cmu = _cmu_syllables(tok)
        if cmu:
            return _merge_syllables(cmu)
        if g:
            return g
        if warn:
            self._warn(f"guessed pronunciation for '{tok}'")
        return _phones_to_syllable(_fallback_phones(tok))

    def _east_asian_piece(self, p: str) -> Syllable:
        chars = [c for c in p if _is_cjk(c) or _is_kana(c)]
        if any(_is_kana(c) for c in chars) or self.lang == "ja":
            if any(_is_cjk(c) for c in chars):
                self._warn("kanji lyrics can't be read without a dictionary; "
                           "they're sung on 'ah' (use kana or romaji lyrics)")
                return ([], ["a"], [])
            s = romaji_syllables(kana_to_romaji(p))
            return _ja_group(s) if s else ([], ["a"], [])
        sylls = [s for s in (_chinese_syllable(c) for c in chars) if s]
        if not sylls:
            self._warn("Chinese lyrics need the 'pypinyin' package; sung on 'ah'")
            return ([], ["a"], [])
        return _merge_syllables(sylls)


_VOICELESS = {"k", "s", "t", "p", "S", "tS", "ts", "h", "f", "C"}


def _ja_group(sylls: list[Syllable]):
    """Morae under one note.  A final i/u after a voiceless consonant is
    devoiced ("kasu" -> "kas", "desu" -> "des"); the n mora joins the
    previous vowel as a coda; anything else stays a separate syllable."""
    out: list = []
    for k, (o, n, c) in enumerate(sylls):
        last = k == len(sylls) - 1
        if out and n == ["n"] and not o:
            po, pn, pc = out[-1]
            out[-1] = (po, pn, pc + ["n"] + c)
            continue
        if out and last and n in (["i"], ["M"]) and o and o[-1] in _VOICELESS:
            po, pn, pc = out[-1]
            out[-1] = (po, pn, pc + o + c)
            continue
        if out and not o and n and n[0] in VOWELS and len(out[-1][1]) == 1 and not out[-1][2]:
            po, pn, pc = out[-1]
            out[-1] = (po, pn + n, c)          # vowel run -> diphthong
            continue
        out.append((list(o), list(n), list(c)))
    return out[0] if len(out) == 1 else Multi(out)


def _merge_syllables(sylls: list[Syllable]) -> Syllable:
    """Squash several syllables into one note's worth of sound."""
    if len(sylls) == 1:
        return sylls[0]
    onset = list(sylls[0][0])
    nucleus = list(sylls[0][1])
    coda: list[str] = list(sylls[0][2])
    for o, nu, c in sylls[1:]:
        if len(nucleus) == 1 and not coda and not o and nu and nu[-1] in VOWELS:
            nucleus.append(nu[-1])          # vowel + vowel -> diphthong
            coda += c
            continue
        coda += o + [v for v in nu if v in CONSONANTS] + c
    return (onset, nucleus[:2], coda)


if __name__ == "__main__":
    ph = Phonemizer("en")
    for t in ["doo", "dn", "dm", "oh", "woh", "hoo", "loo", "dah", "bahm", "dit", "jow",
              "doom", "mm", "jn", "ooh", "night", "dop", "d", "whoo", "doot", "dut", "for",
              "show", "know", "still", "when", "will", "I'm", "your", "soul", "thing", "now"]:
        print(t, ph.word([t]))
    for w in (["beau", "ti", "fu", "l"], ["wi", "ll"], ["bu", "t"], ["yo", "ung"], ["Ju", "l", "y"],
              ["to", "night"], ["e", "lec", "tric"], ["a", "ching"]):
        print(w, ph.word(w))
    j = Phonemizer("ja")
    for w in (["ka", "e", "rou"], ["kasu"], ["you"], ["tsu", "ki"], ["shi", "n", "ku"], ["da"],
              ["dm"], ["dat"], ["te"], ["hoo"], ["tte"], ["n"], ["fon"], ["doo"], ["woh"]):
        print(w, j.word(w))
    print([ph.word([c]) for c in "跳楼机"])
    print(j.word(["かえ", "ろう"]), j.word(["きょ"]))
    print(ph.warnings, j.warnings)
    k = Phonemizer("ko")
    for line in ("사랑해", "음악", "없어", "좋아", "좋다", "학교", "합니다", "신라", "같이", "축하해",
                 "밖에", "나의", "희망", "많아", "읽어", "종로", "국물", "꿈", "쌀", "짜증", "음"):
        print(line, k.korean_line(list(line), [True] * (len(line) - 1) + [False]))
    r = Phonemizer("ko-rom")
    for t in ("saranghae", "neol", "bogo", "sipeo", "eotteoke", "hamkke", "kkum", "doo", "dm"):
        print(t, rr_syllables(t), r.is_korean(t))
