"""Summarization engine: extractive (TF-IDF style scoring) + AI (abstractive) + hybrid.

Pipeline
  text -> clean -> sentences -> (chunking for long docs) -> select / rewrite -> verify -> format
"""
import math
import re
from collections import Counter

import ai_handler
from ai_handler import AIUnavailable

# ----------------------------------------------------------------- constants
MIN_WORDS = 30                 # below this the text is too short to summarize
MAX_WORDS = 400_000            # hard safety cap
CHUNK_WORDS = 1500             # extractive chunk size for long documents
AI_INPUT_WORDS = 3000          # max words sent to the AI in one request
PRESETS = {                    # ratio of original, min words, max words
    "short": (0.10, 40, 120),
    "medium": (0.20, 100, 250),
    "long": (0.35, 200, 500),
}
STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or other
our out over own same she should so some such than that the their theirs them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your yours said
says say may might must shall also however thus""".split())
TERMINATORS = ".!?\u0964\u06d4"          # includes Hindi danda (।) and Urdu full stop
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig", "no",
                 "inc", "ltd", "co", "u.s", "u.k", "approx", "dept", "est", "vol", "al"}

_nltk_state = {"checked": False, "ok": False}


# ----------------------------------------------------------------- sentences
def _nltk_split(text):
    """Use NLTK's punkt tokenizer when installed (with data); else None."""
    if not _nltk_state["checked"]:
        _nltk_state["checked"] = True
        try:
            from nltk.tokenize import sent_tokenize
            sent_tokenize("Test one. Test two.")
            _nltk_state["ok"] = True
        except Exception:
            _nltk_state["ok"] = False
    if not _nltk_state["ok"]:
        return None
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def _regex_split(text):
    pieces = re.split(rf"(?<=[{re.escape(TERMINATORS)}])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9\u0900-\u097F])", text)
    merged, buf = [], ""
    for piece in pieces:
        buf = f"{buf} {piece}".strip() if buf else piece
        last = buf.split()[-1] if buf.split() else ""
        low = last.lower().rstrip(".")
        if low in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", last):
            continue                      # abbreviation/initial: keep joining
        merged.append(buf)
        buf = ""
    if buf:
        merged.append(buf)
    return merged


def split_sentences(text):
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        parts = _nltk_split(para) or _regex_split(para)
        for s in parts:
            s = s.strip()
            if s:
                sentences.extend(_break_long(s))
    return sentences


def _break_long(sentence, limit=80, size=30):
    """Text without punctuation (e.g. raw OCR/lists) becomes ~30-word pseudo-sentences."""
    words = sentence.split()
    if len(words) <= limit:
        return [sentence]
    return [" ".join(words[i:i + size]) + "." for i in range(0, len(words), size)]


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def _tokens(s):
    return re.findall(r"[^\W_][\w'\-]*", s.lower())


def _content_words(s):
    return [w for w in _tokens(s) if w not in STOPWORDS and len(w) > 2]


# ------------------------------------------------------------------- chunking
def chunk_text(text, max_words=CHUNK_WORDS):
    """Split a long document into chunks on sentence boundaries."""
    chunks, cur, count = [], [], 0
    for s in split_sentences(text):
        w = word_count(s)
        if cur and count + w > max_words:
            chunks.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------- extractive
def _score(sentences):
    n = len(sentences)
    cw = [_content_words(s) for s in sentences]
    freq, df = Counter(), Counter()
    for words in cw:
        freq.update(words)
        df.update(set(words))
    weight = {w: freq[w] * math.log(1 + n / df[w]) for w in freq}
    top = max(weight.values(), default=1) or 1
    scores = []
    for i, (s, words) in enumerate(zip(sentences, cw)):
        uniq = set(words)
        base = sum(weight[w] / top for w in uniq) / (len(uniq) ** 0.6 + 1)
        bonus = 1.0
        if re.search(r"\d", s):
            bonus += 0.10                                          # numbers / dates carry facts
        bonus += min(0.15, 0.03 * len(re.findall(r"(?<![.!?]\s)\b[A-Z][a-z]+", s[1:])))  # names
        if i == 0:
            bonus += 0.30                                          # opening sentence
        elif i < max(2, n // 20):
            bonus += 0.10
        wc = len(s.split())
        if wc > 45 or wc < 8:
            bonus -= 0.20
        if re.match(r"(?i)(however|but|and|also|this|that|these|those|he|she|they|it)\b", s):
            bonus -= 0.08                                          # likely needs context
        scores.append(base * max(bonus, 0.3))
    return scores


def _candidate_mask(sentences):
    keep = []
    for s in sentences:
        wc = len(s.split())
        is_heading = wc < 12 and s[-1] not in TERMINATORS
        keep.append(wc >= 5 and not is_heading)
    return keep


def extractive_sentences(text, target_words=None, target_sentences=None):
    """Return the selected sentences in original order."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    scores = _score(sentences)
    mask = _candidate_mask(sentences)
    order = sorted((i for i in range(len(sentences)) if mask[i]), key=lambda i: scores[i], reverse=True)
    chosen, chosen_sets, total = [], [], 0
    for i in order:
        s_set = set(_content_words(sentences[i]))
        if any(s_set and c and len(s_set & c) / len(s_set | c) > 0.5 for c in chosen_sets):
            continue                                               # avoid repetition
        w = word_count(sentences[i])
        if target_sentences:
            if len(chosen) >= target_sentences:
                break
        else:
            if total >= target_words:
                break
            if chosen and total + w > target_words * 1.25:
                continue
        chosen.append(i)
        chosen_sets.append(s_set)
        total += w
    if not chosen:                                                 # fallback: first sentences
        chosen = list(range(min(len(sentences), target_sentences or 3)))
    return [sentences[i] for i in sorted(chosen)]


def extractive_summary(text, target_words=None, target_sentences=None):
    """Extractive summary with map-reduce chunking for long documents."""
    total = word_count(text)
    if total <= CHUNK_WORDS * 1.3:
        return extractive_sentences(text, target_words, target_sentences)
    chunks = chunk_text(text)
    out = []
    for c in chunks:
        share = word_count(c) / total
        if target_sentences:
            k = max(1, round(target_sentences * share))
            out.extend(extractive_sentences(c, target_sentences=k))
        else:
            out.extend(extractive_sentences(c, target_words=max(15, target_words * share)))
    return _dedupe(out)


def _dedupe(sentences, threshold=0.5):
    """Drop near-duplicate sentences that were picked from different chunks."""
    kept, sets = [], []
    for s in sentences:
        cur = set(_content_words(s))
        if any(cur and o and len(cur & o) / len(cur | o) > threshold for o in sets):
            continue
        kept.append(s)
        sets.append(cur)
    return kept


# ------------------------------------------------------------------- targets
def compute_target(orig_words, length="medium", custom_type=None, custom_value=None):
    """Return (target_words, target_sentences)."""
    if custom_type:
        try:
            value = int(custom_value)
        except (TypeError, ValueError):
            raise ValueError("Custom length must be a whole number.")
        if custom_type == "sentences":
            if not 1 <= value <= 50:
                raise ValueError("Custom sentence count must be between 1 and 50.")
            return value * 22, value
        if custom_type == "words":
            if not 10 <= value <= 2000:
                raise ValueError("Custom word count must be between 10 and 2000.")
            return min(value, max(10, int(orig_words * 0.8))), None
        raise ValueError("Custom type must be 'words' or 'sentences'.")
    if length not in PRESETS:
        raise ValueError("Length must be short, medium or long.")
    ratio, lo, hi = PRESETS[length]
    target = max(lo, min(hi, int(orig_words * ratio)))
    return min(target, max(15, int(orig_words * 0.6))), None


# -------------------------------------------------------------- verification
def _numbers(text):
    text = re.sub(r"(?m)^\s*(\d+[.)]|[-\u2022*])\s+", "", text)
    return {n.rstrip(".,").replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def unsupported_numbers(source, summary):
    """Numbers in the summary that never appear in the source (possible hallucination)."""
    return _numbers(summary) - _numbers(source)


# ---------------------------------------------------------------- formatting
def _strip_bullet(line):
    return re.sub(r"^\s*(?:[-\u2022*]|\d+[.)])\s*", "", line).strip()


def to_bullets(sentences):
    return "\n".join(f"\u2022 {_strip_bullet(s)}" for s in sentences if s.strip())


def normalise_ai_output(text, bullets):
    text = re.sub(r"\*\*|__|^#+\s*", "", text, flags=re.M).strip()
    text = re.sub(r"(?i)^(here(?:'s| is) (?:a |the )?summary[^:\n]*:)\s*", "", text).strip()
    if bullets:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) <= 1:                                         # model ignored bullet format
            lines = split_sentences(text)
        return to_bullets(lines)
    return re.sub(r"\n{3,}", "\n\n", text)


# -------------------------------------------------------------- orchestration
def summarize(text, mode="hybrid", length="medium", custom_type=None, custom_value=None, bullets=False):
    """Main entry point. Returns a dict with the summary and metadata."""
    orig_words = word_count(text)
    if orig_words == 0:
        raise ValueError("Please enter or upload some text to summarize.")
    real_words = len(re.findall(r"[^\W\d_]{2,}", text))
    if real_words < MIN_WORDS and orig_words >= MIN_WORDS:
        raise ValueError("The text does not contain enough readable words to summarize.")
    if orig_words < MIN_WORDS:
        raise ValueError(f"Text is too short to summarize ({orig_words} words). Please provide at least {MIN_WORDS} words.")
    if orig_words > MAX_WORDS:
        raise ValueError(f"Text is too long ({orig_words:,} words). Maximum is {MAX_WORDS:,} words.")
    if mode not in ("extractive", "abstractive", "hybrid"):
        raise ValueError("Mode must be extractive, abstractive or hybrid.")

    target_w, target_s = compute_target(orig_words, length, custom_type, custom_value)
    warnings, method = [], "extractive"
    summary = None

    if mode != "extractive":
        try:
            summary = _ai_pipeline(text, mode, target_w, target_s, bullets)
            bad = unsupported_numbers(text, summary)
            if bad:
                warnings.append("AI output contained figures not found in the source "
                                f"({', '.join(sorted(bad)[:5])}); used extractive summary instead.")
                summary = None
            else:
                method = "abstractive" if mode == "abstractive" else "hybrid"
        except AIUnavailable as exc:
            warnings.append(f"{exc} Used extractive summary instead.")
            summary = None

    if summary is None:
        sents = extractive_summary(text, target_w if not target_s else None, target_s)
        summary = to_bullets(sents) if bullets else " ".join(sents)

    sw = word_count(summary)
    return {
        "summary": summary,
        "method": method,
        "original_words": orig_words,
        "summary_words": sw,
        "reduction_percent": round(100 * (1 - sw / orig_words), 1),
        "warnings": warnings,
    }


def _ai_pipeline(text, mode, target_w, target_s, bullets):
    orig = word_count(text)
    if mode == "hybrid" or orig > AI_INPUT_WORDS:
        # Extractive pre-selection keeps the AI grounded in real sentences and handles long documents.
        budget = min(AI_INPUT_WORDS, max(int(target_w * 2.5), 300))
        if orig > budget:
            text = " ".join(extractive_summary(text, target_words=budget))
    raw = ai_handler.summarize_with_ai(text, target_w, target_s, bullets)
    return normalise_ai_output(raw, bullets)
