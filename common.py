import re

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

def light_clean(text) -> str:
    """Minimal cleaning. Keeps every word, so it is SAFE to feed to translation/QA."""
    if text is None:
        return ""
    t = str(text)
    t = _TASHKEEL.sub("", t).replace(_TATWEEL, "")
    return re.sub(r"\s+", " ", t).strip()

def normalize_for_ml(text) -> str:
    """Letter normalisation for classical ML only (classification / retrieval).
    No stop-word removal and no stemming: those destroyed your translation input."""
    t = light_clean(text)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\u0621-\u064A0-9\s]", " ", t)   # drop punctuation
    return re.sub(r"\s+", " ", t).strip()
