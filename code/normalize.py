"""
normalize.py — Step 1: turn raw business names / addresses into comparable forms.

Handles the noise patterns observed in the training data:
  * accents / diacritics             ("Ínvestment", "Léarning")
  * Indic scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam...) -> Latin
  * leetspeak digits inside words     ("H0rizon", "5mart", "c0m")
  * website-style names               ("caangel.com", "www.x.com")
  * phone / id numbers glued on       ("Surgical Prairie - 4143439782")
  * honorifics & legal suffixes       ("Shri", "M/s", "Pvt Ltd", "LLC", "Inc")
  * duplicated words                  ("Palghar Palghar")
  * address formats: shuffled components, state names vs codes vs native script,
    street-type abbreviations, unit / PO box / house-number decorations.
"""
import json
import re
import unicodedata
from unidecode import unidecode

import config

# ----------------------------------------------------------------------------- shared helpers
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_HAS_ALPHA = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"[0-9]")


def _is_latin(tok: str) -> bool:
    for ch in tok:
        if ch.isalpha() and not ("LATIN" in unicodedata.name(ch, "")):
            return False
    return True


def to_ascii(s: str) -> str:
    return unidecode(s or "").lower()


def skeleton(tok: str) -> str:
    """Rough phonetic key: keep first letter, drop later vowels, collapse doubles.
    Makes transliterated Hindi ('limittedd', 'praaivett') line up with English
    ('limited', 'private')."""
    if not tok:
        return tok
    tok = tok.replace("ph", "f").replace("w", "v").replace("sh", "s").replace("z", "j")
    out = [tok[0]]
    for ch in tok[1:]:
        if ch in "aeiouyh":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


# ----------------------------------------------------------------------------- names
HONORIFICS = {"shri", "sri", "shree", "sree", "shrii", "srii", "smt", "mr", "mrs", "ms", "m s", "dr", "the", "messrs", "mess"}
LEGAL = {
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "llp", "l l p", "pc", "plc", "lp", "pllc", "india", "opc", "dba",
}
# skeletons of legal words so transliterated forms ('praaivett limittedd', 'praa li') are removed too
LEGAL_SKEL = {skeleton(w) for w in LEGAL if " " not in w} | {"pr", "l", "lmtd", "prvt"}
TLDS = ("com", "c0m", "net", "org", "in", "co", "io", "biz", "info", "us")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:" + "|".join(TLDS) + r")(?:\.[a-z]{2})?\b")


def _fix_leet(tok: str) -> str:
    # only rewrite digits inside tokens that also contain letters ('h0rizon' -> 'horizon')
    if _HAS_ALPHA.search(tok) and _HAS_DIGIT.search(tok):
        return tok.translate(_LEET)
    return tok


# Indic-script token -> English token, learned from training pairs by build_translit.py
_TRANSLIT_PATH = config.OUTPUT_DIR / "translit_dict.json"
TRANSLIT = json.loads(_TRANSLIT_PATH.read_text(encoding="utf-8")) if _TRANSLIT_PATH.exists() else {}
_INDIC = re.compile(r"[ऀ-ൿ]")
_TOK = re.compile(r"[^\s,()\[\]/.&+\-]+")


def translate_indic(raw: str) -> str:
    if not TRANSLIT or not _INDIC.search(raw):
        return raw
    return _TOK.sub(lambda m: TRANSLIT.get(m.group(0), m.group(0)), raw)


def normalize_name(raw: str) -> dict:
    """Returns dict with:
       tokens  : list of core tokens (legal words / honorifics removed)
       core    : ' '.join(tokens)
       compact : core with no spaces (lets 'caangel.com' match 'CA Angel')
       skel    : phonetic skeleton string (for transliterated names)
       is_domain / is_translit flags
    """
    raw = raw or ""
    is_translit = any(not _is_latin(t) for t in raw.split())
    s = to_ascii(translate_indic(raw)).strip()
    is_domain = False
    m = _DOMAIN.match(s.replace(" ", "")) if "." in s else None
    if m:
        s = m.group(1)
        is_domain = True
    s = s.replace("&", " and ").replace("+", " ")
    s = re.sub(r"\b\d{6,}\b", " ", s)          # glued-on phone / registration numbers
    toks = [_fix_leet(t) for t in _NON_ALNUM.split(s) if t]
    # merge single letters like 'm s' / 'l l c' before filtering
    joined = " ".join(toks)
    for multi in ("m s", "l l c", "l l p", "p c"):
        joined = re.sub(rf"\b{multi}\b", multi.replace(" ", ""), joined)
    toks = joined.split()
    core = []
    for t in toks:
        if t in HONORIFICS or t in LEGAL or t == "and":
            continue
        if is_translit and skeleton(t) in LEGAL_SKEL:
            continue
        if t.isdigit() and len(t) >= 4:
            continue
        if core and core[-1] == t:              # 'palghar palghar'
            continue
        core.append(t)
    if not core:                                # name was only legal words: keep something
        core = [t for t in toks if t != "and"][:3]
    core_s = " ".join(core)
    return {
        "tokens": core,
        "core": core_s,
        "compact": core_s.replace(" ", ""),
        "skel": " ".join(skeleton(t) for t in core),
        "is_domain": is_domain,
        "is_translit": is_translit,
    }


# ----------------------------------------------------------------------------- addresses
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "orissa",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "dl": "delhi", "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
}
STATE_PHRASES = sorted({v for v in list(US_STATES.values()) + list(IN_STATES.values())} | {"tamilnadu", "uttaranchal"},
                       key=len, reverse=True)
STATE_CODES = set(US_STATES) | set(IN_STATES)
_STATE_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in STATE_PHRASES) + r")\b")

STREET = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr", "lane": "ln",
    "court": "ct", "boulevard": "blvd", "place": "pl", "trail": "trl", "highway": "hwy", "parkway": "pkwy",
    "circle": "cir", "terrace": "ter", "square": "sq", "mount": "mt", "saint": "st", "north": "n",
    "south": "s", "east": "e", "west": "w", "nagar": "ngr", "marg": "rd", "sector": "sec", "county": "cty",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
}
ADDR_NOISE = {
    "unit", "apt", "apartment", "suite", "ste", "fl", "floor", "po", "box", "pmb", "no", "h", "hno",
    "null", "none", "nr", "near", "opp", "city", "of", "town", "township", "bldg", "the", "and", "c", "o",
    "flat", "shop", "plot", "door", "usa", "us", "india",
}


def normalize_address(raw: str) -> dict:
    """Returns dict with:
       words   : set of informative word tokens (states / decorations removed)
       nums    : set of number-ish tokens ('1622', '21/14' -> {'21','14'})
       text    : sorted words joined, for fuzzy comparison
       empty   : True if address missing
    """
    raw = raw or ""
    # drop tokens written in non-Latin script (in this data they are state names)
    parts = [p for p in re.split(r"(\s+|,)", raw) if p and _is_latin(p)]
    s = to_ascii("".join(parts))
    s = _STATE_RE.sub(" ", s)
    toks = [t for t in _NON_ALNUM.split(s) if t]
    words, nums = [], set()
    for t in toks:
        if t.isdigit():
            nums.add(t.lstrip("0") or "0")
            continue
        if _HAS_DIGIT.search(t):                # '1st', '5-cc' pieces, '27th'
            d = re.sub(r"\D", "", t)
            if d:
                nums.add(d.lstrip("0") or "0")
            if re.fullmatch(r"\d+(st|nd|rd|th)", t):
                continue
            t = re.sub(r"\d", "", t)
            if len(t) < 2:
                continue
        if t in STATE_CODES or t in ADDR_NOISE:
            continue
        t = STREET.get(t, t)
        if len(t) < 2 and t not in ("n", "s", "e", "w"):
            continue
        words.append(t)
    wset = set(words)
    return {
        "words": wset,
        "nums": nums,
        "text": " ".join(sorted(wset)),
        "num_text": " ".join(sorted(nums)),
        "empty": len(wset) == 0 and len(nums) == 0,
    }


if __name__ == "__main__":
    tests = ["Shri Kga Ínvestment Private Límited", "caangel.com", "जैन मैनेजमेंट प्राइवेट लिमिटेड",
             "Surgical Prairie - 4143439782", "H0rizon Trading Private Limited", "Palghar Palghar WIND Private Limited",
             "M/s Kga Investment Private", "Vivek +-[Co]"]
    for t in tests:
        print(t, "->", normalize_name(t))
    for a in ["Cno. ##804/5/2/S5 Khanapur Road, Khanapur Sangli Vita, Sangli, MH",
              "OH, Columbus, 5559 Orville Avenue", "1702 Pine Avenue, CITY OF MENOMONIE, WI",
              "591, BR, Patna, Behind Of Alankar Place Boring Road",
              "5-CC, IIND ADDITIONAL MANDI, SIRSA, हरियाणा"]:
        print(a, "->", normalize_address(a))
