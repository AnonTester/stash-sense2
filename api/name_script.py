"""Detects whether a performer name is written in Latin ("western") script
or something else (Japanese, Cyrillic, Thai, etc.), and resolves which of
a performer's name/aliases to show as the display name when the "Prefer
Western Names" setting (settings.py's prefer_western_names) is on.

Performers crawled from a non-Latin-script source (javdatabase.com,
javstash.org today; other scripts down the line) often have their
canonical name stored in that script, with a romanized/English name
captured only as an alias (see config.py's aliases_json_path). This makes
it hard to visually compare a match against other sources. Display-only: never
rewrites an existing local Stash performer's own data, and never changes
which real-world person a match refers to -- see matching.py's own
comment on where this gets applied.
"""

import unicodedata
from typing import Optional

# Unicode category prefixes covering the "Letter" categories
# (Lu/Ll/Lt/Lm/Lo) this treats as script signal; digits/spaces/punctuation
# are ignored entirely rather than counted as "not western", so a name
# like "Seiko。" (a full-width Japanese period) isn't misjudged as
# non-western just for a trailing punctuation mark.
_LETTER_CATEGORIES = ("Lu", "Ll", "Lt", "Lm", "Lo")

# Unicode block ranges (inclusive) this treats as "western" (Latin-script)
# letters -- covers plain ASCII, accented Latin (Portuguese/French/German/
# etc. names), and the IPA/phonetic extensions sometimes used in romanized
# transliterations. Deliberately NOT "is ASCII" -- a name like "Beyoncé" or
# "Renée" must count as western.
_WESTERN_RANGES = (
    (0x0041, 0x005A),  # Basic Latin uppercase
    (0x0061, 0x007A),  # Basic Latin lowercase
    (0x00C0, 0x02AF),  # Latin-1 Supplement + Latin Extended-A/B + IPA Extensions
    (0x1E00, 0x1EFF),  # Latin Extended Additional
)


def _is_western_letter(char: str) -> bool:
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in _WESTERN_RANGES)


def is_western_script(text: Optional[str]) -> bool:
    """True iff `text`'s own letters are majority Latin-script.

    Counts only characters unicodedata classifies as letters (ignoring
    digits, spaces, and punctuation), so a name like "Seiko。" is judged
    purely on "Seiko" and a hyphenated western name like "Marie-Claire"
    isn't penalized for the hyphen. A string with no letters at all
    (empty, digits-only, punctuation-only) is NOT western -- there's
    nothing here worth preferring over an already-western name.
    """
    if not text:
        return False
    letters = [c for c in text if unicodedata.category(c) in _LETTER_CATEGORIES]
    if not letters:
        return False
    western_count = sum(1 for c in letters if _is_western_letter(c))
    return western_count > len(letters) / 2


def resolve_display_name(
    universal_id: str,
    canonical_name: str,
    aliases: Optional[dict[str, list[str]]],
    prefer_western: bool,
) -> tuple[str, Optional[str]]:
    """Returns (name_to_show, original_name_or_None).

    original_name is only ever set when a swap actually happened, so a
    caller can tell "show an aka line" apart from "nothing changed"
    without re-deriving the same script check itself.

    Swaps only when ALL of:
    - prefer_western is on (the user's own opt-in setting)
    - canonical_name itself is NOT already western (nothing to improve)
    - this performer has at least one alias that IS western (the first
      such alias, in aliases.json's own stored order -- that file's own
      exporter writes them in the crawled source's original order, not
      re-sorted, so "first" is a stable, deterministic pick)

    Any other case (setting off, already western, no western alias on
    file) returns (canonical_name, None) unchanged."""
    if not prefer_western or is_western_script(canonical_name):
        return canonical_name, None
    for alias in (aliases or {}).get(universal_id, []):
        if is_western_script(alias):
            return alias, canonical_name
    return canonical_name, None
