"""Masked exercises: the only signal that reading a sheet worked.

Reading a derivation feels like understanding and reliably is not. The sheet
is right there, every step justified, and nothing in the act of reading it
distinguishes "I could reproduce this" from "I followed along".

So hide one piece and ask for it back. Two kinds, matching the two halves of
the governing constraint:

  step    -- reproduce a move in a derivation. Tests the mechanics.
  symbol  -- say what a term in the equation is. Weaker than a step: it
             tests recall of what a symbol denotes, not what changes when you
             change it, because the sheets do not record the consequence
             per symbol. It is what a stepless definition can support.

No model is involved. Generating an exercise is hiding something already
written, and grading is reading the answer that was hidden. That makes the
whole layer free, deterministic and offline, which is what lets it be used
often enough to matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that carry no recall load: if the only thing unique about a masked
# step is that it says "therefore", there is nothing to reproduce.
STOPWORDS = frozenset("""
    a an the and or but if then than that this these those with without within
    for from into onto over under above below between across through during
    is are was were be been being has have had do does did can could will
    would shall should may might must of to in on at by as it its their there
    here we us our you your which what when where while both each any all some
    such same other another more most less least very much many few one two
    also thus hence therefore so because since given note that follows follow
    obtain obtained yields yield gives give write writing written let letting
    using use used apply applying applied where above below step form
""".split())

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _content_tokens(text: str) -> set[str]:
    """Meaning-bearing words, lowercased. Short tokens and filler dropped."""
    return {t.lower() for t in TOKEN.findall(text or "")
            if t.lower() not in STOPWORDS}


def _elsewhere(sheet: dict, skip_step: int | None = None,
               skip_symbol: int | None = None) -> set[str]:
    """Every content token the reader can still see once one piece is hidden."""
    parts: list[str] = [str(sheet.get("statement", ""))]
    for index, step in enumerate(sheet.get("steps", [])):
        if index != skip_step:
            parts.append(str(step.get("text", "")))
    for index, symbol in enumerate(sheet.get("symbols", [])):
        if index != skip_symbol:
            parts.append(str(symbol.get("meaning", "")))
    parts += [str(a) for a in sheet.get("assumptions", [])]
    parts += [str(f) for f in sheet.get("failure_modes", [])]
    return _content_tokens("\n".join(parts))


def unique_content(sheet: dict, index: int) -> set[str]:
    """Tokens this step contributes that no other visible part of the sheet has.

    The independent comparison. An exercise is only a test if what it hides
    cannot be read off the page around it -- a step whose every term is
    restated two lines later asks the reader to copy, not to recall.
    """
    steps = sheet.get("steps", [])
    if not 0 <= index < len(steps):
        return set()
    return _content_tokens(str(steps[index].get("text", ""))) - _elsewhere(
        sheet, skip_step=index)


def unique_symbol_content(sheet: dict, index: int) -> set[str]:
    """Same test for a symbol's meaning."""
    symbols = sheet.get("symbols", [])
    if not 0 <= index < len(symbols):
        return set()
    return _content_tokens(str(symbols[index].get("meaning", ""))) - _elsewhere(
        sheet, skip_symbol=index)


@dataclass
class Exercise:
    node_id: str
    kind: str          # "step" | "symbol"
    question: str
    context: list[str]
    answer: str
    hint: str = ""
    unique_terms: int = 0
    masked: list[int] = field(default_factory=list)


STEP_REF = re.compile(r"(?i)\bstep\s*(\d+)\b")


def forward_leaks(sheet: dict, index: int) -> list[int]:
    """Later steps that name this one and show what it produced.

    The vacuity guard is lexical: it asks whether this step's *words* appear
    elsewhere. This leak is structural. Masking step 3 of marginalization left
    step 4 saying "this expectation form ... Step 3 becomes p_X(x) = \int_y
    \delta(x-g(y)) p_Y(y) dy" -- the answer's form, handed over, with every
    word of it legitimately absent from step 3's own vocabulary.

    Demoting such steps is the wrong repair: they are back-referenced precisely
    because they carry the central move, and avoiding them leaves the reader
    drilling the peripheral ones. Mask the leak instead.
    """
    steps = sheet.get("steps", [])
    if not 0 <= index < len(steps):
        return []
    number = steps[index].get("n", index + 1)
    return [j for j in range(index + 1, len(steps))
            if any(int(m) == number
                   for m in STEP_REF.findall(str(steps[j].get("text", ""))))]


MIN_UNIQUE = 2


def maskable_steps(sheet: dict, min_unique: int = MIN_UNIQUE) -> list[int]:
    """Step indices worth hiding, hardest first.

    Hardest means most content the rest of the sheet does not give away.
    """
    scored = [(len(unique_content(sheet, i)), i)
              for i in range(len(sheet.get("steps", [])))]
    return [i for score, i in sorted(scored, reverse=True) if score >= min_unique]


def maskable_symbols(sheet: dict, min_unique: int = MIN_UNIQUE) -> list[int]:
    """Symbols worth asking about, terms in the equation first.

    Ranking by rarity alone is actively wrong here. A symbol is rare exactly
    when nothing else in the sheet refers to it, which is the signature of an
    aside rather than of importance -- it put `F_s`, a filtration, at the top
    of the markov_assumption drill, when measure-theoretic framing is out of
    scope for this project by construction.

    So a symbol that actually appears in the statement outranks one that does
    not, and rarity only breaks ties. That is the operational test the project
    is built around: for every term *in the equations*, can you say why it is
    there?
    """
    statement = str(sheet.get("statement", ""))
    scored = []
    for i, entry in enumerate(sheet.get("symbols", [])):
        name = str(entry.get("symbol", "")).strip()
        first = name.split(",")[0].strip()
        in_statement = bool(first) and first in statement
        scored.append((in_statement, len(unique_symbol_content(sheet, i)), i))
    return [i for in_statement, score, i in sorted(scored, reverse=True)
            if score >= min_unique]


def step_exercise(sheet: dict, index: int) -> Exercise:
    steps = sheet.get("steps", [])
    hidden = sorted({index, *forward_leaks(sheet, index)})
    context = [f"[{s.get('n', i + 1)}] {s.get('text', '')}"
               if i not in hidden else f"[{s.get('n', i + 1)}] ???"
               for i, s in enumerate(steps)]
    numbers = [steps[i].get("n", i + 1) for i in hidden]
    invokes: list[str] = []
    for i in hidden:
        invokes += [str(x) for x in steps[i].get("invokes", [])]
    if len(numbers) == 1:
        question = f"Reproduce step {numbers[0]}."
    else:
        listed = ", ".join(str(n) for n in numbers[:-1])
        question = (f"Reproduce steps {listed} and {numbers[-1]}"
                    f" (the later one restates the first, so it is hidden too).")
    return Exercise(
        node_id=str(sheet.get("node_id", "")),
        kind="step",
        question=question,
        context=context,
        answer="\n\n".join(f"[{steps[i].get('n', i + 1)}] "
                            f"{steps[i].get('text', '')}" for i in hidden),
        # What the step invokes is the honest hint: it names the move without
        # performing it.
        hint=("uses: " + ", ".join(dict.fromkeys(invokes))) if invokes else "",
        unique_terms=len(unique_content(sheet, index)),
        masked=hidden,
    )


def symbol_exercise(sheet: dict, index: int) -> Exercise:
    symbol = sheet.get("symbols", [])[index]
    name = str(symbol.get("symbol", ""))
    dims = str(symbol.get("dimensions", "") or "")
    return Exercise(
        node_id=str(sheet.get("node_id", "")),
        kind="symbol",
        # Deliberately not "and what changes if you change it" -- that is the
        # operational test, but a symbol's `meaning` does not carry the
        # consequence, so asking it would promise more than the answer holds.
        question=f"What is {name}?",
        context=[str(sheet.get("statement", ""))],
        answer=str(symbol.get("meaning", "")),
        hint=f"dimensions: {dims}" if dims else "",
        unique_terms=len(unique_symbol_content(sheet, index)),
    )


def exercises_for(sheet: dict, limit: int = 3) -> list[Exercise]:
    """Hardest exercises this sheet can support, steps before symbols.

    A stepless sheet is not excluded: a definition still has terms, and asking
    what each one is remains the operational test.
    """
    out = [step_exercise(sheet, i) for i in maskable_steps(sheet)[:limit]]
    if len(out) < limit:
        out += [symbol_exercise(sheet, i)
                for i in maskable_symbols(sheet)[:limit - len(out)]]
    return out


def available_count(sheet: dict) -> int:
    """Every exercise this sheet can support, ignoring any display limit."""
    return len(maskable_steps(sheet)) + len(maskable_symbols(sheet))


def earned_state(passed: int, presented: int, available: int) -> str | None:
    """What a drill result is worth: "demonstrated", "read", or nothing.

    DEMONSTRATED requires answering everything the sheet can ask, not
    everything it happened to ask. Without the `available` comparison a
    display limit silently buys the strongest claim in the profile -- running
    with one exercise and getting it right marked a node demonstrated.
    """
    if passed <= 0:
        return None
    if presented >= available and passed >= available:
        return "demonstrated"
    return "read"
