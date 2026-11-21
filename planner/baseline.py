"""Rule-based baseline planner: fixed command templates -> fixed skill sequences (no LLM).

Templates (case does not matter, a trailing '.', '!' or '?' is ignored, 'please' in front too):

    move     <verb> <books> [back] <prep> <target>
             verb  put | place | move | bring | take | return | carry
             prep  in | into | to | on | onto
             books one book, or two joined by "and" ("Deep Learning and Robotics")
    find     where is <book>          |  find <book>
    check    check / verify [that | if | whether] <book> is in | on <target>

    <book>   the exact title or id, optionally with "the" in front and / or "book" after
             ("the Physics book", "Machine Learning", "ml")
    <target> a compartment     "S2-A3", "shelf S2-A3", "compartment S2-A3"
             a table           "Table B", "table b"
             a shelf           "shelf S2", "S2": its first compartment (A1..A4, then B1..B4)
                               with a free slot
             any shelf         "the shelf", "a shelf", "the shelves", "shelf": the compartment
                               with a free slot whose parking pose is nearest the book's

A move becomes, per book:
    FIND_BOOK(book) -> NAVIGATE(where it is) -> GRASP(book) -> NAVIGATE(target)
    -> PLACE(book, target) -> VERIFY(book, target)
(only VERIFY if the book is already there). Anything else is rejected with the reason; e.g.
"Take the book closest to Table D and place it in the empty shelf nearest to the window" has no
template. That brittleness is the point of the baseline (README 3.1).
"""
import re
from dataclasses import dataclass

from planner import plan as P
from planner import world as W

VERBS = r"put|place|move|bring|take|return|carry"
PREPS = r"in|into|to|on|onto"


@dataclass
class ParseResult:
    plan: list               # [Action], None if the command was not understood
    template: str            # which template matched, "" if none
    reason: str              # why it failed / what was resolved

    @property
    def ok(self):
        return self.plan is not None


def parse(command, world):
    """Plan for a command, using the world state to resolve where books are and which
    compartment a vague target means."""
    c = command.strip().rstrip(".!?").strip()
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"^please ", "", c, flags=re.IGNORECASE)

    m = re.fullmatch(rf"(?:{VERBS}) (?P<books>.+?)(?: back)? (?:{PREPS}) (?P<target>.+)", c, re.IGNORECASE)
    if m:
        books = split_books(m.group("books"), world)
        if books is None:
            return ParseResult(None, "", f"no book called '{m.group('books')}' "
                                         "(the baseline needs the exact title)")
        return move(books, m.group("target"), world)

    m = re.fullmatch(r"(?:where is|find) (?P<book>.+)", c, re.IGNORECASE)
    if m:
        b = match_book(m.group("book"), world)
        if b is None:
            return ParseResult(None, "", f"no book called '{m.group('book')}'")
        return ParseResult([P.Action("FIND_BOOK", [world["books"][b]["title"]])], "find", "")

    m = re.fullmatch(r"(?:check|verify)(?: that| if| whether)? (?P<book>.+?) is (?:in|on) (?P<target>.+)",
                     c, re.IGNORECASE)
    if m:
        b = match_book(m.group("book"), world)
        t = P.canonical_target(clean_target(m.group("target")))
        if b is None:
            return ParseResult(None, "", f"no book called '{m.group('book')}'")
        if t is None:
            return ParseResult(None, "", f"'{m.group('target')}' is not a table or a compartment")
        return ParseResult([P.Action("VERIFY", [world["books"][b]["title"], t])], "check", "")

    return ParseResult(None, "", "no template matches this command")


def clean_target(t):
    t = re.sub(r"^(?:the |a )", "", t.strip(), flags=re.IGNORECASE)
    return re.sub(r"^compartment ", "", t, flags=re.IGNORECASE)


def match_book(text, world):
    """Book id for '[the] <title or id> [book]', or None."""
    t = re.sub(r"^the ", "", text.strip(), flags=re.IGNORECASE)
    for cand in (t, re.sub(r" book$", "", t, flags=re.IGNORECASE)):
        b = W.book_id(world, cand)
        if b is not None:
            return b
    return None


def split_books(text, world):
    """[ids] for one book or 'A and B' (a title may itself contain 'and', e.g. 'Signals and
    Systems', so the whole text is tried first), or None."""
    b = match_book(text, world)
    if b is not None:
        return [b]
    parts = re.split(r" and ", text, flags=re.IGNORECASE)
    for k in range(1, len(parts)):
        a, c = match_book(" and ".join(parts[:k]), world), match_book(" and ".join(parts[k:]), world)
        if a is not None and c is not None and a != c:
            return [a, c]
    return None


def resolve_target(text, book, w):
    """Concrete target ('Table B' / 'S2-A3') for the target text, given the symbolic world w at
    that point of the plan; (target, None) or (None, reason)."""
    t = clean_target(text)
    exact = P.canonical_target(t)
    if exact is not None:
        return exact, None
    m = re.fullmatch(r"(?:shelf )?(S\d)", t, re.IGNORECASE)
    if m:
        shelf = m.group(1).upper()
        if shelf not in W.META["shelves"]:
            return None, f"there is no shelf {shelf}"
        for comp, c in w["compartments"].items():          # fixed order: A1..A4, B1..B4
            if c["shelf"] == shelf and c["free_slots"] > 0:
                return comp, None
        return None, f"shelf {shelf} is full"
    if re.fullmatch(r"shel(?:f|ves)", t, re.IGNORECASE):
        src = w["books"][book]["at"]
        free = [comp for comp, c in w["compartments"].items() if c["free_slots"] > 0]
        if not free:
            return None, "every shelf compartment is full"
        if src.startswith(("Table", "S")):
            return min(free, key=lambda comp: (W.distance(w, src, comp), comp)), None
        return free[0], None
    return None, f"'{text}' is not a table, a shelf or a compartment"


def move(books, target_text, world):
    plan, notes = [], []
    w = world
    for b in books:
        title = w["books"][b]["title"]
        target, why = resolve_target(target_text, b, w)
        if target is None:
            return ParseResult(None, "move", why)
        if target != clean_target(target_text):
            notes.append(f"'{target_text}' -> {target} for {title}")
        s = w["books"][b]
        if s["at"] == target:
            steps = [P.Action("VERIFY", [title, target])]
        else:
            steps = [P.Action("FIND_BOOK", [title])]
            if s["kind"] != "held":
                steps += [P.Action("NAVIGATE", [s["at"]]), P.Action("GRASP", [title])]
            steps += [P.Action("NAVIGATE", [target]), P.Action("PLACE", [title, target]),
                      P.Action("VERIFY", [title, target])]
        plan += steps
        w = P.validate(plan, world).world        # the state after this book, for the next one
    return ParseResult(plan, "move", "; ".join(notes))
