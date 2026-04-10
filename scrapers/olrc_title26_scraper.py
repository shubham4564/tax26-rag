"""
26 USC – Full Title Bulk Extractor  v8
=======================================

ROOT CAUSE FIX (v8):
  Previous versions assumed h3 tags separate the three regions.
  The actual OLRC HTML uses h4 tags throughout, distinguished by CSS class:

    <h3 class="section-head">   §1381. Organizations to which part applies
    <h4 class="subsection-head"> (a) In general
    <h4 class="subsection-head"> (b) Tax on certain farmers' cooperatives
    <h4 class="subsection-head"> (c) Cross reference
    <h4 class="note-head">      <strong>Editorial Notes</strong>        ← BOUNDARY
    <h4 class="note-head">      Amendments                              ← editorial sub
    <h4 class="note-head">      <strong>Statutory Notes…</strong>       ← BOUNDARY
    <h4 class="note-head">      Effective Date of 2017 Amendment        ← statutory sub
    <h4 class="note-head">      Effective Date                          ← statutory sub

  The two block-level boundaries are identified by:
    1. CSS class = "note-head"  AND
    2. Contains a <strong> child (the bold wrapper)  AND
    3. Text matches "editorial notes" or "statutory notes"

  Statute subsections use class="subsection-head", "paragraph-head", etc.
  Note sub-headings use class="note-head" WITHOUT a <strong> child.

  page_structure() v8:
    - Serialises the document content div to raw HTML
    - Finds the two boundary h4.note-head[strong] tags in source order
    - Splits the HTML string at those exact positions
    - Each parser receives only its own clean slice
"""

import re, json, time, asyncio, argparse, logging, os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from functools import partial

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TITLE           = 26
SEC_START       = 1
SEC_END         = 9834
DEFAULT_WORKERS = 20
DELAY_SECS      = 0.1
RETRY_LIMIT     = 3
RETRY_DELAY     = 4.0
BATCH_SIZE      = 200

BASE_URL = (
    "https://uscode.house.gov/view.xhtml"
    "?hl=false&edition=prelim"
    "&req=granuleid%3AUSC-prelim-title{t}-section{s}"
    "&num=0"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (usc-extractor/8.0; legal-research)",
    "Accept":     "text/html,application/xhtml+xml",
}

OUTPUT_DIR   = Path("../data/raw/federal/usc_title26_olrc")
SECTIONS_DIR = OUTPUT_DIR / "sections"

# ── Hierarchy constants ────────────────────────────────────────────────────────
LEVEL_ORDER = ["section","subsection","paragraph","subparagraph","clause","subclause"]
LABEL_PATS  = [
    ("subsection",   re.compile(r"^\([a-z]\)$")),
    ("paragraph",    re.compile(r"^\(\d+\)$")),
    ("subparagraph", re.compile(r"^\([A-Z]\)$")),
    ("clause",       re.compile(r"^\([ivxlcdm]+\)$")),
    ("subclause",    re.compile(r"^\([IVXLCDM]+\)$")),
]

# CSS classes used for statute content (never notes)
STATUTE_HEAD_CLASSES = {
    "section-head", "subsection-head", "paragraph-head",
    "subparagraph-head", "clause-head", "item-head",
}

_SRC_CREDIT_INLINE = re.compile(
    r"^\s*\((?:Added\s+|Amended\s+)?(?:Pub\.\s*L\.|Aug\.|Sept\.|Oct\.|"
    r"Nov\.|Dec\.|Jan\.|Feb\.|Mar\.|Apr\.|May|June|July)"
)
_FIELD   = re.compile(r"field-(?:start|end):[a-z\-]+")
_JS      = re.compile(r"(?://[^\n]*|/\*.*?\*/)", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WS      = re.compile(r"\s+")
_STATUTORY_PAT = re.compile(
    r"effective date|short title|savings provision|termination date|"
    r"sunset provision|applicability|construction", re.I)
_SRC_CREDIT_RE = re.compile(
    r"\((?:Added\s+|Amended\s+)?(?:Pub\.\s*L\.|Aug\.|Sept\.|Oct\.|Nov\.|Dec\.|"
    r"Jan\.|Feb\.|Mar\.|Apr\.|May|June|July)[^)]{10,}\)", re.DOTALL)

def clean(t: str) -> str:
    t = _COMMENT.sub(" ", t or "")
    t = _FIELD.sub(" ", t)
    t = _JS.sub(" ", t)
    return _WS.sub(" ", t).strip()

def classify_label(lbl: str) -> str:
    for lvl, pat in LABEL_PATS:
        if pat.match(lbl): return lvl
    return "unknown"

def extract_pub_law(text: str) -> str:
    m = re.search(
        r"(Pub\.\s*L\.\s*[\d–\-]+(?:,\s*(?:div\.|title|§)[^,]+)*"
        r"(?:,\s*\w+\.\s*\d+,\s*\d{4})?(?:,\s*\d+ Stat\. [\d\-]+)?)", text)
    return clean(m.group(1)) if m else ""

def extract_effective_date(text: str) -> str:
    m = re.search(r"((?:taxable years|effective|apply to|applies to)[^.]+\.)", text, re.I)
    return clean(m.group(1)) if m else ""

def is_not_found(soup: BeautifulSoup) -> bool:
    t = soup.get_text()
    return bool(re.search(r"Result\s+0\s+of\s+0", t)
                or re.search(r"no\s+results\s+found", t, re.I))

def detect_section_type(soup: BeautifulSoup) -> str:
    h = soup.find(class_="section-head")
    if h:
        ht = clean(h.get_text()).lower()
        if "repealed" in ht: return "repealed"
        if "reserved" in ht: return "reserved"
    body = soup.find(class_="statute-body") or soup.find(class_="statutory-body")
    if body:
        t = clean(body.get_text())
        if len(t) < 120 and re.search(r"see section \d+", t, re.I):
            return "cross_ref"
    return "normal"


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class TaxTableRow:
    income_range: str; tax_formula: str
    def to_dict(self): return {"income_range": self.income_range, "tax_formula": self.tax_formula}

@dataclass
class Node:
    level: str; label: str; heading: str; text: str
    table: Optional[list] = None
    children: list = field(default_factory=list)
    def to_dict(self):
        d = {"level":self.level,"label":self.label,"heading":self.heading,"text":self.text}
        if self.table:    d["table"]    = [r.to_dict() for r in self.table]
        if self.children: d["children"] = [c.to_dict() for c in self.children]
        return d

def _parse_table_tag(tag: Tag) -> list:
    rows = []
    for tr in tag.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            inc, tax = clean(cells[0].get_text()), clean(cells[1].get_text())
            if inc and tax: rows.append(TaxTableRow(inc, tax))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# THE FIX — page_structure() v8
#
# Key insight from reading the raw HTML:
#
#   All headings are h4 elements.
#   Statute subsections:  class="subsection-head"  (no <strong>)
#   Block boundaries:     class="note-head"  WITH a <strong> child
#   Note sub-headings:    class="note-head"  WITHOUT a <strong> child
#
#   The two block boundary h4s have text (stripped of <strong>) matching:
#     "Editorial Notes"
#     "Statutory Notes and Related Subsidiaries"
#
#   Strategy:
#     1. Find all h4.note-head tags that have a <strong> child
#     2. Identify which is editorial and which is statutory by text content
#     3. Use their byte position in str(soup) to split the HTML into 3 slices
# ══════════════════════════════════════════════════════════════════════════════

def _is_block_boundary(tag: Tag) -> bool:
    """True if this is a note-head h4 with a bold child (= top-level block marker)."""
    if tag.name != "h4": return False
    cls = " ".join(tag.get("class", []))
    if "note-head" not in cls: return False
    # Must contain <strong> or <b> to be a block header
    return bool(tag.find(["strong", "b"]))

def page_structure(soup: BeautifulSoup) -> dict:
    """
    Split the document into statute_body / editorial_notes / statutory_notes
    by locating the two bold note-head h4 boundary tags.

    Returns dict with three HTML fragment strings.
    """
    # Find all candidate boundary tags
    ed_tag   = None
    stat_tag = None

    for tag in soup.find_all("h4"):
        if not _is_block_boundary(tag):
            continue
        label = clean(tag.get_text()).lower()
        if "editorial" in label and "notes" in label and ed_tag is None:
            ed_tag = tag
        elif "statutory" in label and "notes" in label and stat_tag is None:
            stat_tag = tag
        if ed_tag and stat_tag:
            break

    # Serialise the full document
    full_html = str(soup)

    def find_pos(html: str, tag: Tag) -> int:
        """Find the byte position of tag in html using its outer HTML."""
        if tag is None:
            return -1
        outer = str(tag)
        pos = html.find(outer)
        if pos != -1:
            return pos
        # Fallback: search by a safe prefix of the tag text
        text_snippet = clean(tag.get_text())[:30]
        # Build a looser search pattern
        escaped = re.escape(text_snippet)
        m = re.search(r'<h4[^>]*class="[^"]*note-head[^"]*"[^>]*>.*?' + escaped,
                      html, re.DOTALL)
        return m.start() if m else -1

    ed_pos   = find_pos(full_html, ed_tag)   if ed_tag   else -1
    stat_pos = find_pos(full_html, stat_tag) if stat_tag else -1

    if ed_pos == -1 and stat_pos == -1:
        # No notes at all — everything is statute body
        return {"statute_body": full_html, "editorial_notes": "", "statutory_notes": ""}

    if ed_pos == -1:
        # Only statutory notes found
        return {
            "statute_body":    full_html[:stat_pos],
            "editorial_notes": "",
            "statutory_notes": full_html[stat_pos:],
        }

    if stat_pos == -1:
        # Only editorial notes found
        return {
            "statute_body":    full_html[:ed_pos],
            "editorial_notes": full_html[ed_pos:],
            "statutory_notes": "",
        }

    return {
        "statute_body":    full_html[:ed_pos],
        "editorial_notes": full_html[ed_pos:stat_pos],
        "statutory_notes": full_html[stat_pos:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# STATUTE PARSER
# Only processes h4 tags with statute-related CSS classes.
# ══════════════════════════════════════════════════════════════════════════════

def parse_statute(statute_html: str, sec_num: int) -> Node:
    frag = BeautifulSoup(statute_html, "html.parser")

    # Section heading: class="section-head"
    h3 = frag.find(class_="section-head")
    heading = clean(h3.get_text()) if h3 else f"§{sec_num}"
    root = Node("section", f"§{sec_num}", heading, "")

    # Only pick up subsection/paragraph heads — NOT note-heads
    statute_h4s = [
        t for t in frag.find_all("h4")
        if any(c in " ".join(t.get("class", [])) for c in STATUTE_HEAD_CLASSES)
    ]

    if not statute_h4s:
        # Flat text section
        body = frag.find(class_="statute-body") or frag
        root.text = clean(body.get_text())
        return root

    segments = []
    for h in statute_h4s:
        prose, tbls = [], []
        for sib in h.next_siblings:
            n = getattr(sib, "name", None)
            if n == "h4": break
            if n in ("p","div") and _SRC_CREDIT_INLINE.match(sib.get_text()): break
            if n == "table": tbls.append(sib)
            elif n:
                t = clean(sib.get_text())
                if t: prose.append(t)
            elif isinstance(sib, NavigableString):
                t = clean(str(sib))
                if t: prose.append(t)
        segments.append((clean(h.get_text()), " ".join(prose), tbls))

    stack = [root]
    for h_text, prose, tbls in segments:
        m = re.match(r"^(\([^)]+\))\s*(.*)", h_text)
        lbl = m.group(1) if m else ""
        hdr = clean(m.group(2)) if m else h_text
        lvl = classify_label(lbl) if m else "paragraph"
        tbl_rows = []
        for tbl in tbls: tbl_rows.extend(_parse_table_tag(tbl))
        node = Node(lvl, lbl, hdr, prose, tbl_rows or None)
        depth = LEVEL_ORDER.index(lvl) if lvl in LEVEL_ORDER else len(LEVEL_ORDER)
        while len(stack) > 1:
            pd = LEVEL_ORDER.index(stack[-1].level) if stack[-1].level in LEVEL_ORDER else 0
            if pd < depth: break
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    return root


# ══════════════════════════════════════════════════════════════════════════════
# NOTE SECTION COLLECTOR
# Shared by both editorial and statutory parsers.
# Collects (heading_text, body_text) from h4.note-head tags that do NOT
# have a <strong> child (those are block boundaries, already excluded by slicing).
# ══════════════════════════════════════════════════════════════════════════════

def _collect_note_sections(html_fragment: str) -> list[tuple[str, str]]:
    """
    Walk every h4.note-head (without <strong>) in the fragment.
    Returns [(heading_text, body_text), ...]
    """
    if not html_fragment.strip():
        return []
    frag = BeautifulSoup(html_fragment, "html.parser")
    results = []
    for h4 in frag.find_all("h4"):
        cls = " ".join(h4.get("class", []))
        if "note-head" not in cls:
            continue
        # Skip the block boundary tags themselves
        if _is_block_boundary(h4):
            continue
        heading = clean(h4.get_text())
        parts = []
        for sib in h4.next_siblings:
            if getattr(sib, "name", None) == "h4":
                break
            t = clean(sib.get_text()) if hasattr(sib, "get_text") else clean(str(sib))
            if t: parts.append(t)
        results.append((heading, " ".join(parts)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EDITORIAL NOTES PARSER
# Receives only the editorial HTML slice — cannot contain statutory content.
# ══════════════════════════════════════════════════════════════════════════════

def parse_editorial_notes(editorial_html: str) -> dict:
    result = {
        "title": "Editorial Notes",
        "references_in_text": [],
        "amendments":         [],
        "other_notes":        [],
    }
    for sh_text, full in _collect_note_sections(editorial_html):
        sh_lower = sh_text.lower()

        if sh_lower == "references in text":
            result["references_in_text"] = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", full) if s.strip()
            ]

        elif sh_lower == "amendments":
            blocks = re.split(r"(?=\b(?:19|20)\d{2}\b\s*[-–])", full.strip())
            amendments = []
            for blk in blocks:
                blk = blk.strip()
                if not blk: continue
                m = re.match(r"^((19|20)\d{2})\s*[-–](.*)", blk, re.DOTALL)
                if not m: continue
                year, rest = m.group(1), clean(m.group(3))
                entries = [
                    e.strip() for e in
                    re.split(r"(?=Subsec\.\s*\([a-z]\)|Section\s+\d)", rest)
                    if e.strip()
                ]
                amendments.append({"year": year, "entries": entries})
            result["amendments"] = amendments

        else:
            if full:
                result["other_notes"].append({
                    "heading": sh_text,
                    "text":    clean(full),
                })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STATUTORY NOTES PARSER
# Receives only the statutory HTML slice — cannot contain editorial content.
# ══════════════════════════════════════════════════════════════════════════════

def parse_statutory_notes(statutory_html: str) -> dict:
    result = {"title": "Statutory Notes and Related Subsidiaries", "notes": []}
    for sh_text, full in _collect_note_sections(statutory_html):
        if full:
            result["notes"].append({
                "heading":        sh_text,
                "pub_law":        extract_pub_law(full),
                "effective_date": extract_effective_date(full),
                "text":           clean(full),
            })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════

_SRC_CREDIT_RE2 = re.compile(
    r"\((?:Added\s+|Amended\s+)?(?:Pub\.\s*L\.|Aug\.|Sept\.|Oct\.|Nov\.|Dec\.|"
    r"Jan\.|Feb\.|Mar\.|Apr\.|May|June|July)[^)]{10,}\)", re.DOTALL)

def _extract_breadcrumb(soup: BeautifulSoup) -> dict:
    crumb = {}
    for a in soup.find_all("a", href=True):
        href = a.get("href","")
        if "granuleid" not in href or "title26" not in href.lower(): continue
        txt = clean(a.get_text())
        if   re.search(r"subtitle[A-Z]",   href, re.I) and "chapter"    not in href.lower(): crumb.setdefault("subtitle",   txt)
        elif re.search(r"chapter\d+",       href, re.I) and "subchapter" not in href.lower(): crumb.setdefault("chapter",    txt)
        elif re.search(r"subchapter[A-Z]",  href, re.I) and "-part"      not in href.lower(): crumb.setdefault("subchapter", txt)
        elif re.search(r"-part\d+",          href, re.I):                                     crumb.setdefault("part",       txt)
    return crumb

def parse_metadata(soup: BeautifulSoup, sec_num: int, statute_html: str) -> dict:
    crumb    = _extract_breadcrumb(soup)
    currency = soup.find(string=re.compile(r"Text contains those laws", re.I))
    sc_tag   = soup.find(class_="source-credit")
    sc_text  = clean(sc_tag.get_text()) if sc_tag else ""
    if not sc_text:
        frag = BeautifulSoup(statute_html, "html.parser")
        m    = _SRC_CREDIT_RE2.search(frag.get_text())
        if m: sc_text = clean(m.group(0)[:1000])
    return {
        "usc_location": {"title":"Title 26 – Internal Revenue Code",**crumb,"section":f"§{sec_num}"},
        "currency":      clean(str(currency)) if currency else "",
        "source_credit": sc_text,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_section_doc(html: str, sec_num: int) -> Optional[dict]:
    """Pure function: HTML string → dict. No I/O. Safe for multiprocessing."""
    soup = BeautifulSoup(html, "html.parser")
    if is_not_found(soup): return None
    pt = soup.get_text()
    if f"§{sec_num}" not in pt and f"§ {sec_num}" not in pt and "Result 1 of 1" not in pt:
        return None

    blocks    = page_structure(soup)
    sec_type  = detect_section_type(soup)
    statute   = parse_statute(blocks["statute_body"], sec_num)
    editorial = parse_editorial_notes(blocks["editorial_notes"])
    statutory = parse_statutory_notes(blocks["statutory_notes"])
    metadata  = parse_metadata(soup, sec_num, blocks["statute_body"])

    return {
        "section_number":  sec_num,
        "section_type":    sec_type,
        "metadata":        metadata,
        "statute":         statute.to_dict(),
        "editorial_notes": editorial,
        "statutory_notes": statutory,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTTP FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str) -> Optional[str]:
    s = requests.Session()
    for attempt in range(RETRY_LIMIT):
        try:
            r = s.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404: return None
            if r.status_code == 200: return r.text
        except Exception as e:
            log.warning(f"HTTP attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL MODES (unchanged from v6)
# ══════════════════════════════════════════════════════════════════════════════

async def _async_worker(session, sec_num: int, sem: asyncio.Semaphore) -> Optional[dict]:
    url = BASE_URL.format(t=TITLE, s=sec_num)
    async with sem:
        for attempt in range(RETRY_LIMIT):
            try:
                async with session.get(url, headers=HEADERS,
                                       timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 404: return None
                    html = await r.text()
                    await asyncio.sleep(DELAY_SECS)
                    return build_section_doc(html, sec_num)
            except Exception as e:
                log.warning(f"§{sec_num} async attempt {attempt+1}: {e}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return None

async def _run_async(todo: list, n_workers: int) -> list:
    sem  = asyncio.Semaphore(n_workers)
    conn = aiohttp.TCPConnector(limit=n_workers, ssl=False)
    found = []
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = {sec: asyncio.create_task(_async_worker(session, sec, sem)) for sec in todo}
        total, done = len(tasks), 0
        for sec, task in tasks.items():
            doc = await task; done += 1
            if doc: _save(doc, sec); found.append(doc)
            if done % BATCH_SIZE == 0: log.info(f"  async [{done}/{total}] last=§{sec}")
    return found

def _thread_worker(sec_num: int) -> Optional[dict]:
    html = _http_get(BASE_URL.format(t=TITLE, s=sec_num))
    if not html: return None
    time.sleep(DELAY_SECS)
    return build_section_doc(html, sec_num)

def _run_thread(todo: list, n_workers: int) -> list:
    found = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_thread_worker, sec): sec for sec in todo}
        done = 0
        for fut in as_completed(futures):
            sec = futures[fut]; done += 1
            try:
                doc = fut.result()
                if doc: _save(doc, sec); found.append(doc)
            except Exception as e:
                log.warning(f"§{sec} thread error: {e}")
            if done % BATCH_SIZE == 0: log.info(f"  thread [{done}/{len(todo)}] last=§{sec}")
    return found

def _process_worker(sec_num: int) -> Optional[dict]:
    html = _http_get(BASE_URL.format(t=TITLE, s=sec_num))
    if not html: return None
    time.sleep(DELAY_SECS)
    return build_section_doc(html, sec_num)

def _run_process(todo: list, n_workers: int) -> list:
    found = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_worker, sec): sec for sec in todo}
        done = 0
        for fut in as_completed(futures):
            sec = futures[fut]; done += 1
            try:
                doc = fut.result()
                if doc: _save(doc, sec); found.append(doc)
            except Exception as e:
                log.warning(f"§{sec} process error: {e}")
            if done % BATCH_SIZE == 0: log.info(f"  process [{done}/{len(todo)}] last=§{sec}")
    return found

async def _hybrid_worker(session, sec_num, sem, loop, parse_pool):
    url = BASE_URL.format(t=TITLE, s=sec_num)
    async with sem:
        for attempt in range(RETRY_LIMIT):
            try:
                async with session.get(url, headers=HEADERS,
                                       timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 404: return None
                    html = await r.text()
                    await asyncio.sleep(DELAY_SECS)
                    return await loop.run_in_executor(
                        parse_pool, partial(build_section_doc, html, sec_num))
            except Exception as e:
                log.warning(f"§{sec_num} hybrid attempt {attempt+1}: {e}")
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return None

async def _run_hybrid(todo: list, n_async: int, n_procs: int) -> list:
    loop = asyncio.get_event_loop()
    sem  = asyncio.Semaphore(n_async)
    conn = aiohttp.TCPConnector(limit=n_async, ssl=False)
    found = []
    with ProcessPoolExecutor(max_workers=n_procs) as parse_pool:
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = {sec: asyncio.create_task(
                _hybrid_worker(session, sec, sem, loop, parse_pool)) for sec in todo}
            total, done = len(tasks), 0
            for sec, task in tasks.items():
                doc = await task; done += 1
                if doc: _save(doc, sec); found.append(doc)
                if done % BATCH_SIZE == 0: log.info(f"  hybrid [{done}/{total}] last=§{sec}")
    return found

def _run_sync(todo: list) -> list:
    found = []
    for i, sec in enumerate(todo, 1):
        html = _http_get(BASE_URL.format(t=TITLE, s=sec))
        if html:
            doc = build_section_doc(html, sec)
            if doc: _save(doc, sec); found.append(doc)
        if i % BATCH_SIZE == 0: log.info(f"  sync [{i}/{len(todo)}] last=§{sec}")
        time.sleep(DELAY_SECS)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def run(mode: str, sec_range: range, n_workers: int, resume: bool) -> list:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    todo, preloaded = [], []
    for sec in sec_range:
        fp = SECTIONS_DIR / f"sec_{sec:04d}.json"
        if resume and fp.exists():
            with open(fp) as f: preloaded.append(json.load(f))
        else:
            todo.append(sec)
    log.info(f"Mode={mode}  workers={n_workers}  todo={len(todo)}  preloaded={len(preloaded)}")
    if not todo: return preloaded
    t0 = time.time()
    if mode == "async":
        if not ASYNC_AVAILABLE: mode = "thread"
        else: new = asyncio.run(_run_async(todo, n_workers))
    if mode == "thread":   new = _run_thread(todo, n_workers)
    elif mode == "process": new = _run_process(todo, n_workers)
    elif mode == "hybrid":
        if not ASYNC_AVAILABLE: new = _run_process(todo, n_workers)
        else: new = asyncio.run(_run_hybrid(todo, n_async=n_workers*2, n_procs=n_workers))
    elif mode == "sync":    new = _run_sync(todo)
    elapsed = time.time() - t0
    log.info(f"Fetched {len(new)} sections in {elapsed:.1f}s  ({len(new)/elapsed:.1f} §/s)")
    return preloaded + new


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def _save(doc: dict, sec: int):
    with open(SECTIONS_DIR/f"sec_{sec:04d}.json","w",encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

def _has_tables(node: dict) -> bool:
    return bool(node.get("table")) or any(_has_tables(c) for c in node.get("children",[]))

def save_index(docs: list):
    idx = sorted([{
        "section_number":           d["section_number"],
        "section_type":             d.get("section_type","normal"),
        "label":                    d.get("statute",{}).get("label",""),
        "heading":                  d.get("statute",{}).get("heading",""),
        **d.get("metadata",{}).get("usc_location",{}),
        "has_tables":               _has_tables(d.get("statute",{})),
        "amendment_years":          [a["year"] for a in d.get("editorial_notes",{}).get("amendments",[])],
        "other_editorial_headings": [n["heading"] for n in d.get("editorial_notes",{}).get("other_notes",[])],
        "statutory_note_count":     len(d.get("statutory_notes",{}).get("notes",[])),
    } for d in docs], key=lambda x: x["section_number"])
    with open(OUTPUT_DIR/"title26_index.json","w",encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    log.info(f"Index → {OUTPUT_DIR/'title26_index.json'}  ({len(idx)} sections)")

def save_combined(docs: list):
    with open(OUTPUT_DIR/"title26_combined.json","w",encoding="utf-8") as f:
        json.dump({"title":"Title 26 – Internal Revenue Code",
                   "sections":sorted(docs,key=lambda d:d["section_number"])},
                  f, indent=2, ensure_ascii=False)
    log.info(f"Combined → {OUTPUT_DIR/'title26_combined.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cpu = os.cpu_count() or 4
    ap  = argparse.ArgumentParser(description="26 USC bulk extractor v8")
    ap.add_argument("--mode",    choices=["async","thread","process","hybrid","sync"],
                                 default="hybrid")
    ap.add_argument("--workers", type=int, default=min(8, cpu))
    ap.add_argument("--start",   type=int, default=SEC_START)
    ap.add_argument("--end",     type=int, default=SEC_END)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--resume",  action="store_true")
    args = ap.parse_args()

    sec_range = range(args.start, args.end + 1)
    log.info(f"26 USC extractor v8 | §{args.start}–§{args.end} | mode={args.mode} | workers={args.workers}")

    docs = run(args.mode, sec_range, args.workers, args.resume)
    log.info(f"Total sections extracted: {len(docs)}")
    save_index(docs)
    if args.combine: save_combined(docs)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()