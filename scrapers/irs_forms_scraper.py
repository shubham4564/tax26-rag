"""
IRS Forms & Publications Scraper
---------------------------------
Scrapes https://www.irs.gov/forms-instructions-and-publications
across ALL pages, filters English-only forms, then downloads PDFs.

Requirements:
    pip install requests beautifulsoup4 tqdm
"""

import re
import time
import os
import csv
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL      = "https://www.irs.gov/forms-instructions-and-publications"
ITEMS_PER_PAGE = 200          # max allowed: 200  (reduces total page requests)
DOWNLOAD_DIR  = Path("irs_forms_english")
CSV_LOG       = Path("irs_forms_english.csv")
REQUEST_DELAY = 1.0           # seconds between page requests (be polite)
MAX_RETRIES   = 3
TIMEOUT       = 30

# Language suffixes that appear in the Product Number column for non-English docs.
# Pattern matches anything like "(SP)", "(ZH-S)", "(GUJ)", "(AR)", etc.
LANG_SUFFIX_RE = re.compile(r"\([A-Z]{2,5}(?:-[A-Z]{1,2})?\)\s*$", re.IGNORECASE)

# Title keywords that indicate a translated version
LANG_TITLE_RE  = re.compile(
    r"\((Arabic|Bengali|Farsi|French|Gujarati|Haitian Creole|Italian|Japanese|"
    r"Khmer|Korean|Punjabi|Polish|Portuguese|Russian|Spanish|Tagalog|Urdu|"
    r"Vietnamese|Chinese)[^)]*Version\)",
    re.IGNORECASE,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; IRS-Forms-Downloader/1.0; "
            "+https://github.com/your-repo)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def fetch_page(session: requests.Session, page: int) -> BeautifulSoup | None:
    """Fetch one listing page and return its BeautifulSoup tree."""
    params = {
        "items_per_page": ITEMS_PER_PAGE,
        "page": page,
        "order": "natural_sort_field",
        "sort": "asc",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as exc:
            log.warning("Page %d – attempt %d/%d failed: %s", page, attempt, MAX_RETRIES, exc)
            time.sleep(2 ** attempt)
    return None


def get_total_pages(soup: BeautifulSoup) -> int:
    """Read total item count from the page and compute total pages."""
    # "Showing 1 - 200 of 3113"
    match = re.search(r"of\s+([\d,]+)", soup.get_text())
    if match:
        total_items = int(match.group(1).replace(",", ""))
        pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        log.info("Total items: %d  →  %d pages to scrape", total_items, pages)
        return pages
    return 125  # safe fallback


def is_english(product_number: str, title: str) -> bool:
    """
    Return True only for English-language documents.

    Rules:
      • Product number must NOT end with a language suffix like (SP), (ZH-S) …
      • Title must NOT contain a "… (Spanish Version)" style phrase
    """
    if LANG_SUFFIX_RE.search(product_number.strip()):
        return False
    if LANG_TITLE_RE.search(title.strip()):
        return False
    return True


def parse_rows(soup: BeautifulSoup) -> list[dict]:
    """Extract form rows from the listing table on one page."""
    rows = []
    table = soup.find("table")
    if not table:
        return rows

    for tr in table.find_all("tr")[1:]:   # skip header row
        cols = tr.find_all("td")
        if len(cols) < 4:
            continue

        anchor    = cols[0].find("a")
        if not anchor:
            continue

        product   = anchor.get_text(strip=True)
        pdf_href  = anchor.get("href", "").strip()
        title     = cols[1].get_text(strip=True)
        rev_date  = cols[2].get_text(strip=True)
        post_date = cols[3].get_text(strip=True)

        # Resolve relative URLs
        if pdf_href and not pdf_href.startswith("http"):
            pdf_href = urljoin("https://www.irs.gov", pdf_href)

        # Only keep rows that link directly to a PDF
        if not pdf_href.lower().endswith(".pdf"):
            continue

        if is_english(product, title):
            rows.append({
                "product": product,
                "title":   title,
                "rev_date": rev_date,
                "post_date": post_date,
                "pdf_url": pdf_href,
            })
    return rows


def sanitize_filename(product: str, title: str) -> str:
    """Build a safe filename: '<product> - <title>.pdf'"""
    raw = f"{product} - {title}.pdf"
    # Replace characters that are illegal on Windows/macOS/Linux
    return re.sub(r'[\\/:*?"<>|]', "_", raw)


def download_pdf(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single PDF; return True on success."""
    if dest.exists():
        log.debug("Already exists, skipping: %s", dest.name)
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            return True
        except requests.RequestException as exc:
            log.warning("Download failed (attempt %d/%d): %s – %s",
                        attempt, MAX_RETRIES, dest.name, exc)
            time.sleep(2 ** attempt)
            if dest.exists():         # remove partial file
                dest.unlink()
    return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    session = make_session()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Scrape all listing pages ──────────────────────────────────────
    log.info("=== Phase 1: Collecting form metadata ===")
    first_soup  = fetch_page(session, 0)
    if not first_soup:
        log.error("Could not fetch page 0. Aborting.")
        return

    total_pages = get_total_pages(first_soup)
    all_forms: list[dict] = parse_rows(first_soup)

    for page_idx in tqdm(range(1, total_pages), desc="Scraping pages", unit="pg"):
        time.sleep(REQUEST_DELAY)
        soup = fetch_page(session, page_idx)
        if soup:
            all_forms.extend(parse_rows(soup))

    log.info("Found %d English-language forms/publications", len(all_forms))

    # ── Step 2: Write CSV index ────────────────────────────────────────────────
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["product","title","rev_date","post_date","pdf_url"])
        writer.writeheader()
        writer.writerows(all_forms)
    log.info("CSV index written → %s", CSV_LOG)

    # ── Step 3: Download PDFs ─────────────────────────────────────────────────
    log.info("=== Phase 2: Downloading %d PDFs ===", len(all_forms))
    ok = fail = 0
    for form in tqdm(all_forms, desc="Downloading", unit="pdf"):
        fname = sanitize_filename(form["product"], form["title"])
        dest  = DOWNLOAD_DIR / fname
        if download_pdf(session, form["pdf_url"], dest):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)   # light throttle between PDF downloads

    log.info("=== Done ===  ✓ %d downloaded  ✗ %d failed", ok, fail)


if __name__ == "__main__":
    main()