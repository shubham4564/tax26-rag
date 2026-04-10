"""
Collect explanatory tax documents:
- IRS publications
- IRS form instructions
- Curated third-party explanatory sources
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SourceDoc:
    name: str
    url: str
    source_type: str


class ExplanatoryDocsCollector:
    """Download and catalog explanatory sources for Phase 1."""

    IRS_BASE = "https://www.irs.gov"

    PUBLICATIONS = [
        "17",
        "334",
        "463",
        "501",
        "502",
        "504",
        "525",
        "527",
        "535",
        "550",
        "590-a",
        "590-b",
    ]

    FORM_PAGES = [
        "about-form-1040",
        "about-schedule-a-form-1040",
        "about-schedule-c-form-1040",
        "about-schedule-d-form-1040",
        "about-schedule-e-form-1040",
        "about-schedule-se-form-1040",
        "about-form-1040-es",
        "about-form-w-9",
    ]

    # Curated and reviewable list of external explanatory resources.
    THIRD_PARTY_SOURCES = [
        SourceDoc(
            name="Tax Foundation - Individual Income Taxes",
            url="https://taxfoundation.org/taxedu/glossary/individual-income-tax/",
            source_type="third_party",
        ),
        SourceDoc(
            name="Tax Foundation - Corporate Income Taxes",
            url="https://taxfoundation.org/taxedu/glossary/corporate-income-tax/",
            source_type="third_party",
        ),
        SourceDoc(
            name="Urban-Brookings Tax Policy Center - Briefing Book",
            url="https://www.taxpolicycenter.org/briefing-book/what-are-sources-revenue-federal-government",
            source_type="third_party",
        ),
    ]

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Research/Educational Tax IR System)",
            }
        )

        self.pub_dir = Path("data/raw/federal/irs_publications")
        self.form_dir = Path("data/raw/federal/irs_form_instructions")
        self.third_party_dir = Path("data/raw/explanatory/third_party")

        self.pub_dir.mkdir(parents=True, exist_ok=True)
        self.form_dir.mkdir(parents=True, exist_ok=True)
        self.third_party_dir.mkdir(parents=True, exist_ok=True)

    def run(self, start_year: int = 2020, end_year: int = 2026) -> Dict:
        """Execute end-to-end collection."""
        pub_records = self._download_publications(start_year, end_year)
        form_records = self._download_form_instructions()
        third_party_records = self._download_third_party_sources()

        summary = {
            "start_year": start_year,
            "end_year": end_year,
            "publications_downloaded": len(pub_records),
            "form_instructions_downloaded": len(form_records),
            "third_party_docs_downloaded": len(third_party_records),
            "created_at": datetime.now().isoformat(),
        }

        summary_path = Path("data/raw/explanatory/summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _download_publications(self, start_year: int, end_year: int) -> List[Dict]:
        records: List[Dict] = []
        for year in range(start_year, end_year + 1):
            for pub in self.PUBLICATIONS:
                url = self._publication_url(pub, year)
                filename = f"pub_{pub}_{year}.pdf".replace("-", "")
                target = self.pub_dir / filename
                if self._download_binary(url, target):
                    records.append(
                        {
                            "doc_type": "publication",
                            "year": year,
                            "id": pub,
                            "url": url,
                            "path": str(target),
                        }
                    )
        self._write_manifest(self.pub_dir / "manifest.json", records)
        return records

    def _download_form_instructions(self) -> List[Dict]:
        records: List[Dict] = []
        for slug in self.FORM_PAGES:
            page_url = f"{self.IRS_BASE}/forms-pubs/{slug}"
            links = self._discover_instruction_pdf_links(page_url)
            for link in links:
                filename = link.split("/")[-1]
                target = self.form_dir / filename
                if self._download_binary(link, target):
                    records.append(
                        {
                            "doc_type": "form_instruction",
                            "form_page": slug,
                            "url": link,
                            "path": str(target),
                        }
                    )
        self._write_manifest(self.form_dir / "manifest.json", records)
        return records

    def _download_third_party_sources(self) -> List[Dict]:
        records: List[Dict] = []
        for source in self.THIRD_PARTY_SOURCES:
            time.sleep(self.rate_limit)
            try:
                response = self.session.get(source.url, timeout=60)
                response.raise_for_status()
            except Exception as exc:
                logger.warning("Failed third-party fetch %s: %s", source.url, exc)
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else source.name
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join([p for p in paragraphs if p])

            slug = source.name.lower().replace(" ", "_").replace("-", "_")
            path = self.third_party_dir / f"{slug}.json"
            payload = {
                "name": source.name,
                "title": title,
                "url": source.url,
                "source_type": source.source_type,
                "text": text,
                "downloaded_at": datetime.now().isoformat(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            records.append({"name": source.name, "url": source.url, "path": str(path)})

        self._write_manifest(self.third_party_dir / "manifest.json", records)
        return records

    def _publication_url(self, pub: str, year: int) -> str:
        # IRS naming differs for prior-year archives.
        normalized = pub.lower()
        if year >= datetime.now().year - 1:
            return f"{self.IRS_BASE}/pub/irs-pdf/p{normalized}.pdf"
        return f"{self.IRS_BASE}/pub/irs-prior/p{normalized}--{year}.pdf"

    def _discover_instruction_pdf_links(self, page_url: str) -> List[str]:
        time.sleep(self.rate_limit)
        try:
            response = self.session.get(page_url, timeout=60)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to fetch form page %s: %s", page_url, exc)
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        links: List[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(" ", strip=True).lower()
            if "/pub/irs-pdf/" not in href:
                continue
            if not href.lower().endswith(".pdf"):
                continue

            filename = href.split("/")[-1].lower()
            # IRS instruction PDFs typically start with i* or include instruction-specific patterns.
            is_instruction_pdf = (
                filename.startswith("i")
                or "instruction" in text
                or "inst" in text
                or "gi" in filename
            )
            if not is_instruction_pdf:
                continue

            url = href if href.startswith("http") else f"{self.IRS_BASE}{href}"
            links.append(url)

        return sorted(set(links))

    def _download_binary(self, url: str, target: Path) -> bool:
        if target.exists() and target.stat().st_size > 0:
            return True

        time.sleep(self.rate_limit)
        try:
            response = self.session.get(url, stream=True, timeout=120)
            if response.status_code == 404:
                return False
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Download failed %s: %s", url, exc)
            return False

        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and target.suffix.lower() == ".pdf":
            # Keep strict to avoid saving HTML error pages as PDFs.
            return False

        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return target.exists() and target.stat().st_size > 0

    def _write_manifest(self, path: Path, records: List[Dict]) -> None:
        payload = {
            "count": len(records),
            "records": records,
            "generated_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main() -> None:
    collector = ExplanatoryDocsCollector(rate_limit=0.75)
    summary = collector.run(start_year=2020, end_year=2026)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
