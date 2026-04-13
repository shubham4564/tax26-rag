"""
Normalize IRS PDF filenames in a folder.

Target naming:
- Form <form_no>.pdf
- Instruction <form_no>.pdf
- Publication <pub_no>.pdf
- Schedule files are preserved as:
  Form <form_no> Schedule <sched>.pdf
  Instruction <form_no> Schedule <sched>.pdf

Examples:
- Form 941 - Employer's QUARTERLY Federal Tax Return.pdf
  -> Form 941.pdf
- Instruction 941 (Schedule B) - Instructions for Schedule B (Form 941)....pdf
  -> Instruction 941 Schedule B.pdf
- Publication 17 - Your Federal Income Tax (For Individuals).pdf
  -> Publication 17.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KIND_RE = re.compile(r"^(Form|Instruction|Publication)\s+(.+)$", re.IGNORECASE)
SCHEDULE_RE = re.compile(
    r"^(?P<base>.+?)\s*\((?P<schedule>Schedule\s+[^)]+)\)\s*$",
    re.IGNORECASE,
)
SCHEDULE_FIRST_RE = re.compile(
    r"^(?P<schedule>Schedule\s+[^()]+)\s*\(\s*Form\s+(?P<base>[^)]+)\s*\)\s*$",
    re.IGNORECASE,
)


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_suffix_description(rest: str) -> str:
    # Keep only the first segment before long title text.
    # Most files are like: "941 - Employer's Quarterly...".
    parts = rest.split(" - ", 1)
    return _clean_spaces(parts[0])


def _normalize_publication_id(core: str) -> str:
    # Keep numeric/suffix id only (e.g., 15-A, 55-B, 850).
    # Drop locale tags like "(en-vie)" from publication titles.
    m = re.match(r"^([0-9]+(?:-[A-Za-z0-9]+)?)", core)
    return m.group(1) if m else core


def _normalize_kind(kind: str) -> str:
    kind_lower = kind.lower()
    if kind_lower == "form":
        return "Form"
    if kind_lower == "instruction":
        return "Instruction"
    return "Publication"


def _normalize_schedule_text(schedule_text: str) -> str:
    cleaned = _clean_spaces(schedule_text)
    return re.sub(r"(?i)^schedule", "Schedule", cleaned, count=1)


def build_new_name(file_name: str) -> str | None:
    path = Path(file_name)
    if path.suffix.lower() != ".pdf":
        return None

    stem = path.stem

    # Handle schedule-first names like: "Schedule C (Form 1040)"
    sf = SCHEDULE_FIRST_RE.match(stem)
    if sf:
        base = _clean_spaces(sf.group("base"))
        sched = _normalize_schedule_text(sf.group("schedule"))
        return f"Form {base} {sched}.pdf"

    m = KIND_RE.match(stem)
    if not m:
        return None

    kind = _normalize_kind(m.group(1))
    rest = _strip_suffix_description(m.group(2))

    sched_match = SCHEDULE_RE.match(rest)
    if sched_match:
        base = _clean_spaces(sched_match.group("base"))
        sched = _normalize_schedule_text(sched_match.group("schedule"))
        return f"{kind} {base} {sched}.pdf"

    if kind == "Publication":
        pub_id = _normalize_publication_id(rest)
        return f"Publication {pub_id}.pdf"

    return f"{kind} {rest}.pdf"


def unique_destination(dest: Path, occupied_lc: set[str]) -> Path:
    stem = dest.stem
    suffix = dest.suffix
    candidate = dest
    idx = 2
    while candidate.name.lower() in occupied_lc:
        candidate = dest.with_name(f"{stem} ({idx}){suffix}")
        idx += 1
    occupied_lc.add(candidate.name.lower())
    return candidate


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text)


def rename_files(folder: Path, apply: bool) -> None:
    files = sorted(p for p in folder.iterdir() if p.is_file())
    raw_candidates: list[tuple[Path, str]] = []

    for src in files:
        new_name = build_new_name(src.name)
        if not new_name:
            continue
        if new_name == src.name:
            continue
        raw_candidates.append((src, new_name))

    if not raw_candidates:
        _safe_print("No files need renaming.")
        return

    sources_to_rename = {src for src, _ in raw_candidates}
    occupied_lc = {p.name.lower() for p in files if p not in sources_to_rename}
    planned: list[tuple[Path, Path]] = []

    for src, new_name in raw_candidates:
        dest = unique_destination(src.with_name(new_name), occupied_lc)
        if dest.name.lower() != src.name.lower():
            planned.append((src, dest))

    if not planned:
        _safe_print("No files need renaming.")
        return

    _safe_print(f"Planned renames: {len(planned)}")
    for src, dest in planned:
        _safe_print(f"- {src.name} -> {dest.name}")

    if not apply:
        _safe_print("\nDry run only. Use --apply to execute.")
        return

    # Two-phase rename prevents conflicts when destinations are currently
    # occupied by files that are also being renamed in the same batch.
    temp_paths: list[tuple[Path, Path]] = []
    temp_occupied_lc = {p.name.lower() for p in files}

    for idx, (src, _) in enumerate(planned, start=1):
        temp_name = f".__renaming__.{idx}{src.suffix}"
        temp = src.with_name(temp_name)
        while temp.name.lower() in temp_occupied_lc:
            idx += 1
            temp = src.with_name(f".__renaming__.{idx}{src.suffix}")
        temp_occupied_lc.add(temp.name.lower())
        src.rename(temp)
        temp_paths.append((temp, src))

    src_name_to_temp = {orig.name.lower(): temp for temp, orig in temp_paths}
    for src, dest in planned:
        temp_src = src_name_to_temp[src.name.lower()]
        temp_src.rename(dest)

    _safe_print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename IRS PDF files to short canonical names.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="irs_forms_english",
        help="Target folder containing PDF files (default: irs_forms_english)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename files")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    rename_files(folder, apply=args.apply)


if __name__ == "__main__":
    main()
