"""Download the official Riftbound Core Rules PDF and parse it into data/processed/rules.jsonl.

    python -m scripts.fetch_rules                    # download from the pinned official URL
    python -m scripts.fetch_rules --pdf path/to.pdf  # use a PDF you downloaded yourself

The PDF and parsed rules are Riot's text, so they are not committed (see .gitignore); this script
recreates them. The SHA-256 check tells you whether you have the same version the eval was built on.
"""
import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

from app.config import RAW_PDF, RULES_JSONL, RULES_SHA256, RULES_URL, RULES_VERSION
from app.corpus import parse_pdf, save_rules


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course project fetch script)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=Path, help="use a local PDF instead of downloading")
    a = p.parse_args()

    if a.pdf:
        RAW_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.pdf, RAW_PDF)
    elif not RAW_PDF.exists():
        print(f"Downloading {RULES_URL}")
        try:
            download(RULES_URL, RAW_PDF)
        except Exception as e:
            raise SystemExit(f"Download failed ({e}).\nDownload the Core Rules PDF from "
                             f"https://playriftbound.com/en-us/rules-hub/ and run: python -m scripts.fetch_rules --pdf <file>")

    digest = hashlib.sha256(RAW_PDF.read_bytes()).hexdigest()
    if digest == RULES_SHA256:
        print(f"PDF matches the pinned version ({RULES_VERSION}).")
    else:
        print(f"WARNING: this PDF is not the pinned {RULES_VERSION} version. The rules may have changed, "
              "so some eval answers may no longer be correct.")

    rules = parse_pdf(RAW_PDF)
    save_rules(rules)
    print(f"Parsed {len(rules)} rules -> {RULES_JSONL}")


if __name__ == "__main__":
    main()
