#!/usr/bin/env python3
"""Scrape the NHM 2024 new-species article and save clean text output.

Usage:
  python scripts/scrape_nhm_new_species.py
  python scripts/scrape_nhm_new_species.py --output data/new_species_2024.txt --include-source-url
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_URL = (
    "https://www.nhm.ac.uk/discover/news/2024/december/"
    "dicaprios-snake-saurons-piranha-natural-history-museum-describe-190-new-species-2024.html"
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.nhm.ac.uk/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape and clean the NHM 2024 new-species article into a local text file."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Article URL to scrape.")
    parser.add_argument(
        "--output",
        default="data/new_species_2024.txt",
        help="Output text file path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--include-source-url",
        action="store_true",
        help="Prepend SOURCE_URL header to output file.",
    )
    return parser.parse_args()


def extract_clean_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    blocks = soup.find_all(
        "div", class_=lambda c: isinstance(c, str) and c.startswith("_text_")
    )

    clean: list[str] = []
    for block in blocks:
        text = re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip()
        if len(text) < 120:
            continue
        if "Receive email updates" in text or "We use cookies" in text:
            continue
        clean.append(text)

    if clean:
        return clean

    # Fallback if NHM markup changes: use main body text.
    fallback = soup.find("main") or soup
    fallback_text = re.sub(r"\s+", " ", fallback.get_text(" ", strip=True)).strip()
    return [fallback_text] if fallback_text else []


def main() -> int:
    args = parse_args()

    try:
        resp = requests.get(args.url, headers=DEFAULT_HEADERS, timeout=args.timeout)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        raise SystemExit(
            f"HTTP error while scraping source ({code}). "
            "If this is a 403, use the Drive-based download cell in the notebook."
        ) from exc
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    paragraphs = extract_clean_blocks(resp.text)
    if not paragraphs:
        raise SystemExit("No article text could be extracted from the page.")

    final_text = "\n\n".join(paragraphs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        if args.include_source_url:
            f.write(f"SOURCE_URL: {args.url}\n\n")
        f.write(final_text)

    print(f"Saved {output_path} ({len(final_text)} chars)")
    print(final_text[:500] + "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
