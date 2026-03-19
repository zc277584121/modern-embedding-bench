"""Prepare real needle-in-a-haystack evaluation data.

Fetches Wikipedia articles as haystacks and creates fictional but realistic
needle facts with corresponding queries.

Usage:
    uv run python scripts/prepare_needle_data.py [--preview]
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "needle_haystack"

# Wikipedia articles to fetch (diverse topics)
WIKI_ARTICLES = [
    "Photosynthesis",
    "History_of_the_Internet",
    "Plate_tectonics",
    "Human_digestive_system",
    "Quantum_computing",
    "Renaissance",
    "Climate_change",
    "DNA",
    "International_Space_Station",
    "Supply_and_demand",
]

# Fictional needle facts — realistic but clearly fictional entities
NEEDLE_FACTS = [
    {
        "needle": "The Meridian Corporation reported quarterly revenue of $847.3 million in Q3 2025, a 23% increase over the previous year.",
        "query": "What was Meridian Corporation's quarterly revenue?",
        "category": "finance",
    },
    {
        "needle": "Clinical trials for Veritaxin (compound VRX-4419) showed a 78.2% remission rate in patients with treatment-resistant lymphoma over a 36-month follow-up period.",
        "query": "What was the remission rate for Veritaxin in clinical trials?",
        "category": "medicine",
    },
    {
        "needle": "The Arclight Observatory detected gravitational wave signal GW-20250714 originating from a neutron star merger approximately 340 million light-years away in the constellation Pisces.",
        "query": "Where did the Arclight Observatory detect gravitational wave signal GW-20250714?",
        "category": "science",
    },
    {
        "needle": "Server cluster NX-East-07 is scheduled for emergency maintenance on April 12, 2026 from 02:00 to 06:00 UTC due to firmware vulnerability CVE-2025-31847.",
        "query": "When is server cluster NX-East-07 scheduled for maintenance?",
        "category": "technology",
    },
    {
        "needle": "The recommended dosage for Paxorin (zaleptimine hydrochloride) is 175mg taken orally twice daily with food, reduced to 100mg for patients with hepatic impairment.",
        "query": "What is the recommended dosage for Paxorin?",
        "category": "medicine",
    },
    {
        "needle": "The Crossfield Bridge project was completed at a final cost of $2.1 billion, running 14 months behind the original schedule due to unexpected geological conditions in the river bed.",
        "query": "What was the final cost of the Crossfield Bridge project?",
        "category": "engineering",
    },
    {
        "needle": "The Kensington Protocol requires all Level 3 classified documents to be encrypted using the AES-512 standard and stored in air-gapped facilities with biometric access controls updated every 90 days.",
        "query": "What encryption standard does the Kensington Protocol require for Level 3 documents?",
        "category": "security",
    },
    {
        "needle": "Professor Elena Vasquez of the Instituto Nacional de Astrofisica published a paper demonstrating that exoplanet Kepler-7742b has an atmospheric composition of 43% nitrogen, 31% carbon dioxide, and 18% water vapor.",
        "query": "What is the atmospheric composition of exoplanet Kepler-7742b?",
        "category": "science",
    },
    {
        "needle": "The Thornfield Academy entrance examination requires a minimum score of 2340 out of 3000 points, with at least 780 points in the quantitative reasoning section and 720 points in verbal analysis.",
        "query": "What is the minimum score required for the Thornfield Academy entrance exam?",
        "category": "education",
    },
    {
        "needle": "The autonomous vehicle fleet operated by TransitLink logged 4.7 million miles in 2025 with a safety record of 0.12 incidents per 100,000 miles, making it the lowest incident rate among commercial autonomous vehicle operators.",
        "query": "What was TransitLink's autonomous vehicle safety incident rate?",
        "category": "technology",
    },
]

HAYSTACK_LENGTHS = [1000, 4000, 8000, 16000, 32000]
NEEDLE_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def fetch_wikipedia_article(title: str) -> str:
    """Fetch plain text of a Wikipedia article via MediaWiki API (extracts)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",  # Plain text, no HTML
        "exlimit": "1",
        "format": "json",
    }
    headers = {
        "User-Agent": "mm-embedding-bench/0.1 (research; https://github.com/example)",
    }
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    data = resp.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")

    # Light cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_haystack(articles: list[str], target_length: int) -> str:
    """Build a haystack of approximately target_length characters from articles."""
    combined = " ".join(articles)
    if len(combined) >= target_length:
        # Truncate at word boundary
        truncated = combined[:target_length]
        last_space = truncated.rfind(" ")
        if last_space > target_length * 0.9:
            return truncated[:last_space]
        return truncated
    # Repeat if needed
    repeats = target_length // len(combined) + 1
    combined = " ".join([combined] * repeats)
    truncated = combined[:target_length]
    last_space = truncated.rfind(" ")
    if last_space > target_length * 0.9:
        return truncated[:last_space]
    return truncated


def insert_needle(haystack: str, needle: str, position: float) -> str:
    """Insert needle at the specified position (0.0=start, 1.0=end)."""
    if position <= 0.0:
        return needle + " " + haystack
    if position >= 1.0:
        return haystack + " " + needle

    insert_idx = int(len(haystack) * position)
    # Find nearest word boundary
    space_idx = haystack.rfind(" ", 0, insert_idx)
    if space_idx > insert_idx * 0.8:
        insert_idx = space_idx + 1

    return haystack[:insert_idx] + " " + needle + " " + haystack[insert_idx:]


def save_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    import sys
    preview_only = "--preview" in sys.argv

    print("=== Needle-in-Haystack Data Preparation ===\n")

    # Step 1: Fetch Wikipedia articles
    print("Fetching Wikipedia articles...")
    articles = []
    for title in WIKI_ARTICLES:
        print(f"  Fetching: {title}...", end=" ")
        try:
            text = fetch_wikipedia_article(title)
            articles.append({"title": title, "text": text})
            print(f"OK ({len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.5)

    if not articles:
        print("No articles fetched. Exiting.")
        return

    # Combine article texts for haystack building
    article_texts = [a["text"] for a in articles]
    total_chars = sum(len(t) for t in article_texts)
    print(f"\nTotal article text: {total_chars:,} chars from {len(articles)} articles")

    # Step 2: Build test cases
    print("\nBuilding test cases...")
    haystacks = []
    for length in HAYSTACK_LENGTHS:
        haystack_text = build_haystack(article_texts, length)
        haystacks.append({
            "length": length,
            "text": haystack_text,
            "actual_length": len(haystack_text),
        })
        print(f"  Haystack {length:>6d} chars -> actual {len(haystack_text):>6d} chars")

    needles = []
    for needle_info in NEEDLE_FACTS:
        needles.append({
            "needle": needle_info["needle"],
            "query": needle_info["query"],
            "category": needle_info["category"],
        })

    # Build full test cases (for verification)
    test_cases = []
    for hs in haystacks:
        for pos in NEEDLE_POSITIONS:
            for needle_info in NEEDLE_FACTS:
                doc = insert_needle(hs["text"], needle_info["needle"], pos)
                test_cases.append({
                    "length": hs["length"],
                    "position": pos,
                    "needle": needle_info["needle"],
                    "query": needle_info["query"],
                    "category": needle_info["category"],
                    "doc_length": len(doc),
                })

    # Step 3: Print preview
    print(f"\n{'='*70}")
    print(f"  Data Summary")
    print(f"{'='*70}")
    print(f"  Wikipedia articles:    {len(articles)}")
    print(f"  Haystack lengths:      {HAYSTACK_LENGTHS}")
    print(f"  Needle positions:      {NEEDLE_POSITIONS}")
    print(f"  Needle facts:          {len(NEEDLE_FACTS)}")
    print(f"  Total test cases:      {len(test_cases)} ({len(HAYSTACK_LENGTHS)} lengths × {len(NEEDLE_POSITIONS)} positions × {len(NEEDLE_FACTS)} needles)")

    print(f"\n{'='*70}")
    print(f"  Sample Needles")
    print(f"{'='*70}")
    for i, n in enumerate(NEEDLE_FACTS[:5]):
        print(f"\n  [{i+1}] Category: {n['category']}")
        print(f"      Needle: {n['needle'][:80]}...")
        print(f"      Query:  {n['query']}")

    print(f"\n{'='*70}")
    print(f"  Sample Haystack Snippets (start/middle)")
    print(f"{'='*70}")
    for hs in haystacks[:3]:
        print(f"\n  Length {hs['length']} chars:")
        print(f"    Start: {hs['text'][:100]}...")
        print(f"    Middle: ...{hs['text'][len(hs['text'])//2 - 50:len(hs['text'])//2 + 50]}...")

    if preview_only:
        print("\n[Preview mode] No files written. Remove --preview to save.")
        return

    # Save data
    save_jsonl(DATA_DIR / "haystacks.jsonl", haystacks)
    save_jsonl(DATA_DIR / "needles.jsonl", needles)

    # Save articles separately for reference
    articles_clean = [{"title": a["title"], "length": len(a["text"])} for a in articles]
    save_jsonl(DATA_DIR / "articles_meta.jsonl", articles_clean)

    print(f"\nData saved to {DATA_DIR}/")
    print(f"  haystacks.jsonl:     {len(haystacks)} haystacks")
    print(f"  needles.jsonl:       {len(needles)} needle facts")
    print(f"  articles_meta.jsonl: {len(articles)} article metadata")


if __name__ == "__main__":
    main()
