#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path


SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[4]

def _norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _norm_float(v: Any) -> Optional[float]:
    s = _norm_str(v)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_bool_or_none(v: Any) -> Optional[bool]:
    s = _norm_str(v)
    if s is None:
        return None
    sl = s.lower()
    if sl in {"1", "true", "t", "yes", "y"}:
        return True
    if sl in {"0", "false", "f", "no", "n"}:
        return False
    # Use None for ambiguous cases to avoid data noise
    return None


def pick_first_nonempty(row: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        if k in row:
            v = _norm_str(row.get(k))
            if v is not None:
                return v
    return None

def has_english_speaking_country(countries: Optional[str]) -> bool:
    """
    Example countries: "USA, UK" / "United States, Canada" / "UK" / "" / None
    """
    if not countries:
        return False

    # Split by comma, trim spaces, and normalize case
    parts = [c.strip().lower() for c in countries.split(",") if c.strip()]
    if not parts:
        return False

    # Include frequent variants and synonyms from the data
    english_aliases = {
        "usa",
        "uk",
        "great britain",
        "canada",
        "australia",
        "new zealand",
        "ireland",
    }

    return any(p in english_aliases for p in parts)

def pick_title(row: Dict[str, Any]) -> Optional[str]:
    title = _norm_str(row.get("title"))
    title_orig = _norm_str(row.get("title_orig"))
    countries = _norm_str(row.get("countries"))

    # If an English-speaking country is included, prefer title_orig, then title
    if has_english_speaking_country(countries):
        return title_orig or title
    # Otherwise prefer title, then title_orig
    return title or title_orig


def build_id2items(csv_path: str) -> Dict[str, Dict[str, Any]]:
    id2items: Dict[str, Dict[str, Any]] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing.")

        for row in reader:
            item_id = _norm_str(row.get("item_id"))
            if item_id is None:
                continue  # Skip when item_id is missing

            item: Dict[str, Any] = {
                "item_id": int(float(item_id)) if item_id.replace(".", "", 1).isdigit() else item_id,
                "content_type": _norm_str(row.get("content_type")),
                "title": pick_title(row),
                "release_year": _norm_float(row.get("release_year")),
                "genres": _norm_str(row.get("genres")),
                "countries": _norm_str(row.get("countries")),
                "for_kids": _norm_bool_or_none(row.get("for_kids")),
                "age_rating": _norm_float(row.get("age_rating")),
                "description": _norm_str(row.get("description")),
                "keywords": _norm_str(row.get("keywords")),
                # Apply fallbacks because actor/director column names may differ by CSV schema
                "actors": pick_first_nonempty(row, "actors_transliterated", "actors_translated", "actors"),
                "directors": pick_first_nonempty(row, "directors_translated", "directors_transliterated", "directors", "transliterated"),
            }

            id2items[str(item["item_id"])] = item

    return id2items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument(
        "--save_path",
        default=str(SUPPLEMENTARY_ROOT / "data/trace/mts-kion_processed/id2items.json"),
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    data = build_id2items(args.csv_path)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} items -> {args.save_path}")


if __name__ == "__main__":
    main()
