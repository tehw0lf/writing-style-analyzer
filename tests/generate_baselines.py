#!/usr/bin/env python3
"""
Generate baseline analysis files from synthetic test fixtures.

These baselines are used for regression testing to ensure that
code changes don't break existing functionality.

Usage:
    python tests/generate_baselines.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from german_academic_analyzer import analyze_text


def generate_baseline(text_path: Path, baseline_path: Path):
    """Generate baseline analysis from text file."""
    # Read text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Analyze
    analysis = analyze_text(text)

    # Save baseline
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated baseline: {baseline_path.name}")
    print(f"   Word count: {analysis['word_count']}")
    print(f"   Transitions: {analysis['total_transitions']}")
    print()


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"
    baselines_dir = Path(__file__).parent / "baselines"

    # Ensure baselines directory exists
    baselines_dir.mkdir(exist_ok=True)

    # Generate baselines for synthetic texts
    fixtures = [
        ("sample_text_short.txt", "baseline_short_text.json"),
        ("sample_text_long.txt", "baseline_long_text.json"),
    ]

    print("Generating baselines from synthetic test fixtures...\n")

    for text_file, baseline_file in fixtures:
        text_path = fixtures_dir / text_file
        baseline_path = baselines_dir / baseline_file

        if not text_path.exists():
            print(f"⚠️  Skipping {text_file}: file not found")
            continue

        generate_baseline(text_path, baseline_path)

    print("✅ All baselines generated successfully!")
    print(f"\nBaselines saved to: {baselines_dir}")
