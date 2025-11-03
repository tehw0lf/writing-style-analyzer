#!/usr/bin/env python3
"""
German Academic Text Analyzer - Universal Library

This module provides general-purpose analysis functions for German academic writing.
It is NOT user-specific and can be used with any German academic text or profile.

Usage:
    from german_academic_analyzer import analyze_text, compare_to_profile

    # Analyze any German text
    metrics = analyze_text(your_text)

    # Compare generated text against any profile
    comparison = compare_to_profile(generated_text, profile_json, target_words)

Features:
- Transition word detection (8 categories, authoritative patterns)
- Passive voice estimation
- Sentence and paragraph metrics
- Lexical analysis
- Profile-aware comparison (works with any user's profile)

Created: 2025-11-02
Author: writing-style-analyzer project
License: MIT
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


# ============================================================================
# UNIVERSAL LINGUISTIC PATTERNS (Authoritative German Academic)
# ============================================================================

TRANSITIONS = {
    "additive": [
        "außerdem", "zusätzlich", "und", "sowie", "überdies", "ferner",
        "weiterhin", "darüber hinaus", "zudem", "ebenso", "gleichermaßen"
    ],
    "contrastive": [
        "jedoch", "aber", "allerdings", "dennoch", "dagegen", "hingegen",
        "andererseits", "im gegensatz", "demgegenüber", "vielmehr"
    ],
    "causal": [
        "daher", "somit", "da", "aufgrund", "folglich", "demnach",
        "infolgedessen", "deshalb", "weil", "deswegen", "aus diesem grund"
    ],
    "temporal": [
        "zunächst", "dann", "anschließend", "schließlich", "bevor",
        "nachdem", "während", "daraufhin", "danach", "zuvor", "zuerst"
    ],
    "conclusive": [
        "insgesamt", "zusammenfassend", "abschließend", "letztendlich",
        "im ergebnis", "somit", "insofern"
    ],
    "conditional": [
        "wenn", "falls", "sofern", "insofern", "vorausgesetzt",
        "unter der bedingung", "angenommen"
    ],
    "clarifying": [
        "das heißt", "mit anderen worten", "genauer gesagt", "d.h.",
        "anders ausgedrückt", "sprich"
    ],
    "concessive": [
        "obwohl", "trotzdem", "obgleich", "wenngleich", "trotz",
        "nichtsdestotrotz", "gleichwohl"
    ]
}

PASSIVE_INDICATORS = [
    r"\bwird\b", r"\bwerden\b", r"\bwurde\b", r"\bwurden\b",
    r"\bworden\b", r"\bgeworden\b", r"\bwäre\b", r"\bwären\b"
]


# ============================================================================
# CORE ANALYSIS FUNCTIONS (Universal - No User-Specific Logic)
# ============================================================================

def analyze_text(text: str) -> Dict:
    """
    Analyze German academic text for linguistic features.

    This function is completely universal and works with ANY German text.
    No user-specific assumptions or expectations.

    Args:
        text: German text to analyze (str)

    Returns:
        dict with metrics:
            - word_count: Total words
            - sentence_count: Total sentences
            - avg_sentence_length: Average words per sentence
            - paragraph_count: Total paragraphs
            - avg_paragraph_length: Average sentences per paragraph
            - semicolons: Semicolon count
            - total_transitions: Total transition words found
            - transition_density: Transitions per 100 words
            - transitions_by_category: Dict with counts per category
            - passive_indicators: Count of passive voice markers
            - passive_percentage_estimate: Estimated % of passive sentences
            - lexical_diversity: Unique words / total words (approximation)
    """
    # Clean text (remove markdown headers, normalize whitespace)
    text_clean = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    text_lower = text_clean.lower()

    # Word count
    words = text_clean.split()
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0

    # Sentence analysis
    sentences = re.split(r'[.!?]+', text_clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Paragraph analysis
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and not p.strip().startswith('#')]
    paragraph_count = len(paragraphs)
    avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0

    # Semicolons
    semicolon_count = text.count(';')

    # Transition word analysis
    transition_counts = {}
    total_transitions = 0

    for category, words_list in TRANSITIONS.items():
        count = 0
        found_words = []
        for word in words_list:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                count += matches
                found_words.append(f"{word} ({matches})")

        transition_counts[category] = {
            "count": count,
            "words": found_words
        }
        total_transitions += count

    # Passive voice estimation
    passive_count = 0
    for pattern in PASSIVE_INDICATORS:
        passive_count += len(re.findall(pattern, text_lower))

    passive_percentage = (passive_count / sentence_count * 100) if sentence_count > 0 else 0
    transition_density = (total_transitions / word_count * 100) if word_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "paragraph_count": paragraph_count,
        "avg_paragraph_length": round(avg_paragraph_length, 2),
        "semicolons": semicolon_count,
        "total_transitions": total_transitions,
        "transition_density": round(transition_density, 2),
        "transitions_by_category": transition_counts,
        "passive_indicators": passive_count,
        "passive_percentage_estimate": round(passive_percentage, 1),
        "lexical_diversity": round(lexical_diversity, 3),
    }


def load_profile(profile_path: str) -> Dict:
    """
    Load a user's writing style profile from JSON.

    Args:
        profile_path: Path to profile JSON file

    Returns:
        dict: Profile data
    """
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_expected_metrics(profile: Dict, target_word_count: int) -> Dict:
    """
    Calculate expected metrics for generated text based on ANY user's profile.

    This function is profile-aware but NOT user-specific. It works with
    any profile by extracting the user's actual metrics and scaling them.

    Args:
        profile: Profile JSON dict (any user's profile)
        target_word_count: Target word count for generated text

    Returns:
        dict: Expected metrics scaled to target word count
    """
    # Handle both flat and nested profile structures
    if 'basic' in profile.get('metrics', {}):
        # Nested structure (v2 profiles)
        baseline_words = profile['metrics']['basic']['total_words']
        sentence_length = profile['metrics']['basic']['avg_sentence_length']
        paragraph_length = profile['metrics']['basic']['avg_paragraph_length']
        lexical_diversity = profile['metrics']['basic']['lexical_diversity']
        passive_ratio = profile['metrics']['voice_and_style'].get('passive_ratio', 0)
    else:
        # Flat structure (generic profiles)
        baseline_words = profile['metrics'].get('total_words', 1000)
        sentence_length = profile['metrics'].get('avg_sentence_length', 20)
        paragraph_length = profile['metrics'].get('avg_paragraph_length', 3)
        lexical_diversity = profile['metrics'].get('lexical_diversity', 0.3)
        passive_ratio = profile['metrics'].get('passive_voice_ratio', 0.4)

    scale_factor = target_word_count / baseline_words

    # Extract user's actual metrics
    expected = {
        "sentence_length": sentence_length,
        "paragraph_length": paragraph_length,
        "lexical_diversity": lexical_diversity,
        "passive_voice_percentage": passive_ratio * 100,
    }

    # Scale transition counts if available
    if 'transitions' in profile:
        # Handle v2 structure
        if isinstance(profile['transitions'], dict) and 'by_category' in profile['metrics'].get('transitions', {}):
            for category, data in profile['metrics']['transitions']['by_category'].items():
                if isinstance(data, dict) and 'count' in data:
                    expected[f"{category}_transitions"] = round(data['count'] * scale_factor)
        # Handle flat structure
        else:
            for category, patterns in profile['transitions'].items():
                if isinstance(patterns, list):
                    expected[f"{category}_transitions"] = round(len(patterns) * scale_factor)

    return expected


def compare_to_profile(text: str, profile: Dict, target_word_count: int) -> Dict:
    """
    Compare generated text against ANY user's profile.

    This is the main validation function - works with any profile!

    Args:
        text: Generated text to analyze
        profile: User's profile JSON dict
        target_word_count: Target word count

    Returns:
        dict: Comprehensive comparison with metrics and quality checks
    """
    # Analyze the generated text
    actual = analyze_text(text)

    # Calculate expected metrics from profile
    expected = calculate_expected_metrics(profile, target_word_count)

    # Comparison results
    comparison = {
        "actual": actual,
        "expected": expected,
        "differences": {},
        "quality_checks": []
    }

    # Calculate differences
    if "sentence_length" in expected:
        diff = actual['avg_sentence_length'] - expected['sentence_length']
        comparison['differences']['sentence_length'] = {
            "actual": actual['avg_sentence_length'],
            "expected": expected['sentence_length'],
            "difference": round(diff, 2),
            "percentage": round((diff / expected['sentence_length'] * 100), 1) if expected['sentence_length'] > 0 else 0
        }

        # Quality check
        if abs(diff) <= 3:
            comparison['quality_checks'].append("✅ Sentence length within acceptable range")
        else:
            comparison['quality_checks'].append(f"⚠️ Sentence length off by {abs(diff):.1f} words")

    # Word count accuracy
    word_count_diff = actual['word_count'] - target_word_count
    word_count_percentage = abs(word_count_diff) / target_word_count * 100 if target_word_count > 0 else 0
    comparison['differences']['word_count'] = {
        "actual": actual['word_count'],
        "target": target_word_count,
        "difference": word_count_diff,
        "percentage": round(word_count_percentage, 1)
    }

    if word_count_percentage <= 15:
        comparison['quality_checks'].append(f"✅ Word count accurate ({actual['word_count']}/{target_word_count}, {word_count_percentage:.1f}% error)")
    else:
        comparison['quality_checks'].append(f"⚠️ Word count off by {word_count_percentage:.1f}%")

    # Transition density check
    if 'transitions' in profile or 'transitions' in profile.get('metrics', {}):
        # Calculate expected density from profile structure
        if 'basic' in profile.get('metrics', {}):
            # V2 nested structure
            expected_density = profile['metrics']['transitions'].get('transition_density', 4.23)
        else:
            # Flat structure - calculate from transitions list
            total_profile_transitions = sum(len(patterns) if isinstance(patterns, list) else 0
                                           for patterns in profile.get('transitions', {}).values())
            baseline_words = profile.get('metrics', {}).get('total_words', 1000)
            expected_density = (total_profile_transitions / baseline_words) * 100

        density_diff = actual['transition_density'] - expected_density

        comparison['differences']['transition_density'] = {
            "actual": actual['transition_density'],
            "expected": round(expected_density, 2),
            "difference": round(density_diff, 2)
        }

        if abs(density_diff) <= 1.5:
            comparison['quality_checks'].append("✅ Transition density matches profile")
        else:
            comparison['quality_checks'].append(f"⚠️ Transition density differs by {abs(density_diff):.1f}/100w")

    return comparison


def generate_report(comparison: Dict, test_name: str = "Analysis") -> str:
    """
    Generate a human-readable markdown report from comparison results.

    Args:
        comparison: Comparison dict from compare_to_profile()
        test_name: Name of the test/analysis

    Returns:
        str: Markdown formatted report
    """
    actual = comparison['actual']
    expected = comparison['expected']
    differences = comparison['differences']

    output = [f"## {test_name}\n"]

    # Basic metrics
    output.append("### Basic Metrics")
    output.append(f"- **Word count:** {actual['word_count']} "
                 f"(target: {differences['word_count']['target']}, "
                 f"{differences['word_count']['percentage']:.1f}% error)")
    output.append(f"- **Sentences:** {actual['sentence_count']}")
    output.append(f"- **Avg sentence length:** {actual['avg_sentence_length']} words "
                 f"(expected: {expected.get('sentence_length', 'N/A')})")
    output.append(f"- **Paragraphs:** {actual['paragraph_count']}")
    output.append(f"- **Lexical diversity:** {actual['lexical_diversity']}")
    output.append(f"- **Semicolons:** {actual['semicolons']}\n")

    # Transitions
    output.append("### Transition Analysis")
    output.append(f"- **Total:** {actual['total_transitions']} "
                 f"({actual['transition_density']}/100 words)\n")

    for category in TRANSITIONS.keys():
        count = actual['transitions_by_category'][category]['count']
        words = actual['transitions_by_category'][category]['words']
        output.append(f"- **{category.capitalize()}:** {count}")
        if words:
            output.append(f"  - Found: {', '.join(words[:5])}")  # Show first 5

    output.append("")

    # Passive voice
    output.append("### Passive Voice")
    output.append(f"- **Indicators found:** {actual['passive_indicators']}")
    output.append(f"- **Estimated percentage:** {actual['passive_percentage_estimate']}%")
    if 'passive_voice_percentage' in expected:
        output.append(f"- **Expected:** ~{expected['passive_voice_percentage']:.0f}%\n")
    else:
        output.append("")

    # Quality checks
    output.append("### Quality Assessment")
    output.extend(comparison['quality_checks'])
    output.append("")

    return "\n".join(output)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Example usage as a standalone script.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python german_academic_analyzer.py <text_file> [profile_json] [target_words]")
        print("\nExamples:")
        print("  # Just analyze a text:")
        print("  python german_academic_analyzer.py my_text.txt")
        print("\n  # Compare against profile:")
        print("  python german_academic_analyzer.py my_text.txt profile.json 1000")
        sys.exit(1)

    text_file = sys.argv[1]

    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    if len(sys.argv) >= 4:
        # Full comparison mode
        profile_file = sys.argv[2]
        target_words = int(sys.argv[3])

        profile = load_profile(profile_file)
        comparison = compare_to_profile(text, profile, target_words)
        report = generate_report(comparison, f"Analysis of {text_file}")
        print(report)
    else:
        # Analysis only mode
        metrics = analyze_text(text)
        print(f"## Analysis of {text_file}\n")
        print(f"**Word count:** {metrics['word_count']}")
        print(f"**Sentences:** {metrics['sentence_count']}")
        print(f"**Avg sentence length:** {metrics['avg_sentence_length']}")
        print(f"**Transition density:** {metrics['transition_density']}/100w")
        print(f"**Total transitions:** {metrics['total_transitions']}")
        print(f"**Passive voice estimate:** {metrics['passive_percentage_estimate']}%")
        print(f"**Lexical diversity:** {metrics['lexical_diversity']}")


if __name__ == "__main__":
    main()
