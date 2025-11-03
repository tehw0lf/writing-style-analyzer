# Analysis Scripts Overview

This document explains the purpose of each Python script in the project.

**Last Updated:** 2025-11-03

---

## Core Scripts (Public)

### `analyze.py` ⭐ MAIN TOOL

**Purpose:** Profile generation - the core tool of this project

**What it does:**
- Analyzes your writing samples (text files, PDFs, DOCX, ODT, LaTeX)
- Extracts linguistic patterns using local LLMs
- Generates JSON/MD profiles of your writing style

**When to use:** When you want to create a NEW writing style profile from your texts

**Usage:**
```bash
uv run analyze.py --input texts/my-writing --output profiles/my-style.json
```

**See:** `README.md` and `QUICKSTART.md` for full documentation

---

## Universal Analysis Library

### `german_academic_analyzer.py` 🌍 GENERALIZABLE

**Purpose:** Universal German academic text analysis library

**What it does:**
- Analyzes ANY German academic text for linguistic features
- Detects 222+ transition words across 8 categories
- Estimates passive voice, sentence length, lexical diversity
- Can compare generated text against ANY user's profile (not just yours!)

**When to use:**
- You want to analyze German academic text WITHOUT creating a profile
- You want to validate AI-generated text against your (or someone else's) profile
- You're building tools that need German text analysis

**Usage:**
```bash
# Just analyze a text (no profile needed)
uv run python german_academic_analyzer.py my_text.txt

# Compare against a profile
uv run python german_academic_analyzer.py generated_text.txt profiles/my-profile.json 1000

# Use as a library in your code
from german_academic_analyzer import analyze_text, compare_to_profile
metrics = analyze_text(your_text)
```

**Key Feature:** Works with ANY German academic profile! ✅

---

## Quick Reference

| Script | Location | Purpose | Input |
|--------|----------|---------|-------|
| **analyze.py** | Root | Create profiles from writing samples | Your text files (.txt, .md, .pdf, .docx, .odt, .tex) |
| **german_academic_analyzer.py** | Root | Analyze ANY German academic text | Any text + any profile (optional) |

---

## Using the Scripts

### Creating Your Own Profile

```bash
# 1. Collect your writing samples
mkdir -p texts/my-writing
# Copy your files (.txt, .md, .pdf, .docx, .odt) into texts/my-writing/

# 2. Run the analyzer
uv run analyze.py --input texts/my-writing --output profiles/my-style.json --profile-type my-style

# 3. Use the generated profile
# - profiles/my-style.json (for analysis)
# - profiles/my-style.md (for AI assistants)
```

### Analyzing Generated Text

```python
# Use german_academic_analyzer.py as a library!
from german_academic_analyzer import compare_to_profile, generate_report, load_profile

# Load YOUR profile
my_profile = load_profile("profiles/my-style.json")

# Analyze YOUR generated text
with open("my_generated_text.txt") as f:
    text = f.read()

# Compare against YOUR metrics
comparison = compare_to_profile(text, my_profile, target_words=1000)
report = generate_report(comparison, "My Test")
print(report)
```

---

## Technical Details

### Universal Components (in `german_academic_analyzer.py`)

**Transition Detection:**
- 8 categories: additive, contrastive, causal, temporal, conclusive, conditional, clarifying, concessive
- 222+ patterns from authoritative sources (Duden, academic style guides)
- Works for ANY German academic text

**Passive Voice Detection:**
- Standard German passive constructions: wird, werden, wurde, wurden, worden, etc.
- Universal for German language

**Metrics Calculation:**
- Word count, sentence count, paragraph count
- Lexical diversity (unique/total words)
- Transition density (per 100 words)
- All universal - no user assumptions

---

## Related Documentation

- **README.md** - Main project documentation
- **QUICKSTART.md** - Quick start guide for beginners
- **CLAUDE.md** - Developer guide and architecture overview

---

**Last Updated:** 2025-11-03
**Purpose:** Document the core analysis scripts available to all users
