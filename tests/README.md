# Writing Style Analyzer Test Suite

Automated test suite for the writing-style-analyzer project.

**Privacy-First Design:** Uses ONLY synthetic test data. No personal profiles or texts are included in the repository.

## Quick Start

```bash
# Run all tests
uv run pytest tests/

# Run with coverage report
uv run pytest tests/ --cov=. --cov-report=term-missing

# Run specific test categories
uv run pytest tests/ -m profile      # Profile validation only
uv run pytest tests/ -m analysis     # Analysis function tests only
uv run pytest tests/ -m regression   # Regression tests only

# Run verbose mode
uv run pytest tests/ -v

# Run and stop at first failure
uv run pytest tests/ -x
```

## Test Suite Overview

### Test Categories

**Profile Validation Tests** (`test_profiles.py`)
- Validates profile JSON structure
- Checks required fields and data types
- Verifies metric consistency
- Compares default vs excellence profiles

**Analysis Function Tests** (`test_analysis.py`)
- Tests `analyze_text()` function
- Tests `compare_to_profile()` function
- Tests `generate_report()` function
- Edge case handling

**Regression Tests** (`test_regression.py`)
- Compares current results against baselines
- Ensures code changes don't break functionality
- Tests analysis determinism
- Validates consistency across different texts

### Test Fixtures

All test data is **synthetic and generic** (no personal data):

**Profiles:**
- `fixtures/sample_profile_default.json` - Generic German academic profile (5 categories)
- `fixtures/sample_profile_excellence.json` - Generic excellence profile (8 categories)

**Texts:**
- `fixtures/sample_text_short.txt` - ~100-word German academic text
- `fixtures/sample_text_long.txt` - ~200-word German academic text

**Baselines:**
- `baselines/baseline_short_text.json` - Expected analysis of short text
- `baselines/baseline_long_text.json` - Expected analysis of long text

## Test Statistics

```
Total Tests: 49 core + 6 optional LLM validation
├── Profile Tests: 15
├── Analysis Tests: 24
├── Regression Tests: 10
└── LLM Validation Tests: 6 (optional, requires API keys)

Coverage: ~95%
Runtime: <1 second (core tests)
         ~30-60 seconds (with LLM validation)
```

## Test Configuration

All test parameters are configured in `tests/config.yaml`:

### Tolerances
- **Word count:** ±15% from target
- **Sentence length:** ±3 words from expected
- **Transition density:** ±1.5 per 100 words
- **Passive voice:** ±10 percentage points
- **Lexical diversity:** ±0.05

### Regression Settings
- **Max deviation:** 5% from baseline
- **Metrics compared:** word_count, avg_sentence_length, transitions, density, passive voice, lexical diversity

## Regenerating Baselines

If you intentionally improve the analysis algorithm and want to update baselines:

```bash
# Regenerate baselines from fixtures
uv run python tests/generate_baselines.py

# Verify new baselines
uv run pytest tests/test_regression.py -v
```

**⚠️ Only update baselines for intentional improvements, not bugs!**

## LLM Validation Tests (Optional)

The test suite includes **optional end-to-end validation** that tests profiles with real LLM providers.

### What Are LLM Validation Tests?

These tests verify that writing style profiles actually work correctly when used with real LLMs:
- Generate text using Claude, GPT-4, or local models
- Analyze the generated text
- Verify it matches the profile's target metrics

### Setup

**1. Install optional dependencies:**
```bash
uv pip install -e ".[llm-validation]"
```

**2. Configure API keys:**
```bash
# For Anthropic/Claude
export ANTHROPIC_API_KEY="your-key-here"

# For OpenAI/GPT
export OPENAI_API_KEY="your-key-here"

# For Open WebUI/Ollama (local models)
export OPENWEBUI_BASE_URL="http://localhost:11434/v1"
export OPENWEBUI_API_KEY="dummy-key-for-local"  # optional for local
```

**3. Run tests:**
```bash
# Run ALL tests including LLM validation
uv run pytest tests/

# Run ONLY LLM validation tests
uv run pytest tests/test_llm_validation.py -v

# Run for specific provider
uv run pytest tests/test_llm_validation.py -k anthropic
uv run pytest tests/test_llm_validation.py -k openai
uv run pytest tests/test_llm_validation.py -k openwebui
```

### What Gets Tested?

**Core Validation:**
- ✅ Generated text matches profile's word count target (±25%)
- ✅ Sentence length matches profile (±5 words)
- ✅ Passive voice ratio matches profile (±15%)
- ✅ Transition categories from profile appear in text (≥70%)
- ✅ Transition density matches profile (±3.0 per 100 words)

**Advanced Validation:**
- ✅ Excellence profile uses advanced categories (conditional, clarifying, concessive)
- ✅ Consistency across multiple runs with same profile
- ✅ Cross-provider comparison (all providers produce valid output)

### Cost & Privacy

**Cost Warning:** LLM validation tests make real API calls which may incur costs:
- Each test run: ~100-300 tokens per test case
- Full suite: ~2000-5000 tokens total
- Estimated cost: $0.01-0.05 per full test run (varies by provider)

**Privacy:** LLM tests use ONLY synthetic fixtures from `tests/fixtures/`. No personal profiles or texts are sent to APIs.

### Auto-Skip Behavior

Tests automatically skip if no API keys are configured:
```
tests/test_llm_validation.py::test_profile_validation SKIPPED
Reason: No LLM API keys configured (optional tests)
```

This means:
- ✅ Tests run in CI without secrets (auto-skip)
- ✅ No setup required if you don't want LLM validation
- ✅ Can run in CI by setting API keys as GitHub secrets (recommended for profile validation)
- ✅ Enable locally for development testing

### Running in CI/CD (Optional)

To enable LLM validation in GitHub Actions:

1. **Add API keys as repository secrets:**
   - Settings → Secrets → Actions → New repository secret
   - Add `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`

2. **Tests will automatically run when keys are present:**
   ```yaml
   # .github/workflows/test.yml
   - name: Run tests with LLM validation
     env:
       ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
     run: uv run pytest tests/ -v
   ```

3. **Optional: Run on schedule to monitor profile quality:**
   ```yaml
   on:
     push:
       paths:
         - 'user-profiles/profiles/*.json'  # Run when profiles change
     schedule:
       - cron: '0 0 * * 0'  # Weekly on Sunday
   ```

### Configuration

All LLM test settings are in `tests/config.yaml`:
```yaml
llm_validation:
  providers:
    anthropic:
      model: "claude-sonnet-4-20250514"
    openai:
      model: "gpt-4-turbo"
    openwebui:
      model: "qwen2.5:3b"  # or any Ollama model

  validation:
    word_count_tolerance: 25.0
    sentence_length_tolerance: 5.0
    # ... more settings
```

## Privacy & Publishing

### What's Safe to Publish
✅ All files in `tests/` directory
✅ All files in `tests/fixtures/` (synthetic data)
✅ All files in `tests/baselines/` (synthetic baselines)
✅ Test suite code

### What's Protected (Gitignored)
❌ `user-profiles/` - All personal profiles and texts
❌ `texts/` - Personal text samples
❌ `tests/baselines/personal_*` - Personal baselines
❌ `tests/personal_test_results/` - Personal test results

### Running Tests on Personal Data (Local Only)

You can create personal tests locally (never committed):

```bash
# Example: Test your personal profile (not in repo)
uv run python german_academic_analyzer.py \
    user-profiles/test-prompts/results/test1-default-sonnet-4-5-500w.txt \
    user-profiles/profiles/academic-default.json \
    500

# This is for local validation only - results won't be committed
```

## Continuous Integration

The test suite is designed to run in CI without secrets:

**.github/workflows/test.yml** (example):
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest tests/ -v --cov=. --cov-report=term-missing
```

## Test Markers

Organize test runs with pytest markers:

```bash
# Run only fast tests (exclude slow)
uv run pytest tests/ -m "not slow"

# Run only profile validation
uv run pytest tests/ -m profile

# Run only regression tests
uv run pytest tests/ -m regression

# Run multiple categories
uv run pytest tests/ -m "profile or analysis"
```

Available markers:
- `profile` - Profile validation tests
- `analysis` - Analysis function tests
- `regression` - Regression tests
- `integration` - Integration tests
- `llm_validation` - LLM validation tests (optional, requires API keys)
- `slow` - Slow-running tests
- `personal` - Tests requiring personal data (not run in CI)

## Writing New Tests

### Adding a Test

```python
import pytest
from german_academic_analyzer import analyze_text

@pytest.mark.analysis
def test_my_new_feature(sample_text_short):
    """Test description."""
    result = analyze_text(sample_text_short)
    assert result['some_metric'] > 0
```

### Adding Fixtures

Edit `tests/conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """Fixture description."""
    return "test data"
```

### Adding Synthetic Data

1. Create text/profile in `tests/fixtures/`
2. Add fixture in `conftest.py`
3. Generate baseline if needed
4. Write tests using the fixture

## Troubleshooting

### All Tests Fail with Import Errors
```bash
# Ensure dependencies are installed
uv sync

# Check Python path
uv run python -c "import german_academic_analyzer"
```

### Regression Tests Fail After Code Change
- If intentional improvement: Regenerate baselines
- If unintentional: Fix the regression
- Check `tests/baselines/` for expected values

### Test Fixtures Not Found
```bash
# Ensure fixtures exist
ls tests/fixtures/

# Check paths in config.yaml
cat tests/config.yaml
```

## Best Practices

1. **Run tests before committing:** `uv run pytest tests/`
2. **Keep tests fast:** Use synthetic data only
3. **Never commit personal data:** Check `.gitignore`
4. **Update baselines carefully:** Only for improvements
5. **Write descriptive test names:** `test_<what>_<when>_<expected>`
6. **Use markers:** Organize tests by category
7. **Keep coverage high:** Aim for >90%

## See Also

- [pytest documentation](https://docs.pytest.org/)
- [Project README](../README.md)
- [Test Configuration](config.yaml)
- [Profile Validation Guide](../user-profiles/PROFILE_GUIDE.md)
