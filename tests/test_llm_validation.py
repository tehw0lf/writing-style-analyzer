"""
LLM Validation Tests - Optional End-to-End Profile Testing

Tests that writing style profiles actually work with real LLM providers.

**OPTIONAL TESTS:** These tests are automatically skipped if no API keys are configured.
- Skip by default (no API keys in CI)
- Can run in CI if you set API keys as GitHub secrets
- Useful for validating profile changes before merging PRs

**Setup:**
Set one or more API keys as environment variables:
- ANTHROPIC_API_KEY - For Claude models
- OPENAI_API_KEY - For GPT models
- OPENWEBUI_BASE_URL + OPENWEBUI_API_KEY - For local/Ollama models

**Cost Warning:** These tests make real API calls which may incur costs.
Each test run uses ~100-300 tokens per test case (~$0.01-0.05 per full run).

**Privacy:** Uses only synthetic test profiles from fixtures/ directory.
No personal data is sent to any API.

**Profiles Tested:**
- tests/fixtures/sample_profile_default.json (German academic, 5 categories)
- tests/fixtures/sample_profile_excellence.json (German academic, 8 categories)
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.llm_clients import get_available_clients
from german_academic_analyzer import analyze_text


# Load test configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

LLM_CONFIG = CONFIG["llm_validation"]
VALIDATION = LLM_CONFIG["validation"]
TEST_SETTINGS = LLM_CONFIG["test_settings"]

# Get available LLM clients
AVAILABLE_CLIENTS = get_available_clients(LLM_CONFIG)

# Skip all tests in this module if no API keys are configured
pytestmark = pytest.mark.skipif(
    len(AVAILABLE_CLIENTS) == 0,
    reason="No LLM API keys configured (optional tests - set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENWEBUI_BASE_URL)",
)


@pytest.fixture
def test_profiles():
    """Load test profiles from fixtures"""
    fixtures_dir = Path(__file__).parent / "fixtures"
    profiles = {}
    for profile_file in TEST_SETTINGS["test_profiles"]:
        profile_path = fixtures_dir / profile_file
        with open(profile_path) as f:
            profile_name = profile_file.replace(".json", "")
            profiles[profile_name] = json.load(f)
    return profiles


def create_prompt(profile: dict, word_count: int) -> str:
    """
    Create a prompt that instructs the LLM to write text following the profile.

    Args:
        profile: Profile dictionary with metrics and transition patterns
        word_count: Target word count for generated text

    Returns:
        Formatted prompt string
    """
    language = profile["primary_language"]
    metrics = profile["metrics"]
    transitions = profile.get("transitions", {})

    # Extract key metrics
    avg_sentence_len = metrics["basic"]["avg_sentence_length"]
    passive_pct = metrics["voice_and_style"]["passive_voice_percentage"]

    # Extract transition categories
    transition_cats = list(transitions.keys()) if transitions else []

    if language == "de":
        prompt = f"""Schreibe einen akademischen Absatz mit genau {word_count} Wörtern zum Thema "Digitalisierung in der Bildung".

WICHTIGE STILANFORDERUNGEN:
- Durchschnittliche Satzlänge: ~{avg_sentence_len:.0f} Wörter
- Passiv-Anteil: ~{passive_pct:.0f}%
- Verwende Übergangswörter aus diesen Kategorien: {', '.join(transition_cats)}

Schreibe NUR den Text, keine Metakommentare."""
    else:
        prompt = f"""Write an academic paragraph of exactly {word_count} words on the topic of "Digital Transformation in Education".

IMPORTANT STYLE REQUIREMENTS:
- Average sentence length: ~{avg_sentence_len:.0f} words
- Passive voice ratio: ~{passive_pct:.0f}%
- Use transition words from these categories: {', '.join(transition_cats)}

Write ONLY the text, no meta-commentary."""

    return prompt


class TestLLMProfileValidation:
    """Test that profiles work correctly with real LLMs"""

    @pytest.mark.parametrize("provider_name", list(AVAILABLE_CLIENTS.keys()))
    @pytest.mark.parametrize("word_count", [150, 500])
    def test_generated_text_matches_profile_metrics(
        self, provider_name, word_count, test_profiles
    ):
        """
        Test that LLM-generated text matches the profile's target metrics.

        This is the core validation: we give the LLM a profile and check if
        the generated text actually follows that profile's characteristics.
        """
        client = AVAILABLE_CLIENTS[provider_name]
        profile = test_profiles["sample_profile_default"]

        # Generate text using the profile
        prompt = create_prompt(profile, word_count)
        generated_text = client.generate(
            prompt, timeout=TEST_SETTINGS["timeout"]
        )

        # Analyze the generated text
        analysis = analyze_text(generated_text)

        # Validate word count (±25%)
        actual_words = analysis["word_count"]
        word_count_diff = abs(actual_words - word_count) / word_count * 100
        assert (
            word_count_diff <= VALIDATION["word_count_tolerance"]
        ), f"{provider_name}: Word count {actual_words} vs target {word_count} (diff: {word_count_diff:.1f}%)"

        # Validate sentence length (±5 words)
        expected_sent_len = profile["metrics"]["basic"]["avg_sentence_length"]
        actual_sent_len = analysis["avg_sentence_length"]
        sent_len_diff = abs(actual_sent_len - expected_sent_len)
        assert (
            sent_len_diff <= VALIDATION["sentence_length_tolerance"]
        ), f"{provider_name}: Sentence length {actual_sent_len:.1f} vs expected {expected_sent_len:.1f} (diff: {sent_len_diff:.1f})"

        # Validate passive voice (±15 percentage points)
        expected_passive = profile["metrics"]["voice_and_style"][
            "passive_voice_percentage"
        ]
        actual_passive = analysis["passive_percentage_estimate"]
        passive_diff = abs(actual_passive - expected_passive)
        assert (
            passive_diff <= VALIDATION["passive_voice_tolerance"]
        ), f"{provider_name}: Passive voice {actual_passive:.1f}% vs expected {expected_passive:.1f}% (diff: {passive_diff:.1f}pp)"

    @pytest.mark.parametrize("provider_name", list(AVAILABLE_CLIENTS.keys()))
    def test_transition_categories_appear(self, provider_name, test_profiles):
        """
        Test that transition words from the profile actually appear in generated text.

        We check that at least 70% of the expected transition categories are present.
        """
        client = AVAILABLE_CLIENTS[provider_name]
        profile = test_profiles["sample_profile_default"]
        word_count = 500  # Use medium length for better category coverage

        # Generate text
        prompt = create_prompt(profile, word_count)
        generated_text = client.generate(
            prompt, timeout=TEST_SETTINGS["timeout"]
        )

        # Analyze transitions
        analysis = analyze_text(generated_text)
        found_categories = set(analysis["transitions"].keys())

        # Expected categories from profile
        expected_categories = set(profile["transitions"].keys())

        # Calculate coverage
        coverage = (
            len(found_categories & expected_categories) / len(expected_categories)
        )

        assert (
            coverage >= VALIDATION["min_category_coverage"]
        ), f"{provider_name}: Only {coverage*100:.0f}% of transition categories found (expected ≥{VALIDATION['min_category_coverage']*100:.0f}%). Missing: {expected_categories - found_categories}"

    @pytest.mark.parametrize("provider_name", list(AVAILABLE_CLIENTS.keys()))
    def test_excellence_profile_uses_advanced_categories(
        self, provider_name, test_profiles
    ):
        """
        Test that excellence profile's advanced categories (conditional, clarifying, concessive)
        actually appear in generated text.
        """
        client = AVAILABLE_CLIENTS[provider_name]
        profile = test_profiles["sample_profile_excellence"]
        word_count = 500

        # Generate text
        prompt = create_prompt(profile, word_count)
        generated_text = client.generate(
            prompt, timeout=TEST_SETTINGS["timeout"]
        )

        # Analyze transitions
        analysis = analyze_text(generated_text)
        found_categories = set(analysis["transitions"].keys())

        # Check for excellence-specific categories
        advanced_categories = {"conditional", "clarifying", "concessive"}
        found_advanced = found_categories & advanced_categories

        # At least 2 out of 3 advanced categories should appear
        assert (
            len(found_advanced) >= 2
        ), f"{provider_name}: Only {len(found_advanced)}/3 advanced categories found: {found_advanced}"

    @pytest.mark.parametrize("provider_name", list(AVAILABLE_CLIENTS.keys()))
    def test_consistency_across_runs(self, provider_name, test_profiles):
        """
        Test that the same profile produces consistent results across multiple runs.

        We don't expect identical results (LLMs have randomness), but metrics should
        be within a reasonable range.
        """
        client = AVAILABLE_CLIENTS[provider_name]
        profile = test_profiles["sample_profile_default"]
        word_count = 200  # Short text for faster testing
        num_runs = TEST_SETTINGS["consistency_runs"]

        # Generate multiple texts
        analyses = []
        prompt = create_prompt(profile, word_count)
        for _ in range(num_runs):
            generated_text = client.generate(
                prompt, timeout=TEST_SETTINGS["timeout"]
            )
            analysis = analyze_text(generated_text)
            analyses.append(analysis)

        # Check consistency of sentence length across runs
        sent_lengths = [a["avg_sentence_length"] for a in analyses]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths)
        max_deviation = max(abs(s - avg_sent_len) for s in sent_lengths)

        assert (
            max_deviation <= 5.0
        ), f"{provider_name}: Sentence length varies too much across runs: {sent_lengths} (max deviation: {max_deviation:.1f})"

        # Check consistency of passive voice across runs
        passives = [a["passive_percentage_estimate"] for a in analyses]
        avg_passive = sum(passives) / len(passives)
        max_passive_dev = max(abs(p - avg_passive) for p in passives)

        assert (
            max_passive_dev <= 20.0
        ), f"{provider_name}: Passive voice varies too much across runs: {passives} (max deviation: {max_passive_dev:.1f}pp)"

    @pytest.mark.parametrize("provider_name", list(AVAILABLE_CLIENTS.keys()))
    def test_transition_density_matches_profile(self, provider_name, test_profiles):
        """
        Test that transition word density matches the profile's expected density.
        """
        client = AVAILABLE_CLIENTS[provider_name]
        profile = test_profiles["sample_profile_default"]
        word_count = 500

        # Generate text
        prompt = create_prompt(profile, word_count)
        generated_text = client.generate(
            prompt, timeout=TEST_SETTINGS["timeout"]
        )

        # Analyze transitions
        analysis = analyze_text(generated_text)

        # Expected density from profile
        expected_density = profile["metrics"]["transitions"]["density_per_100_words"]
        actual_density = analysis["transition_density"]

        density_diff = abs(actual_density - expected_density)
        assert (
            density_diff <= VALIDATION["transition_density_tolerance"]
        ), f"{provider_name}: Transition density {actual_density:.1f} vs expected {expected_density:.1f} (diff: {density_diff:.1f})"


class TestLLMProviderComparison:
    """Compare how different LLM providers handle the same profile"""

    @pytest.mark.skipif(
        len(AVAILABLE_CLIENTS) < 2,
        reason="Need at least 2 LLM providers for comparison",
    )
    def test_all_providers_produce_valid_output(self, test_profiles):
        """
        Test that all available providers can generate valid text from the same profile.

        This ensures that our profiles are provider-agnostic.
        """
        profile = test_profiles["sample_profile_default"]
        word_count = 200
        prompt = create_prompt(profile, word_count)

        results = {}
        for provider_name, client in AVAILABLE_CLIENTS.items():
            generated_text = client.generate(
                prompt, timeout=TEST_SETTINGS["timeout"]
            )
            analysis = analyze_text(generated_text)
            results[provider_name] = analysis

            # Each provider should produce valid output
            assert (
                analysis["word_count"] > 0
            ), f"{provider_name} produced empty text"
            assert (
                analysis["sentence_count"] > 0
            ), f"{provider_name} produced no sentences"
            assert (
                len(analysis["transitions"]) > 0
            ), f"{provider_name} used no transitions"

        # All providers should be within reasonable range of each other
        word_counts = [r["word_count"] for r in results.values()]
        word_count_range = max(word_counts) - min(word_counts)
        assert (
            word_count_range <= word_count * 0.5
        ), f"Providers vary too much in word count: {dict(zip(results.keys(), word_counts))}"
