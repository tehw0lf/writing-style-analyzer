"""
Analysis function unit tests.

Tests the core analysis functions from german_academic_analyzer.py
using synthetic test data.
"""

import pytest
from german_academic_analyzer import (
    analyze_text,
    compare_to_profile,
    calculate_expected_metrics,
    generate_report
)


@pytest.mark.analysis
class TestAnalyzeText:
    """Test the analyze_text() function."""

    def test_analyze_short_text(self, sample_text_short):
        """Test analysis of short German academic text."""
        result = analyze_text(sample_text_short)

        # Check all required fields are present
        required_fields = [
            'word_count', 'sentence_count', 'avg_sentence_length',
            'paragraph_count', 'avg_paragraph_length', 'semicolons',
            'total_transitions', 'transition_density',
            'transitions_by_category', 'passive_indicators',
            'passive_percentage_estimate', 'lexical_diversity'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_word_count_is_reasonable(self, sample_text_short):
        """Test that word count is in expected range for short text."""
        result = analyze_text(sample_text_short)

        # Short text should be ~90-150 words
        assert 50 <= result['word_count'] <= 200, (
            f"Word count unexpected: {result['word_count']}"
        )

    def test_sentence_count_is_positive(self, sample_text_short):
        """Test that sentence count is positive."""
        result = analyze_text(sample_text_short)
        assert result['sentence_count'] > 0

    def test_avg_sentence_length_is_reasonable(self, sample_text_short):
        """Test that average sentence length is in academic range."""
        result = analyze_text(sample_text_short)

        # German academic sentences are typically 15-30 words
        assert 10 <= result['avg_sentence_length'] <= 40, (
            f"Avg sentence length unusual: {result['avg_sentence_length']}"
        )

    def test_lexical_diversity_in_valid_range(self, sample_text_short):
        """Test that lexical diversity is between 0 and 1."""
        result = analyze_text(sample_text_short)
        assert 0 < result['lexical_diversity'] <= 1

    def test_transition_categories_present(self, sample_text_short):
        """Test that all transition categories are analyzed."""
        result = analyze_text(sample_text_short)
        transitions = result['transitions_by_category']

        expected_categories = [
            'additive', 'contrastive', 'causal', 'temporal',
            'conclusive', 'conditional', 'clarifying', 'concessive'
        ]

        for category in expected_categories:
            assert category in transitions, f"Missing category: {category}"
            assert 'count' in transitions[category]
            assert 'words' in transitions[category]

    def test_finds_additive_transitions(self, sample_text_short):
        """Test that additive transitions are detected in sample text."""
        result = analyze_text(sample_text_short)
        transitions = result['transitions_by_category']

        # Sample text contains "außerdem" and "darüber hinaus"
        assert transitions['additive']['count'] > 0, (
            "Should detect additive transitions in sample text"
        )

    def test_finds_contrastive_transitions(self, sample_text_short):
        """Test that contrastive transitions are detected in sample text."""
        result = analyze_text(sample_text_short)
        transitions = result['transitions_by_category']

        # Sample text contains "allerdings" and "dennoch"
        assert transitions['contrastive']['count'] > 0, (
            "Should detect contrastive transitions in sample text"
        )

    def test_finds_causal_transitions(self, sample_text_short):
        """Test that causal transitions are detected in sample text."""
        result = analyze_text(sample_text_short)
        transitions = result['transitions_by_category']

        # Sample text contains "somit" and "folglich"
        assert transitions['causal']['count'] > 0, (
            "Should detect causal transitions in sample text"
        )

    def test_finds_passive_voice(self, sample_text_short):
        """Test that passive voice indicators are detected."""
        result = analyze_text(sample_text_short)

        # Sample text contains passive constructions
        assert result['passive_indicators'] > 0, (
            "Should detect passive voice in sample text"
        )

    def test_long_text_has_more_words(self, sample_text_short, sample_text_long):
        """Test that long text has significantly more words than short text."""
        short_result = analyze_text(sample_text_short)
        long_result = analyze_text(sample_text_long)

        assert long_result['word_count'] > short_result['word_count'] * 1.5, (
            "Long text should have significantly more words"
        )

    def test_transition_density_calculation(self, sample_text_short):
        """Test that transition density is correctly calculated."""
        result = analyze_text(sample_text_short)

        expected_density = (result['total_transitions'] / result['word_count']) * 100
        actual_density = result['transition_density']

        diff = abs(expected_density - actual_density)
        assert diff < 0.1, (
            f"Transition density calculation incorrect: "
            f"expected {expected_density:.2f}, got {actual_density}"
        )


@pytest.mark.analysis
class TestCalculateExpectedMetrics:
    """Test the calculate_expected_metrics() function."""

    def test_scales_to_target_word_count(self, sample_profile_default):
        """Test that metrics are scaled to target word count."""
        target_words = 500
        expected = calculate_expected_metrics(sample_profile_default, target_words)

        # Baseline is 1000 words, so scale factor is 0.5
        assert 'sentence_length' in expected
        # Sentence length should not be scaled (it's an average)
        assert expected['sentence_length'] == sample_profile_default['metrics']['basic']['avg_sentence_length']

    def test_returns_all_expected_fields(self, sample_profile_default):
        """Test that all expected metric fields are returned."""
        expected = calculate_expected_metrics(sample_profile_default, 1000)

        required_fields = [
            'sentence_length', 'paragraph_length',
            'lexical_diversity', 'passive_voice_percentage'
        ]

        for field in required_fields:
            assert field in expected, f"Missing expected field: {field}"


@pytest.mark.analysis
class TestCompareToProfile:
    """Test the compare_to_profile() function."""

    def test_returns_comparison_structure(self, sample_text_short, sample_profile_default):
        """Test that comparison returns expected structure."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)

        required_fields = ['actual', 'expected', 'differences', 'quality_checks']
        for field in required_fields:
            assert field in comparison, f"Missing field: {field}"

    def test_calculates_word_count_difference(self, sample_text_short, sample_profile_default):
        """Test that word count difference is calculated."""
        target = 150
        comparison = compare_to_profile(sample_text_short, sample_profile_default, target)

        assert 'word_count' in comparison['differences']
        word_diff = comparison['differences']['word_count']

        assert 'actual' in word_diff
        assert 'target' in word_diff
        assert 'difference' in word_diff
        assert 'percentage' in word_diff

        assert word_diff['target'] == target

    def test_calculates_sentence_length_difference(self, sample_text_short, sample_profile_default):
        """Test that sentence length difference is calculated."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)

        assert 'sentence_length' in comparison['differences']
        sent_diff = comparison['differences']['sentence_length']

        assert 'actual' in sent_diff
        assert 'expected' in sent_diff
        assert 'difference' in sent_diff
        assert 'percentage' in sent_diff

    def test_generates_quality_checks(self, sample_text_short, sample_profile_default):
        """Test that quality checks are generated."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)

        quality_checks = comparison['quality_checks']
        assert isinstance(quality_checks, list)
        assert len(quality_checks) > 0

        # Check format (should be strings starting with ✅ or ⚠️)
        for check in quality_checks:
            assert isinstance(check, str)
            assert check.startswith('✅') or check.startswith('⚠️')


@pytest.mark.analysis
class TestGenerateReport:
    """Test the generate_report() function."""

    def test_generates_markdown_report(self, sample_text_short, sample_profile_default):
        """Test that report is generated in markdown format."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)
        report = generate_report(comparison, "Test Report")

        assert isinstance(report, str)
        assert len(report) > 0

        # Check for markdown elements
        assert '##' in report  # Headers
        assert '-' in report   # List items
        assert '**' in report  # Bold text

    def test_report_contains_all_sections(self, sample_text_short, sample_profile_default):
        """Test that report contains all expected sections."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)
        report = generate_report(comparison, "Test Report")

        expected_sections = [
            'Basic Metrics',
            'Transition Analysis',
            'Passive Voice',
            'Quality Assessment'
        ]

        for section in expected_sections:
            assert section in report, f"Missing section: {section}"

    def test_report_includes_test_name(self, sample_text_short, sample_profile_default):
        """Test that report includes the test name."""
        comparison = compare_to_profile(sample_text_short, sample_profile_default, 150)
        test_name = "My Custom Test"
        report = generate_report(comparison, test_name)

        assert test_name in report


@pytest.mark.analysis
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_handling(self):
        """Test behavior with empty text."""
        result = analyze_text("")

        assert result['word_count'] == 0
        assert result['sentence_count'] == 0
        assert result['total_transitions'] == 0

    def test_single_sentence_text(self):
        """Test analysis of single-sentence text."""
        text = "Dies ist ein einzelner Satz."
        result = analyze_text(text)

        assert result['sentence_count'] == 1
        assert result['word_count'] > 0

    def test_text_without_transitions(self):
        """Test text with no transition words."""
        text = "Erste Aussage. Zweite Aussage. Dritte Aussage."
        result = analyze_text(text)

        # May have "und" but otherwise minimal transitions
        assert result['total_transitions'] >= 0
        assert result['transition_density'] >= 0
