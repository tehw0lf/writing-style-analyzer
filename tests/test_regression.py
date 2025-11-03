"""
Regression tests.

Tests that analysis results match known baselines, ensuring that
code changes don't break existing functionality.

Uses ONLY synthetic test data and baselines (no personal data).
"""

import pytest
import json
from pathlib import Path
from german_academic_analyzer import analyze_text


@pytest.mark.regression
class TestRegressionAgainstBaselines:
    """Test that current analysis matches saved baselines."""

    def load_baseline(self, baselines_dir, baseline_name):
        """Load a baseline analysis file."""
        baseline_path = baselines_dir / baseline_name
        with open(baseline_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_short_text_matches_baseline(
        self, sample_text_short, baselines_dir, test_config
    ):
        """Test that short text analysis matches baseline."""
        # Load baseline
        baseline = self.load_baseline(baselines_dir, "baseline_short_text.json")

        # Analyze current
        current = analyze_text(sample_text_short)

        # Get tolerance
        max_deviation = test_config['regression']['max_deviation_percentage']
        metrics_to_compare = test_config['regression']['metrics_to_compare']

        # Compare each metric
        for metric in metrics_to_compare:
            baseline_value = baseline[metric]
            current_value = current[metric]

            # Skip zero values
            if baseline_value == 0:
                assert current_value == 0, (
                    f"Regression in {metric}: baseline=0, current={current_value}"
                )
                continue

            # Calculate percentage deviation
            deviation_pct = abs((current_value - baseline_value) / baseline_value * 100)

            assert deviation_pct <= max_deviation, (
                f"Regression detected in {metric}: "
                f"baseline={baseline_value}, current={current_value}, "
                f"deviation={deviation_pct:.1f}% (max allowed: {max_deviation}%)"
            )

    def test_long_text_matches_baseline(
        self, sample_text_long, baselines_dir, test_config
    ):
        """Test that long text analysis matches baseline."""
        # Load baseline
        baseline = self.load_baseline(baselines_dir, "baseline_long_text.json")

        # Analyze current
        current = analyze_text(sample_text_long)

        # Get tolerance
        max_deviation = test_config['regression']['max_deviation_percentage']
        metrics_to_compare = test_config['regression']['metrics_to_compare']

        # Compare each metric
        for metric in metrics_to_compare:
            baseline_value = baseline[metric]
            current_value = current[metric]

            # Skip zero values
            if baseline_value == 0:
                assert current_value == 0, (
                    f"Regression in {metric}: baseline=0, current={current_value}"
                )
                continue

            # Calculate percentage deviation
            deviation_pct = abs((current_value - baseline_value) / baseline_value * 100)

            assert deviation_pct <= max_deviation, (
                f"Regression detected in {metric}: "
                f"baseline={baseline_value}, current={current_value}, "
                f"deviation={deviation_pct:.1f}% (max allowed: {max_deviation}%)"
            )

    def test_transition_categories_match_baseline(
        self, sample_text_short, baselines_dir
    ):
        """Test that transition category detection matches baseline."""
        # Load baseline
        baseline = self.load_baseline(baselines_dir, "baseline_short_text.json")

        # Analyze current
        current = analyze_text(sample_text_short)

        # Compare transition categories
        baseline_cats = baseline['transitions_by_category']
        current_cats = current['transitions_by_category']

        for category in baseline_cats.keys():
            baseline_count = baseline_cats[category]['count']
            current_count = current_cats[category]['count']

            assert baseline_count == current_count, (
                f"Regression in {category} transitions: "
                f"baseline={baseline_count}, current={current_count}"
            )


@pytest.mark.regression
class TestConsistencyAcrossTexts:
    """Test that analysis behaves consistently across different texts."""

    def test_longer_text_has_more_words(self, sample_text_short, sample_text_long):
        """Test that longer text consistently has more words."""
        short_analysis = analyze_text(sample_text_short)
        long_analysis = analyze_text(sample_text_long)

        assert long_analysis['word_count'] > short_analysis['word_count'], (
            "Long text should always have more words than short text"
        )

    def test_longer_text_has_more_transitions(
        self, sample_text_short, sample_text_long
    ):
        """Test that longer text has more total transitions."""
        short_analysis = analyze_text(sample_text_short)
        long_analysis = analyze_text(sample_text_long)

        # Absolute count should be higher (not density)
        assert long_analysis['total_transitions'] > short_analysis['total_transitions'], (
            "Long text should have more total transitions"
        )

    def test_analysis_is_deterministic(self, sample_text_short):
        """Test that analyzing same text twice gives identical results."""
        analysis1 = analyze_text(sample_text_short)
        analysis2 = analyze_text(sample_text_short)

        # All metrics should be identical
        assert analysis1 == analysis2, (
            "Analysis should be deterministic (identical results for same text)"
        )


@pytest.mark.regression
@pytest.mark.slow
class TestProfileBasedRegression:
    """Test that profile-based analysis remains consistent."""

    def test_profile_comparison_is_consistent(
        self, sample_text_short, sample_profile_default
    ):
        """Test that profile comparison produces consistent results."""
        from german_academic_analyzer import compare_to_profile

        # Run comparison twice
        comparison1 = compare_to_profile(sample_text_short, sample_profile_default, 150)
        comparison2 = compare_to_profile(sample_text_short, sample_profile_default, 150)

        # Results should be identical
        assert comparison1['actual'] == comparison2['actual']
        assert comparison1['expected'] == comparison2['expected']
        assert comparison1['differences'] == comparison2['differences']

    def test_different_targets_scale_expectations(
        self, sample_text_short, sample_profile_default
    ):
        """Test that different target word counts scale expectations properly."""
        from german_academic_analyzer import compare_to_profile

        comparison_500 = compare_to_profile(sample_text_short, sample_profile_default, 500)
        comparison_1000 = compare_to_profile(sample_text_short, sample_profile_default, 1000)

        # Expected metrics should scale (not actual)
        assert comparison_500['expected'] != comparison_1000['expected']
        assert comparison_500['actual'] == comparison_1000['actual']
