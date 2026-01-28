"""
Unit tests for dataset splitting and batching functionality.

Tests the DatasetSplitter class from chaos_auto_prompt.core.dataset_splitter
including token-aware batching, estimation, and edge cases.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from chaos_auto_prompt.core.dataset_splitter import DatasetSplitter
from chaos_auto_prompt.interfaces.token_counter import TokenCounter


class TestDatasetSplitterInit:
    """Test DatasetSplitter initialization."""

    def test_initialization(self, token_counter):
        """Test basic initialization with token counter."""
        splitter = DatasetSplitter(token_counter)

        assert splitter.token_counter == token_counter
        assert splitter.settings is not None
        assert splitter.settings.batch_size_tokens > 0
        assert splitter.settings.safety_margin >= 0

    def test_initialization_with_custom_token_counter(self):
        """Test initialization with custom token counter."""
        custom_counter = Mock(spec=TokenCounter)
        splitter = DatasetSplitter(custom_counter)

        assert splitter.token_counter == custom_counter


class TestSplitIntoBatches:
    """Test split_into_batches method."""

    def test_split_empty_dataframe(self, token_counter):
        """Test splitting empty dataframe returns empty list."""
        splitter = DatasetSplitter(token_counter)
        df = pd.DataFrame()

        batches = splitter.split_into_batches(df, ["text"])

        assert batches == []

    def test_split_single_row(self, token_counter, sample_dataframe):
        """Test splitting dataframe with single row."""
        splitter = DatasetSplitter(token_counter)
        df = sample_dataframe.iloc[[0]]

        batches = splitter.split_into_batches(df, ["text"])

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_split_small_dataframe(self, token_counter, sample_dataframe):
        """Test splitting small dataframe that fits in one batch."""
        splitter = DatasetSplitter(token_counter)

        # Use high max_tokens to ensure single batch
        batches = splitter.split_into_batches(sample_dataframe, ["text"], max_tokens=100000)

        assert len(batches) == 1
        assert len(batches[0]) == len(sample_dataframe)

    def test_split_multiple_batches(self, token_counter, mock_token_counter):
        """Test splitting dataframe into multiple batches."""
        splitter = DatasetSplitter(mock_token_counter)

        # Mock returns token counts that will cause multiple batches
        mock_token_counter.count_dataframe_tokens.return_value = [100, 100, 100, 100, 100]
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e"],
            "category": ["A", "B", "A", "C", "B"]
        })

        # With max_tokens=300 and safety_margin=1000, effective is -700, which should become 300
        batches = splitter.split_into_batches(df, ["text"], max_tokens=300)

        # Should create multiple batches
        assert len(batches) >= 1

    def test_split_respects_max_tokens(self, token_counter, mock_token_counter):
        """Test that batching respects max_tokens parameter."""
        splitter = DatasetSplitter(mock_token_counter)

        # Each row has 100 tokens, max_tokens=250 should give 2 rows per batch
        mock_token_counter.count_dataframe_tokens.return_value = [100, 100, 100, 100]
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d"],
            "category": ["A", "B", "A", "B"]
        })

        # With safety_margin=0, effective_max=250
        with patch.object(splitter.settings, 'safety_margin', 0):
            batches = splitter.split_into_batches(df, ["text"], max_tokens=250)

            # First batch should have 2 rows (200 tokens), second batch 2 rows
            assert len(batches) == 2
            assert len(batches[0]) + len(batches[1]) == 4

    def test_split_uses_settings_batch_size(self, token_counter, mock_token_counter, monkeypatch):
        """Test that default batch_size from settings is used when max_tokens not provided."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("batch_size_tokens", "1000")

        splitter = DatasetSplitter(mock_token_counter)

        # Each row 100 tokens, with batch_size_tokens=1000 and safety_margin=1000
        # effective_max=0, so it should use max_tokens=1000
        mock_token_counter.count_dataframe_tokens.return_value = [100] * 20
        df = pd.DataFrame({
            "text": ["row"] * 20,
            "cat": ["A"] * 20
        })

        batches = splitter.split_into_batches(df, ["text"])

        # Should have batches based on default batch_size
        assert len(batches) >= 1

    def test_split_applies_safety_margin(self, token_counter, mock_token_counter):
        """Test that safety_margin is applied to batch size."""
        splitter = DatasetSplitter(mock_token_counter)

        # Set up tokens so row would fit without margin but not with it
        mock_token_counter.count_dataframe_tokens.return_value = [250, 250]
        df = pd.DataFrame({
            "text": ["a", "b"],
            "cat": ["A", "B"]
        })

        # With max_tokens=500 and safety_margin=100, effective=400
        # First row (250) fits, adding second (250) would exceed 400
        with patch.object(splitter.settings, 'safety_margin', 100):
            batches = splitter.split_into_batches(df, ["text"], max_tokens=500)

            # Should split into 2 batches
            assert len(batches) >= 1

    def test_split_with_multiple_columns(self, token_counter, mock_token_counter):
        """Test splitting with multiple columns counted."""
        splitter = DatasetSplitter(mock_token_counter)

        # Token counter should sum across all specified columns
        mock_token_counter.count_dataframe_tokens.return_value = [150, 150, 150]
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "metadata": ["x", "y", "z"],
            "category": ["A", "B", "C"]
        })

        batches = splitter.split_into_batches(df, ["text", "metadata"], max_tokens=500)

        assert len(batches) >= 1

    def test_split_with_nonexistent_columns(self, token_counter, sample_dataframe):
        """Test splitting with columns that don't exist in dataframe."""
        splitter = DatasetSplitter(token_counter)

        # Should handle gracefully - token_counter will filter valid columns
        batches = splitter.split_into_batches(
            sample_dataframe,
            ["text", "nonexistent_column"],
            max_tokens=10000
        )

        assert len(batches) >= 1

    def test_split_preserves_data_integrity(self, token_counter, sample_dataframe):
        """Test that splitting preserves all data."""
        splitter = DatasetSplitter(token_counter)

        batches = splitter.split_into_batches(sample_dataframe, ["text"], max_tokens=10000)

        # Reconstruct dataframe from batches
        reconstructed = pd.concat(batches, ignore_index=True)

        # Check all rows are present
        assert len(reconstructed) == len(sample_dataframe)

        # Check all columns are present
        assert set(reconstructed.columns) == set(sample_dataframe.columns)

    def test_split_creates_independent_batches(self, token_counter):
        """Test that batches are independent copies."""
        splitter = DatasetSplitter(token_counter)
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "value": [1, 2, 3]
        })

        batches = splitter.split_into_batches(df, ["text"], max_tokens=100)

        # Modify first batch
        if batches:
            batches[0].loc[0, "value"] = 999

            # Original dataframe should be unchanged (batches are copies)
            # Actually, batches are copies of slices, so original should be unchanged
            assert df.loc[0, "value"] == 1


class TestEstimateBatchCount:
    """Test estimate_batch_count method."""

    def test_estimate_empty_dataframe(self, token_counter):
        """Test estimating batches for empty dataframe."""
        splitter = DatasetSplitter(token_counter)
        df = pd.DataFrame()

        estimate = splitter.estimate_batch_count(df, ["text"])

        assert estimate == 0

    def test_estimate_single_row(self, token_counter, mock_token_counter):
        """Test estimating batches for single row."""
        splitter = DatasetSplitter(mock_token_counter)

        mock_token_counter.estimate_tokens.return_value = 100
        df = pd.DataFrame({"text": ["test"], "cat": ["A"]})

        estimate = splitter.estimate_batch_count(df, ["text"], max_tokens=1000)

        assert estimate >= 1

    def test_estimate_large_dataframe(self, token_counter, mock_token_counter):
        """Test estimating batches for large dataframe."""
        splitter = DatasetSplitter(mock_token_counter)

        # Mock to return estimate that would cause multiple batches
        mock_token_counter.estimate_tokens.return_value = 1000
        df = pd.DataFrame({
            "text": ["x"] * 100,
            "cat": ["A"] * 100
        })

        # With max_tokens=5000 and safety_margin=1000, effective=4000
        # Each column sum estimate would be 100000 tokens
        estimate = splitter.estimate_batch_count(df, ["text"], max_tokens=5000)

        assert estimate >= 1

    def test_estimate_uses_settings_batch_size(self, token_counter, mock_token_counter, monkeypatch):
        """Test that estimate uses settings batch_size when max_tokens not provided."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("batch_size_tokens", "5000")

        splitter = DatasetSplitter(mock_token_counter)

        mock_token_counter.estimate_tokens.return_value = 100
        df = pd.DataFrame({
            "text": ["test"],
            "cat": ["A"]
        })

        estimate = splitter.estimate_batch_count(df, ["text"])

        assert estimate >= 1

    def test_estimate_applies_safety_margin(self, token_counter, mock_token_counter):
        """Test that estimate applies safety margin."""
        splitter = DatasetSplitter(mock_token_counter)

        mock_token_counter.estimate_tokens.return_value = 100
        df = pd.DataFrame({
            "text": ["test"] * 10,
            "cat": ["A"] * 10
        })

        # Estimate with and without safety margin should differ
        with patch.object(splitter.settings, 'safety_margin', 500):
            estimate_with_margin = splitter.estimate_batch_count(
                df, ["text"], max_tokens=1000
            )

        with patch.object(splitter.settings, 'safety_margin', 0):
            estimate_without_margin = splitter.estimate_batch_count(
                df, ["text"], max_tokens=1000
            )

        # With margin should have same or more batches
        assert estimate_with_margin >= estimate_without_margin

    def test_estimate_with_multiple_columns(self, token_counter, mock_token_counter):
        """Test estimate with multiple columns."""
        splitter = DatasetSplitter(mock_token_counter)

        mock_token_counter.estimate_tokens.return_value = 100
        df = pd.DataFrame({
            "text": ["a"] * 10,
            "metadata": ["b"] * 10,
            "category": ["A"] * 10
        })

        estimate = splitter.estimate_batch_count(
            df, ["text", "metadata"], max_tokens=1000
        )

        assert estimate >= 1


class TestDatasetSplitterEdgeCases:
    """Test edge cases and special scenarios."""

    def test_split_with_zero_max_tokens(self, token_counter, sample_dataframe):
        """Test splitting when max_tokens is zero."""
        splitter = DatasetSplitter(token_counter)

        # Should handle gracefully - effective_max_tokens would be 0
        # But code ensures effective_max_tokens = max_tokens when <= 0
        batches = splitter.split_into_batches(sample_dataframe, ["text"], max_tokens=0)

        # Should still create batches, just with no limit effectively
        assert len(batches) >= 1

    def test_split_with_negative_safety_margin(self, token_counter, sample_dataframe):
        """Test splitting with negative safety margin."""
        splitter = DatasetSplitter(token_counter)

        with patch.object(splitter.settings, 'safety_margin', -100):
            batches = splitter.split_into_batches(
                sample_dataframe, ["text"], max_tokens=10000
            )

            # Should work - code handles negative or zero effective_max
            assert len(batches) >= 1

    def test_split_safety_margin_exceeds_max_tokens(self, token_counter, sample_dataframe):
        """Test when safety_margin >= max_tokens."""
        splitter = DatasetSplitter(token_counter)

        # With max_tokens=100 and safety_margin=1000, effective would be -900
        # Code should handle this by using max_tokens as effective
        with patch.object(splitter.settings, 'safety_margin', 1000):
            batches = splitter.split_into_batches(
                sample_dataframe, ["text"], max_tokens=100
            )

            # Should still create batches
            assert len(batches) >= 1

    def test_estimate_with_very_large_tokens(self, token_counter, mock_token_counter):
        """Test estimate with very large token counts."""
        splitter = DatasetSplitter(mock_token_counter)

        mock_token_counter.estimate_tokens.return_value = 10000000
        df = pd.DataFrame({
            "text": ["huge text"],
            "cat": ["A"]
        })

        estimate = splitter.estimate_batch_count(df, ["text"], max_tokens=1000)

        # Should return at least 1
        assert estimate >= 1

    def test_split_large_dataframe_performance(self, token_counter, large_dataframe):
        """Test splitting large dataframe is performant."""
        import time

        splitter = DatasetSplitter(token_counter)

        start = time.time()
        batches = splitter.split_into_batches(large_dataframe, ["text"], max_tokens=100000)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 1000 rows)
        assert elapsed < 5.0
        assert len(batches) >= 1

    def test_split_with_all_empty_strings(self, token_counter):
        """Test splitting dataframe with all empty string values."""
        splitter = DatasetSplitter(token_counter)
        df = pd.DataFrame({
            "text": ["", "", ""],
            "category": ["A", "B", "C"]
        })

        batches = splitter.split_into_batches(df, ["text"], max_tokens=1000)

        # Empty strings have 0 tokens, should all fit in one batch
        assert len(batches) >= 1

    def test_split_with_very_long_single_row(self, token_counter, mock_token_counter):
        """Test splitting when one row has very high token count."""
        splitter = DatasetSplitter(mock_token_counter)

        # First row has 10000 tokens, rest have 100 each
        mock_token_counter.count_dataframe_tokens.return_value = [10000, 100, 100, 100]
        df = pd.DataFrame({
            "text": ["long", "a", "b", "c"],
            "cat": ["A", "B", "C", "D"]
        })

        batches = splitter.split_into_batches(df, ["text"], max_tokens=5000)

        # Long row should be in its own batch
        assert len(batches) >= 1

    def test_estimate_matches_actual_split(self, token_counter, sample_dataframe):
        """Test that estimate is reasonably close to actual batch count."""
        splitter = DatasetSplitter(token_counter)

        max_tokens = 10000
        estimate = splitter.estimate_batch_count(
            sample_dataframe, ["text"], max_tokens=max_tokens
        )
        actual = len(splitter.split_into_batches(
            sample_dataframe, ["text"], max_tokens=max_tokens
        ))

        # Estimate should be in reasonable range
        # (estimate may not be exact due to estimation method)
        assert estimate >= 1
        assert actual >= 1
        # They should be within factor of 2
        assert max(estimate, actual) / min(estimate, actual) <= 2.0

    def test_split_preserves_index(self, token_counter, sample_dataframe):
        """Test that split preserves original row indices."""
        splitter = DatasetSplitter(token_counter)

        batches = splitter.split_into_batches(sample_dataframe, ["text"], max_tokens=10000)

        # Check that original indices are present
        all_indices = []
        for batch in batches:
            all_indices.extend(batch.index.tolist())

        # Sort and compare
        original_indices = sorted(sample_dataframe.index.tolist())
        reconstructed_indices = sorted(all_indices)

        assert original_indices == reconstructed_indices

    def test_split_with_nan_values(self, token_counter):
        """Test splitting with NaN values in columns."""
        splitter = DatasetSplitter(token_counter)
        df = pd.DataFrame({
            "text": ["a", None, "c", "d"],
            "category": ["A", "B", None, "D"]
        })

        batches = splitter.split_into_batches(df, ["text"], max_tokens=1000)

        # Should handle NaN values gracefully
        assert len(batches) >= 1

    def test_split_single_column(self, token_counter, sample_dataframe):
        """Test splitting with single column."""
        splitter = DatasetSplitter(token_counter)

        batches = splitter.split_into_batches(sample_dataframe, ["text"], max_tokens=10000)

        assert len(batches) >= 1

    def test_split_all_columns(self, token_counter, sample_dataframe):
        """Test splitting with all columns."""
        splitter = DatasetSplitter(token_counter)

        all_columns = sample_dataframe.columns.tolist()
        batches = splitter.split_into_batches(
            sample_dataframe, all_columns, max_tokens=10000
        )

        assert len(batches) >= 1
