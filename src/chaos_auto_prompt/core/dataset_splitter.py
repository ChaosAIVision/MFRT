"""
Dataset splitter with configurable token-aware batching.

Uses settings from centralized config management.
"""

from typing import List
import pandas as pd

from chaos_auto_prompt.interfaces.token_counter import TokenCounter
from chaos_auto_prompt.config.settings import get_settings


class DatasetSplitter:
    """Split datasets into batches based on token limits from settings."""

    def __init__(self, token_counter: TokenCounter):
        """
        Initialize with a token counter dependency.

        Args:
            token_counter: TokenCounter implementation for counting tokens
        """
        self.token_counter = token_counter
        self.settings = get_settings()

    def split_into_batches(
        self, df: pd.DataFrame, columns: List[str], max_tokens: int | None = None
    ) -> List[pd.DataFrame]:
        """
        Split dataframe into batches that fit within token limit.

        Uses batch_size_tokens from settings if max_tokens not provided.
        Applies safety_margin from settings to ensure batches stay within limits.

        Args:
            df: DataFrame to split
            columns: Column names to count tokens for
            max_tokens: Maximum tokens per batch (optional, defaults to batch_size_tokens from settings)

        Returns:
            List of DataFrame batches
        """
        if df.empty:
            return []

        # Use configured batch size from settings if not provided
        if max_tokens is None:
            max_tokens = self.settings.batch_size_tokens

        # Apply safety margin from settings
        effective_max_tokens = max_tokens - self.settings.safety_margin

        # Ensure effective_max_tokens is positive
        if effective_max_tokens <= 0:
            effective_max_tokens = max_tokens

        # Count tokens for each row
        row_token_counts = self.token_counter.count_dataframe_tokens(df, columns)

        # Pre-calculate batch boundaries to minimize DataFrame operations
        batch_boundaries = []
        current_batch_start = 0
        current_batch_tokens = 0

        for idx, token_count in enumerate(row_token_counts):
            # If adding this row would exceed limit, finalize current batch
            if (
                current_batch_tokens + token_count > effective_max_tokens
                and idx > current_batch_start
            ):
                batch_boundaries.append((current_batch_start, idx))
                current_batch_start = idx
                current_batch_tokens = token_count
            else:
                current_batch_tokens += token_count

        # Add final batch boundary
        if current_batch_start < len(df):
            batch_boundaries.append((current_batch_start, len(df)))

        # Create DataFrame slices efficiently using boundaries
        batches = []
        for start_idx, end_idx in batch_boundaries:
            # Use iloc with slice for better performance than index lists
            batch_df = df.iloc[start_idx:end_idx].copy()
            batches.append(batch_df)

        return batches

    def estimate_batch_count(
        self, df: pd.DataFrame, columns: List[str], max_tokens: int | None = None
    ) -> int:
        """
        Quickly estimate how many batches will be needed.

        Uses batch_size_tokens from settings if max_tokens not provided.

        Args:
            df: DataFrame to estimate for
            columns: Column names to count tokens for
            max_tokens: Maximum tokens per batch (optional, defaults to batch_size_tokens from settings)

        Returns:
            Estimated number of batches needed
        """
        if df.empty:
            return 0

        # Use configured batch size from settings if not provided
        if max_tokens is None:
            max_tokens = self.settings.batch_size_tokens

        # Apply safety margin from settings
        effective_max_tokens = max_tokens - self.settings.safety_margin

        # Ensure effective_max_tokens is positive
        if effective_max_tokens <= 0:
            effective_max_tokens = max_tokens

        # Use fast estimation
        total_tokens = sum(
            self.token_counter.estimate_tokens(str(df[col].sum()))
            for col in columns
            if col in df.columns
        )

        # Ceiling division
        return max(1, (total_tokens + effective_max_tokens - 1) // effective_max_tokens)
