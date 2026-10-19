"""Memory helpers (extracted from notebooks 04 - preprocessing)."""
import pandas as pd


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Deep memory usage of a DataFrame in MB (same formula as notebook 04)."""
    return df.memory_usage(deep=True).sum() / 1024**2


def memory_reduction_pct(before_mb: float, after_mb: float) -> float:
    """Percentage reduction between two memory measurements."""
    return (before_mb - after_mb) / before_mb * 100
