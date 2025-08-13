"""
Utility functions for XAUUSD market structure detection.

This module provides common helper functions used across the project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


def setup_project_directories(base_path: str = ".") -> Dict[str, str]:
    """
    Create project directory structure.
    
    Args:
        base_path: Base path for project
        
    Returns:
        Dictionary of directory names to paths
    """
    base = Path(base_path)
    
    directories = {
        'data_raw': 'data/raw',
        'data_clean': 'data/clean',
        'data_resampled': 'data/resampled',
        'outputs_events': 'outputs/events',
        'logs': 'logs',
        'notebooks': 'notebooks',
        'tests_data': 'tests/test_data'
    }
    
    created_paths = {}
    for name, path in directories.items():
        full_path = base / path
        full_path.mkdir(parents=True, exist_ok=True)
        created_paths[name] = str(full_path)
        logger.info(f"Created directory: {full_path}")
    
    return created_paths


def validate_ohlc_candle(open_price: float, high: float, low: float, close: float) -> bool:
    """
    Validate individual OHLC candle data.
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        
    Returns:
        True if valid OHLC candle
    """
    if any(pd.isna([open_price, high, low, close])):
        return False
    
    if any(price <= 0 for price in [open_price, high, low, close]):
        return False
    
    # High should be >= max(open, close)
    # Low should be <= min(open, close)
    return (
        high >= max(open_price, close) and
        low <= min(open_price, close) and
        high >= low
    )


def classify_candle_color(open_price: float, close: float) -> str:
    """
    Classify candle color based on open/close relationship.
    
    Args:
        open_price: Opening price
        close: Closing price
        
    Returns:
        'green', 'red', or 'doji'
    """
    if close > open_price:
        return 'green'
    elif close < open_price:
        return 'red'
    else:
        return 'doji'


def calculate_candle_body_pct(open_price: float, high: float, low: float, close: float) -> float:
    """
    Calculate candle body size as percentage of total range.
    
    Args:
        open_price: Opening price
        high: High price
        low: Low price
        close: Closing price
        
    Returns:
        Body percentage (0.0 to 1.0)
    """
    if high == low:  # Avoid division by zero
        return 0.0
    
    body_size = abs(close - open_price)
    total_range = high - low
    
    return body_size / total_range if total_range > 0 else 0.0


def detect_market_gaps(df: pd.DataFrame, gap_threshold_pct: float = 0.2) -> List[Dict]:
    """
    Detect gaps in market data (weekends, holidays).
    
    Args:
        df: DataFrame with datetime column
        gap_threshold_pct: Minimum gap size as % of average price
        
    Returns:
        List of gap information dictionaries
    """
    if len(df) < 2:
        return []
    
    gaps = []
    avg_price = df[['open', 'high', 'low', 'close']].mean().mean()
    
    for i in range(1, len(df)):
        prev_close = df.iloc[i-1]['close']
        curr_open = df.iloc[i]['open']
        
        gap_size = abs(curr_open - prev_close)
        gap_pct = gap_size / avg_price
        
        if gap_pct >= gap_threshold_pct:
            gaps.append({
                'index': i,
                'datetime': df.iloc[i]['datetime'],
                'prev_close': prev_close,
                'curr_open': curr_open,
                'gap_size': gap_size,
                'gap_pct': gap_pct,
                'gap_direction': 'up' if curr_open > prev_close else 'down'
            })
    
    logger.info(f"Detected {len(gaps)} market gaps")
    return gaps


def calculate_price_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive price statistics.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Dictionary of price statistics
    """
    stats = {
        'total_candles': len(df),
        'price_range': {
            'min': float(df[['open', 'high', 'low', 'close']].min().min()),
            'max': float(df[['open', 'high', 'low', 'close']].max().max()),
            'range': float(df[['open', 'high', 'low', 'close']].max().max() - 
                          df[['open', 'high', 'low', 'close']].min().min())
        },
        'candle_analysis': {
            'green_candles': int((df['close'] > df['open']).sum()),
            'red_candles': int((df['close'] < df['open']).sum()),
            'doji_candles': int((df['close'] == df['open']).sum())
        },
        'volatility_metrics': {
            'avg_high_low_spread': float((df['high'] - df['low']).mean()),
            'avg_body_size': float((df['close'] - df['open']).abs().mean()),
            'max_single_candle_move': float((df['high'] - df['low']).max())
        }
    }
    
    # Add percentages
    total = stats['total_candles']
    if total > 0:
        stats['candle_analysis']['green_pct'] = stats['candle_analysis']['green_candles'] / total * 100
        stats['candle_analysis']['red_pct'] = stats['candle_analysis']['red_candles'] / total * 100
        stats['candle_analysis']['doji_pct'] = stats['candle_analysis']['doji_candles'] / total * 100
    
    return stats


def export_to_json(data: Dict, filepath: str, indent: int = 2) -> str:
    """
    Export data to JSON file with proper serialization.
    
    Args:
        data: Data to export
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        Path to saved file
    """
    def json_serializer(obj):
        """JSON serializer for special types."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=json_serializer)
    
    logger.info(f"Exported data to {filepath}")
    return str(filepath)


def load_from_json(filepath: str) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data from {filepath}")
    return data


def create_sample_data(n_candles: int = 1000, 
                      start_price: float = 2000.0,
                      volatility: float = 0.01) -> pd.DataFrame:
    """
    Create sample OHLC data for testing.
    
    Args:
        n_candles: Number of candles to generate
        start_price: Starting price
        volatility: Price volatility factor
        
    Returns:
        DataFrame with sample OHLC data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate datetime index (15-minute intervals)
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    datetimes = [start_time + timedelta(minutes=15*i) for i in range(n_candles)]
    
    # Generate price data using random walk
    price_changes = np.random.normal(0, volatility, n_candles)
    prices = [start_price]
    
    for change in price_changes[:-1]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Create OHLC data
    data = []
    for i, (dt, price) in enumerate(zip(datetimes, prices)):
        # Generate OHLC based on price with some randomness
        open_price = price * (1 + np.random.normal(0, volatility/4))
        close_price = price * (1 + np.random.normal(0, volatility/4))
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/6)))
        low = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/6)))
        
        data.append({
            'datetime': dt,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': np.random.randint(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {n_candles} sample candles")
    return df


def validate_timeframe_compatibility(base_tf: str, target_tf: str) -> bool:
    """
    Validate that target timeframe is compatible with base timeframe.
    
    Args:
        base_tf: Base timeframe (e.g., '15min')
        target_tf: Target timeframe (e.g., '1H')
        
    Returns:
        True if compatible
    """
    # Convert to minutes for comparison
    tf_minutes = {
        '15min': 15,
        '1H': 60,
        '4H': 240,
        '1D': 1440,
        'Daily': 1440
    }
    
    base_min = tf_minutes.get(base_tf)
    target_min = tf_minutes.get(target_tf)
    
    if base_min is None or target_min is None:
        return False
    
    # Target should be a multiple of base
    return target_min >= base_min and target_min % base_min == 0


def format_price(price: float, decimals: int = 2) -> str:
    """Format price with proper decimal places."""
    return f"{price:.{decimals}f}"


def format_datetime(dt: Union[datetime, pd.Timestamp, str], 
                   format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime consistently."""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    return dt.strftime(format_str)


def calculate_time_difference(dt1: datetime, dt2: datetime) -> Dict:
    """
    Calculate time difference with breakdown.
    
    Args:
        dt1: First datetime
        dt2: Second datetime
        
    Returns:
        Dictionary with time difference breakdown
    """
    diff = abs(dt2 - dt1)
    
    return {
        'total_seconds': diff.total_seconds(),
        'total_minutes': diff.total_seconds() / 60,
        'total_hours': diff.total_seconds() / 3600,
        'total_days': diff.days,
        'breakdown': str(diff)
    }


def main():
    """Example usage of utility functions."""
    # Setup directories
    dirs = setup_project_directories()
    print("Created directories:", dirs)
    
    # Generate sample data
    sample_df = create_sample_data(100)
    print(f"Generated sample data with {len(sample_df)} candles")
    
    # Calculate statistics
    stats = calculate_price_statistics(sample_df)
    print("Price statistics:", stats)


if __name__ == "_main_":
    main()