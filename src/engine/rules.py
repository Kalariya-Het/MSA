"""
Market structure rules implementation for XAUUSD detection.

This module contains all the specific rule functions for detecting:
- Two-candle confirmation patterns (red/green)  
- Engulfing patterns for 4H+ timeframes
- HL/LH extraction logic
- Candle color classification
- Price level calculations

All rules follow strict candle-based logic with no indicators or ATR.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketStructureRules:
    """Contains all market structure detection rules."""
    
    def __init__(self, config: Dict):
        """Initialize rules with configuration."""
        self.config = config
        self.color_rules = config['rules']['candle_color_rule']
        self.min_body_pct = config['rules'].get('min_body_pct', 0.0)
        
    def classify_candle_color(self, candle: pd.Series) -> str:
        """
        Classify candle color based on open/close relationship.
        
        Args:
            candle: Pandas Series with open, high, low, close
            
        Returns:
            'green', 'red', or 'doji'
        """
        open_price = candle['open']
        close_price = candle['close']
        
        if close_price > open_price:
            return 'green'
        elif close_price < open_price:
            return 'red'
        else:
            return 'doji'
    
    def is_candle_significant(self, candle: pd.Series) -> bool:
        """
        Check if candle body is significant enough (optional noise filter).
        
        Args:
            candle: Pandas Series with OHLC data
            
        Returns:
            True if candle meets minimum body size requirement
        """
        if self.min_body_pct <= 0:
            return True  # No filtering
        
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close_price = candle['close']
        
        if high == low:  # Avoid division by zero
            return False
        
        body_size = abs(close_price - open_price)
        total_range = high - low
        body_pct = body_size / total_range
        
        return body_pct >= self.min_body_pct
    
    def is_two_red_confirm(self, candle_a: pd.Series, candle_b: pd.Series) -> bool:
        """
        Check for HH confirmation: Two consecutive red candles where 
        second candle's close is below first candle's low.
        
        Args:
            candle_a: First (previous) candle
            candle_b: Second (current) candle
            
        Returns:
            True if pattern confirms HH
        """
        # Both candles must be red (strict)
        if (self.classify_candle_color(candle_a) != 'red' or
            self.classify_candle_color(candle_b) != 'red'):
            return False
        
        # Both candles must be significant (if filtering enabled)
        if not (self.is_candle_significant(candle_a) and 
                self.is_candle_significant(candle_b)):
            return False
        
        # Second candle's close must be below first candle's low (strict)
        return candle_b['close'] < candle_a['low']
    
    def is_two_green_confirm(self, candle_a: pd.Series, candle_b: pd.Series) -> bool:
        """
        Check for LL confirmation: Two consecutive green candles where
        second candle's close is above first candle's high.
        
        Args:
            candle_a: First (previous) candle
            candle_b: Second (current) candle
            
        Returns:
            True if pattern confirms LL
        """
        # Both candles must be green (strict)
        if (self.classify_candle_color(candle_a) != 'green' or
            self.classify_candle_color(candle_b) != 'green'):
            return False
        
        # Both candles must be significant (if filtering enabled)
        if not (self.is_candle_significant(candle_a) and 
                self.is_candle_significant(candle_b)):
            return False
        
        # Second candle's close must be above first candle's high (strict)
        return candle_b['close'] > candle_a['high']
    
    def is_red_engulfing_green(self, green_candle: pd.Series, red_candle: pd.Series) -> bool:
        """
        Check for red engulfing green pattern (HH confirmation on 4H+).
        Full-body engulfing: red candle's body completely covers green candle's body.
        
        Args:
            green_candle: Previous green candle
            red_candle: Current red candle
            
        Returns:
            True if red engulfs green
        """
        # Validate candle colors
        if (self.classify_candle_color(green_candle) != 'green' or
            self.classify_candle_color(red_candle) != 'red'):
            return False
        
        # Full-body engulfing condition:
        # Red candle's open >= Green candle's close
        # Red candle's close <= Green candle's open
        return (red_candle['open'] >= green_candle['close'] and
                red_candle['close'] <= green_candle['open'])
    
    def is_green_engulfing_red(self, red_candle: pd.Series, green_candle: pd.Series) -> bool:
        """
        Check for green engulfing red pattern (LL confirmation on 4H+).
        Full-body engulfing: green candle's body completely covers red candle's body.
        
        Args:
            red_candle: Previous red candle
            green_candle: Current green candle
            
        Returns:
            True if green engulfs red
        """
        # Validate candle colors
        if (self.classify_candle_color(red_candle) != 'red' or
            self.classify_candle_color(green_candle) != 'green'):
            return False
        
        # Full-body engulfing condition:
        # Green candle's open <= Red candle's close
        # Green candle's close >= Red candle's open
        return (green_candle['open'] <= red_candle['close'] and
                green_candle['close'] >= red_candle['open'])
    
    def find_lowest_low_between(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict]:
        """
        Find the lowest low between two candle indices (for HL detection).
        
        Args:
            df: OHLC DataFrame
            start_idx: Starting candle index (inclusive)
            end_idx: Ending candle index (inclusive)
            
        Returns:
            Dictionary with lowest low info or None
        """
        if start_idx > end_idx or start_idx < 0 or end_idx >= len(df):
            return None
        
        # Get subset of data
        subset = df.iloc[start_idx:end_idx + 1]
        
        if len(subset) == 0:
            return None
        
        # Find index of minimum low
        min_low_idx = subset['low'].idxmin()
        min_low_candle = df.loc[min_low_idx]
        
        # Convert to original DataFrame index
        original_idx = df.index.get_loc(min_low_idx)
        
        return {
            'index': original_idx,
            'price': min_low_candle['low'],
            'datetime': min_low_candle['datetime'],
            'candle_data': {
                'open': min_low_candle['open'],
                'high': min_low_candle['high'],
                'low': min_low_candle['low'],
                'close': min_low_candle['close']
            }
        }
    
    def find_highest_high_between(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict]:
        """
        Find the highest high between two candle indices (for LH detection).
        
        Args:
            df: OHLC DataFrame
            start_idx: Starting candle index (inclusive)  
            end_idx: Ending candle index (inclusive)
            
        Returns:
            Dictionary with highest high info or None
        """
        if start_idx > end_idx or start_idx < 0 or end_idx >= len(df):
            return None
        
        # Get subset of data
        subset = df.iloc[start_idx:end_idx + 1]
        
        if len(subset) == 0:
            return None
        
        # Find index of maximum high
        max_high_idx = subset['high'].idxmax()
        max_high_candle = df.loc[max_high_idx]
        
        # Convert to original DataFrame index
        original_idx = df.index.get_loc(max_high_idx)
        
        return {
            'index': original_idx,
            'price': max_high_candle['high'],
            'datetime': max_high_candle['datetime'],
            'candle_data': {
                'open': max_high_candle['open'],
                'high': max_high_candle['high'],
                'low': max_high_candle['low'],
                'close': max_high_candle['close']
            }
        }
    
    def validate_ohlc_candle(self, candle: pd.Series) -> bool:
        """
        Validate individual OHLC candle data integrity.
        
        Args:
            candle: Pandas Series with OHLC data
            
        Returns:
            True if valid OHLC relationships
        """
        try:
            open_price = candle['open']
            high = candle['high']
            low = candle['low']
            close = candle['close']
            
            # Check for NaN values
            if pd.isna([open_price, high, low, close]).any():
                return False
            
            # Check for positive prices
            if any(price <= 0 for price in [open_price, high, low, close]):
                return False
            
            # OHLC relationship validation
            return (
                high >= max(open_price, close) and
                low <= min(open_price, close) and
                high >= low
            )
        except (KeyError, TypeError):
            return False
    
    def calculate_price_levels(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate key price levels for context (optional utility).
        
        Args:
            df: OHLC DataFrame
            window: Lookback window for calculations
            
        Returns:
            Dictionary with calculated levels
        """
        if len(df) < window:
            window = len(df)
        
        recent_data = df.tail(window)
        
        return {
            'highest_high': recent_data['high'].max(),
            'lowest_low': recent_data['low'].min(),
            'avg_high': recent_data['high'].mean(),
            'avg_low': recent_data['low'].mean(),
            'price_range': recent_data['high'].max() - recent_data['low'].min(),
            'volatility': (recent_data['high'] - recent_data['low']).mean()
        }
    
    def check_consecutive_closes_above(self, df: pd.DataFrame, price_level: float, 
                                     start_idx: int, count: int = 2) -> bool:
        """
        Check for consecutive closes above a price level.
        
        Args:
            df: OHLC DataFrame
            price_level: Price level to check against
            start_idx: Starting index to check from
            count: Number of consecutive closes required
            
        Returns:
            True if pattern found
        """
        if start_idx + count > len(df):
            return False
        
        for i in range(count):
            if df.iloc[start_idx + i]['close'] <= price_level:
                return False
        
        return True
    
    def check_consecutive_closes_below(self, df: pd.DataFrame, price_level: float,
                                     start_idx: int, count: int = 2) -> bool:
        """
        Check for consecutive closes below a price level.
        
        Args:
            df: OHLC DataFrame  
            price_level: Price level to check against
            start_idx: Starting index to check from
            count: Number of consecutive closes required
            
        Returns:
            True if pattern found
        """
        if start_idx + count > len(df):
            return False
        
        for i in range(count):
            if df.iloc[start_idx + i]['close'] >= price_level:
                return False
        
        return True
    
    def find_swing_extremes(self, df: pd.DataFrame, lookback: int = 10) -> Dict:
        """
        Find recent swing highs and lows for context.
        
        Args:
            df: OHLC DataFrame
            lookback: Lookback period for finding swings
            
        Returns:
            Dictionary with swing extremes
        """
        if len(df) < lookback * 2 + 1:
            return {'swing_highs': [], 'swing_lows': []}
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': current_high,
                    'datetime': df.iloc[i]['datetime']
                })
            
            # Check for swing low
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': current_low,
                    'datetime': df.iloc[i]['datetime']
                })
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def detect_gap(self, prev_candle: pd.Series, current_candle: pd.Series, 
                   threshold_pct: float = 0.1) -> Optional[Dict]:
        """
        Detect price gaps between candles.
        
        Args:
            prev_candle: Previous candle
            current_candle: Current candle  
            threshold_pct: Minimum gap size as percentage
            
        Returns:
            Gap information or None
        """
        prev_close = prev_candle['close']
        current_open = current_candle['open']
        
        gap_size = abs(current_open - prev_close)
        gap_pct = gap_size / prev_close
        
        if gap_pct >= threshold_pct:
            return {
                'gap_size': gap_size,
                'gap_pct': gap_pct * 100,
                'direction': 'up' if current_open > prev_close else 'down',
                'prev_close': prev_close,
                'current_open': current_open
            }
        
        return None
    
    def is_inside_bar(self, prev_candle: pd.Series, current_candle: pd.Series) -> bool:
        """
        Check if current candle is an inside bar (range within previous candle).
        
        Args:
            prev_candle: Previous candle
            current_candle: Current candle
            
        Returns:
            True if inside bar pattern
        """
        return (current_candle['high'] <= prev_candle['high'] and
                current_candle['low'] >= prev_candle['low'])
    
    def is_outside_bar(self, prev_candle: pd.Series, current_candle: pd.Series) -> bool:
        """
        Check if current candle is an outside bar (range engulfs previous candle).
        
        Args:
            prev_candle: Previous candle
            current_candle: Current candle
            
        Returns:
            True if outside bar pattern
        """
        return (current_candle['high'] > prev_candle['high'] and
                current_candle['low'] < prev_candle['low'])
    
    def calculate_candle_metrics(self, candle: pd.Series) -> Dict:
        """
        Calculate various candle metrics for analysis.
        
        Args:
            candle: OHLC candle data
            
        Returns:
            Dictionary with candle metrics
        """
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        body_size = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        total_range = high - low
        
        return {
            'body_size': body_size,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'total_range': total_range,
            'body_pct': body_size / total_range if total_range > 0 else 0,
            'upper_wick_pct': upper_wick / total_range if total_range > 0 else 0,
            'lower_wick_pct': lower_wick / total_range if total_range > 0 else 0,
            'color': self.classify_candle_color(candle)
        }


def main():
    """Example usage and testing of MarketStructureRules."""
    import yaml
    
    # Sample config
    config = {
        'rules': {
            'candle_color_rule': {
                'red': 'close < open',
                'green': 'close > open',
                'doji': 'close == open'
            },
            'min_body_pct': 0.0
        }
    }
    
    rules = MarketStructureRules(config)
    
    # Test candle data
    test_candle_red = pd.Series({
        'open': 2000.0,
        'high': 2005.0,
        'low': 1995.0,
        'close': 1998.0
    })
    
    test_candle_green = pd.Series({
        'open': 1998.0,
        'high': 2010.0,
        'low': 1996.0,
        'close': 2007.0
    })
    
    # Test color classification
    print("Red candle color:", rules.classify_candle_color(test_candle_red))
    print("Green candle color:", rules.classify_candle_color(test_candle_green))
    
    # Test two-candle patterns
    print("Two red confirm:", rules.is_two_red_confirm(test_candle_red, test_candle_red))
    print("Two green confirm:", rules.is_two_green_confirm(test_candle_green, test_candle_green))
    
    print("MarketStructureRules testing complete!")


if __name__ == "__main__":
    main()