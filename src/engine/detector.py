"""
Core market structure detection engine for XAUUSD.

This module implements the single-pass algorithm to detect:
- HH (Higher High) / LL (Lower Low) 
- HL (Higher Low) / LH (Lower High)
- CHoCH (Change of Character)
- Internal CHoCH

Based on strict candle-based rules with no indicators or ATR.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime
from pathlib import Path
import uuid
import yaml

from .rules import MarketStructureRules
from .state import TrendState, SwingState, MarketEvent, EventType

logger = logging.getLogger(__name__)


class MarketStructureDetector:
    """Core market structure detection engine."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize detector with configuration."""
        self.config = self._load_config(config_path)
        self.rules = MarketStructureRules(self.config)
        self.reset_state()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def reset_state(self):
        """Reset detector state for new analysis."""
        self.trend_state = TrendState()
        self.swing_state = SwingState()
        self.events = []
        self.internal_choch_counters = {
            'lh_after_hl': 0,
            'hl_after_ll': 0
        }
        self.processed_candles = 0
        
        logger.info("Detector state reset")
    
    def detect_market_structure(self, df: pd.DataFrame, timeframe: str = "15min") -> List[MarketEvent]:
        """
        Main entry point for market structure detection.
        
        Args:
            df: DataFrame with OHLC data and datetime column
            timeframe: Timeframe being analyzed
            
        Returns:
            List of detected market structure events
        """
        logger.info(f"Starting market structure detection on {len(df)} candles ({timeframe})")
        
        self.reset_state()
        self.timeframe = timeframe
        
        # Validate input data
        self._validate_input_data(df)
        
        # Single-pass detection algorithm
        for i in range(len(df)):
            self._process_candle(df, i)
        
        # Final cleanup and validation
        self._finalize_detection()
        
        logger.info(f"Detection complete: {len(self.events)} events found")
        return self.events.copy()
    
    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input DataFrame structure."""
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(df) < 2:
            raise ValueError("Need at least 2 candles for detection")
        
        # Check data is sorted by datetime
        if not df['datetime'].is_monotonic_increasing:
            logger.warning("Data not sorted by datetime - sorting now")
            df.sort_values('datetime', inplace=True)
    
    def _process_candle(self, df: pd.DataFrame, i: int):
        """
        Process single candle in the detection algorithm.
        
        Args:
            df: OHLC DataFrame
            i: Current candle index
        """
        current_candle = df.iloc[i]
        self.processed_candles = i + 1
        
        # Step 1: Update candidate extremes
        self._update_candidate_extremes(df, i)
        
        # Step 2: Check for HH/LL confirmations (two-candle patterns)
        if i > 0:  # Need at least 2 candles
            self._check_hh_confirmation(df, i)
            self._check_ll_confirmation(df, i)
        
        # Step 3: Check engulfing patterns for 4H+ timeframes
        if self.timeframe in ['4H', '1D', 'Daily'] and i > 0:
            self._check_engulfing_confirmation(df, i)
        
        # Step 4: Check for HL/LH after confirmed HH/LL
        self._check_hl_after_hh(df, i)
        self._check_lh_after_ll(df, i)
        
        # Step 5: Check for CHoCH after HL/LH
        self._check_choch_after_hl(df, i)
        self._check_choch_after_lh(df, i)
        
        # Step 6: Check for Internal CHoCH
        self._check_internal_choch()
        
        # Debug logging for important candles
        if i < 5 or i % 100 == 0:
            logger.debug(f"Processed candle {i}: trend={self.trend_state.current_trend}, "
                        f"events={len(self.events)}")
    
    def _update_candidate_extremes(self, df: pd.DataFrame, i: int):
        """Update candidate swing highs and lows."""
        current = df.iloc[i]
        
        # Update candidate HH (highest high since last HL)
        if (self.swing_state.candidate_hh_price is None or 
            current['high'] > self.swing_state.candidate_hh_price):
            self.swing_state.candidate_hh_price = current['high']
            self.swing_state.candidate_hh_idx = i
            self.swing_state.candidate_hh_datetime = current['datetime']
        
        # Update candidate LL (lowest low since last LH)
        if (self.swing_state.candidate_ll_price is None or 
            current['low'] < self.swing_state.candidate_ll_price):
            self.swing_state.candidate_ll_price = current['low']
            self.swing_state.candidate_ll_idx = i
            self.swing_state.candidate_ll_datetime = current['datetime']
    
    def _check_hh_confirmation(self, df: pd.DataFrame, i: int):
        """Check for Higher High confirmation via two-candle pattern."""
        if i < 1 or self.swing_state.candidate_hh_idx is None:
            return
        
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        
        # Two consecutive red candles where second close < first low
        if self.rules.is_two_red_confirm(prev_candle, current_candle):
            # Confirm the candidate HH
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.HH,
                datetime=self.swing_state.candidate_hh_datetime,
                price=self.swing_state.candidate_hh_price,
                candle_index=self.swing_state.candidate_hh_idx,
                trigger_index=i,
                trigger_rule="two_red_close_below_low",
                timeframe=self.timeframe,
                notes=f"HH confirmed at {self.swing_state.candidate_hh_price}"
            )
            
            self.events.append(event)
            
            # Update state
            self.trend_state.last_hh_idx = self.swing_state.candidate_hh_idx
            self.trend_state.last_hh_price = self.swing_state.candidate_hh_price
            self.trend_state.last_hh_datetime = self.swing_state.candidate_hh_datetime
            self.trend_state.current_trend = "UP"
            
            # Reset candidate LL after HH confirmation
            self.swing_state.reset_candidate_ll()
            
            logger.info(f"HH confirmed at {self.swing_state.candidate_hh_price} "
                       f"(candle {self.swing_state.candidate_hh_idx})")
    
    def _check_ll_confirmation(self, df: pd.DataFrame, i: int):
        """Check for Lower Low confirmation via two-candle pattern."""
        if i < 1 or self.swing_state.candidate_ll_idx is None:
            return
        
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        
        # Two consecutive green candles where second close > first high
        if self.rules.is_two_green_confirm(prev_candle, current_candle):
            # Confirm the candidate LL
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.LL,
                datetime=self.swing_state.candidate_ll_datetime,
                price=self.swing_state.candidate_ll_price,
                candle_index=self.swing_state.candidate_ll_idx,
                trigger_index=i,
                trigger_rule="two_green_close_above_high",
                timeframe=self.timeframe,
                notes=f"LL confirmed at {self.swing_state.candidate_ll_price}"
            )
            
            self.events.append(event)
            
            # Update state
            self.trend_state.last_ll_idx = self.swing_state.candidate_ll_idx
            self.trend_state.last_ll_price = self.swing_state.candidate_ll_price
            self.trend_state.last_ll_datetime = self.swing_state.candidate_ll_datetime
            self.trend_state.current_trend = "DOWN"
            
            # Reset candidate HH after LL confirmation
            self.swing_state.reset_candidate_hh()
            
            logger.info(f"LL confirmed at {self.swing_state.candidate_ll_price} "
                       f"(candle {self.swing_state.candidate_ll_idx})")
    
    def _check_engulfing_confirmation(self, df: pd.DataFrame, i: int):
        """Check for engulfing pattern confirmation on 4H+ timeframes."""
        if i < 1:
            return
        
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        
        # Check for red engulfing green (HH confirmation)
        if (self.swing_state.candidate_hh_idx is not None and 
            self.rules.is_red_engulfing_green(prev_candle, current_candle)):
            
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.HH,
                datetime=self.swing_state.candidate_hh_datetime,
                price=self.swing_state.candidate_hh_price,
                candle_index=self.swing_state.candidate_hh_idx,
                trigger_index=i,
                trigger_rule="red_engulfing_green_4h",
                timeframe=self.timeframe,
                notes=f"HH confirmed via engulfing at {self.swing_state.candidate_hh_price}"
            )
            
            self.events.append(event)
            self._update_state_after_hh_confirmation()
        
        # Check for green engulfing red (LL confirmation)
        elif (self.swing_state.candidate_ll_idx is not None and 
              self.rules.is_green_engulfing_red(prev_candle, current_candle)):
            
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.LL,
                datetime=self.swing_state.candidate_ll_datetime,
                price=self.swing_state.candidate_ll_price,
                candle_index=self.swing_state.candidate_ll_idx,
                trigger_index=i,
                trigger_rule="green_engulfing_red_4h",
                timeframe=self.timeframe,
                notes=f"LL confirmed via engulfing at {self.swing_state.candidate_ll_price}"
            )
            
            self.events.append(event)
            self._update_state_after_ll_confirmation()
    
    def _check_hl_after_hh(self, df: pd.DataFrame, i: int):
        """Check for HL formation after HH confirmation."""
        if (self.trend_state.last_hh_idx is None or 
            self.trend_state.last_hl_idx is not None and 
            self.trend_state.last_hl_idx > self.trend_state.last_hh_idx):
            return
        
        current_candle = df.iloc[i]
        
        # Check if current candle closes above the last HH
        if current_candle['close'] > self.trend_state.last_hh_price:
            # Find the lowest low between HH and current candle
            start_idx = self.trend_state.last_hh_idx + 1
            end_idx = i
            
            if start_idx <= end_idx:
                hl_info = self.rules.find_lowest_low_between(df, start_idx, end_idx)
                
                if hl_info:
                    event = MarketEvent(
                        id=str(uuid.uuid4()),
                        event_type=EventType.HL,
                        datetime=hl_info['datetime'],
                        price=hl_info['price'],
                        candle_index=hl_info['index'],
                        trigger_index=i,
                        trigger_rule="close_above_hh_triggers_hl",
                        timeframe=self.timeframe,
                        notes=f"HL found at {hl_info['price']} between HH and close above"
                    )
                    
                    self.events.append(event)
                    
                    # Update state
                    self.trend_state.last_hl_idx = hl_info['index']
                    self.trend_state.last_hl_price = hl_info['price']
                    self.trend_state.last_hl_datetime = hl_info['datetime']
                    
                    # Reset internal CHoCH counter
                    self.internal_choch_counters['lh_after_hl'] = 0
                    
                    logger.info(f"HL found at {hl_info['price']} (candle {hl_info['index']})")
    
    def _check_lh_after_ll(self, df: pd.DataFrame, i: int):
        """Check for LH formation after LL confirmation."""
        if (self.trend_state.last_ll_idx is None or 
            self.trend_state.last_lh_idx is not None and 
            self.trend_state.last_lh_idx > self.trend_state.last_ll_idx):
            return
        
        current_candle = df.iloc[i]
        
        # Check if current candle closes below the last LL
        if current_candle['close'] < self.trend_state.last_ll_price:
            # Find the highest high between LL and current candle
            start_idx = self.trend_state.last_ll_idx + 1
            end_idx = i
            
            if start_idx <= end_idx:
                lh_info = self.rules.find_highest_high_between(df, start_idx, end_idx)
                
                if lh_info:
                    event = MarketEvent(
                        id=str(uuid.uuid4()),
                        event_type=EventType.LH,
                        datetime=lh_info['datetime'],
                        price=lh_info['price'],
                        candle_index=lh_info['index'],
                        trigger_index=i,
                        trigger_rule="close_below_ll_triggers_lh",
                        timeframe=self.timeframe,
                        notes=f"LH found at {lh_info['price']} between LL and close below"
                    )
                    
                    self.events.append(event)
                    
                    # Update state
                    self.trend_state.last_lh_idx = lh_info['index']
                    self.trend_state.last_lh_price = lh_info['price']
                    self.trend_state.last_lh_datetime = lh_info['datetime']
                    
                    # Reset internal CHoCH counter
                    self.internal_choch_counters['hl_after_ll'] = 0
                    
                    logger.info(f"LH found at {lh_info['price']} (candle {lh_info['index']})")
    
    def _check_choch_after_hl(self, df: pd.DataFrame, i: int):
        """Check for CHoCH Up→Down after HL confirmation."""
        if self.trend_state.last_hl_idx is None or i < 1:
            return
        
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        
        # Two consecutive closes below HL with first candle being red
        if (prev_candle['close'] < self.trend_state.last_hl_price and
            current_candle['close'] < self.trend_state.last_hl_price and
            self.rules.classify_candle_color(prev_candle) == 'red'):
            
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.CHOCH_UP_DOWN,
                datetime=prev_candle['datetime'],  # First of the two candles
                price=self.trend_state.last_hl_price,
                candle_index=i-1,
                trigger_index=i,
                trigger_rule="two_closes_below_hl_first_red",
                timeframe=self.timeframe,
                notes=f"CHoCH Up→Down at HL {self.trend_state.last_hl_price}"
            )
            
            self.events.append(event)
            self.trend_state.current_trend = "DOWN"
            
            logger.info(f"CHoCH Up→Down detected at HL {self.trend_state.last_hl_price}")
    
    def _check_choch_after_lh(self, df: pd.DataFrame, i: int):
        """Check for CHoCH Down→Up after LH confirmation."""
        if self.trend_state.last_lh_idx is None or i < 1:
            return
        
        prev_candle = df.iloc[i-1]
        current_candle = df.iloc[i]
        
        # Two consecutive closes above LH with first candle being green
        if (prev_candle['close'] > self.trend_state.last_lh_price and
            current_candle['close'] > self.trend_state.last_lh_price and
            self.rules.classify_candle_color(prev_candle) == 'green'):
            
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.CHOCH_DOWN_UP,
                datetime=prev_candle['datetime'],  # First of the two candles
                price=self.trend_state.last_lh_price,
                candle_index=i-1,
                trigger_index=i,
                trigger_rule="two_closes_above_lh_first_green",
                timeframe=self.timeframe,
                notes=f"CHoCH Down→Up at LH {self.trend_state.last_lh_price}"
            )
            
            self.events.append(event)
            self.trend_state.current_trend = "UP"
            
            logger.info(f"CHoCH Down→Up detected at LH {self.trend_state.last_lh_price}")
    
    def _check_internal_choch(self):
        """Check for Internal CHoCH conditions."""
        threshold = self.config['rules']['internal_choch_threshold']
        
        # Count LHs after most recent HL
        if self.trend_state.last_hl_idx is not None:
            lh_count = self._count_lhs_since_hl()
            if lh_count >= threshold:
                self._trigger_internal_choch_up_down()
        
        # Count HLs after most recent LL  
        if self.trend_state.last_ll_idx is not None:
            hl_count = self._count_hls_since_ll()
            if hl_count >= threshold:
                self._trigger_internal_choch_down_up()
    
    def _count_lhs_since_hl(self) -> int:
        """Count LH events since last HL."""
        if self.trend_state.last_hl_idx is None:
            return 0
        
        count = 0
        for event in self.events:
            if (event.event_type == EventType.LH and 
                event.candle_index > self.trend_state.last_hl_idx):
                count += 1
        
        return count
    
    def _count_hls_since_ll(self) -> int:
        """Count HL events since last LL."""
        if self.trend_state.last_ll_idx is None:
            return 0
        
        count = 0
        for event in self.events:
            if (event.event_type == EventType.HL and 
                event.candle_index > self.trend_state.last_ll_idx):
                count += 1
        
        return count
    
    def _trigger_internal_choch_up_down(self):
        """Trigger Internal CHoCH Up→Down."""
        # Avoid duplicate internal CHoCH events
        recent_internal = any(
            event.event_type == EventType.INTERNAL_CHOCH_UP_DOWN and
            event.candle_index > (self.trend_state.last_hl_idx or 0)
            for event in self.events[-10:]  # Check last 10 events
        )
        
        if not recent_internal:
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.INTERNAL_CHOCH_UP_DOWN,
                datetime=self.trend_state.last_hl_datetime,
                price=self.trend_state.last_hl_price,
                candle_index=self.processed_candles - 1,
                trigger_index=self.processed_candles - 1,
                trigger_rule="three_lhs_without_new_hl",
                timeframe=self.timeframe,
                notes="Internal CHoCH Up→Down: 3+ LHs without new HL"
            )
            
            self.events.append(event)
            logger.info("Internal CHoCH Up→Down triggered")
    
    def _trigger_internal_choch_down_up(self):
        """Trigger Internal CHoCH Down→Up."""
        # Avoid duplicate internal CHoCH events
        recent_internal = any(
            event.event_type == EventType.INTERNAL_CHOCH_DOWN_UP and
            event.candle_index > (self.trend_state.last_ll_idx or 0)
            for event in self.events[-10:]  # Check last 10 events
        )
        
        if not recent_internal:
            event = MarketEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.INTERNAL_CHOCH_DOWN_UP,
                datetime=self.trend_state.last_ll_datetime,
                price=self.trend_state.last_ll_price,
                candle_index=self.processed_candles - 1,
                trigger_index=self.processed_candles - 1,
                trigger_rule="three_hls_without_new_lh",
                timeframe=self.timeframe,
                notes="Internal CHoCH Down→Up: 3+ HLs without new LH"
            )
            
            self.events.append(event)
            logger.info("Internal CHoCH Down→Up triggered")
    
    def _update_state_after_hh_confirmation(self):
        """Update state after HH confirmation."""
        self.trend_state.last_hh_idx = self.swing_state.candidate_hh_idx
        self.trend_state.last_hh_price = self.swing_state.candidate_hh_price
        self.trend_state.last_hh_datetime = self.swing_state.candidate_hh_datetime
        self.trend_state.current_trend = "UP"
        self.swing_state.reset_candidate_ll()
    
    def _update_state_after_ll_confirmation(self):
        """Update state after LL confirmation."""
        self.trend_state.last_ll_idx = self.swing_state.candidate_ll_idx
        self.trend_state.last_ll_price = self.swing_state.candidate_ll_price
        self.trend_state.last_ll_datetime = self.swing_state.candidate_ll_datetime
        self.trend_state.current_trend = "DOWN"
        self.swing_state.reset_candidate_hh()
    
    def _finalize_detection(self):
        """Final cleanup and validation of detected events."""
        # Sort events by candle index
        self.events.sort(key=lambda x: x.candle_index)
        
        # Remove any duplicate events (safety check)
        seen_events = set()
        unique_events = []
        
        for event in self.events:
            event_key = (event.event_type, event.candle_index, event.price)
            if event_key not in seen_events:
                seen_events.add(event_key)
                unique_events.append(event)
        
        self.events = unique_events
        
        logger.info(f"Finalized {len(self.events)} unique events")
    
    def export_events_to_dataframe(self) -> pd.DataFrame:
        """Export detected events to DataFrame."""
        if not self.events:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'id', 'datetime', 'timeframe', 'event_type', 'price',
                'candle_index', 'trigger_index', 'trigger_rule', 'notes'
            ])
        
        events_data = []
        for event in self.events:
            events_data.append({
                'id': event.id,
                'datetime': event.datetime,
                'timeframe': event.timeframe,
                'event_type': event.event_type.value,
                'price': event.price,
                'candle_index': event.candle_index,
                'trigger_index': event.trigger_index,
                'trigger_rule': event.trigger_rule,
                'notes': event.notes
            })
        
        return pd.DataFrame(events_data)
    
    def save_events_to_csv(self, filepath: str) -> str:
        """Save events to CSV file."""
        events_df = self.export_events_to_dataframe()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        events_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(events_df)} events to {filepath}")
        return filepath
    
    def get_detection_summary(self) -> Dict:
        """Get summary statistics of detection results."""
        event_counts = {}
        for event_type in EventType:
            count = sum(1 for event in self.events if event.event_type == event_type)
            event_counts[event_type.value] = count
        
        return {
            'total_events': len(self.events),
            'processed_candles': self.processed_candles,
            'current_trend': self.trend_state.current_trend,
            'event_breakdown': event_counts,
            'last_hh': {
                'price': self.trend_state.last_hh_price,
                'index': self.trend_state.last_hh_idx,
                'datetime': self.trend_state.last_hh_datetime.isoformat() if self.trend_state.last_hh_datetime else None
            } if self.trend_state.last_hh_price else None,
            'last_ll': {
                'price': self.trend_state.last_ll_price,
                'index': self.trend_state.last_ll_idx,
                'datetime': self.trend_state.last_ll_datetime.isoformat() if self.trend_state.last_ll_datetime else None
            } if self.trend_state.last_ll_price else None
        }


def main():
    """Example usage of MarketStructureDetector."""
    from src.data.loader import DataLoader
    
    # Initialize components
    detector = MarketStructureDetector()
    loader = DataLoader()
    
    print("MarketStructureDetector initialized successfully")
    print("Ready for Phase 2 detection!")
    

if __name__ == "__main__":
    main()