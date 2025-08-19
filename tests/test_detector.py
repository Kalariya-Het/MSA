"""
Comprehensive unit tests for the Market Structure Detection Engine.

Tests cover:
- Core detection logic (HH/HL/LL/LH)
- Two-candle confirmation patterns
- Engulfing patterns for 4H+ timeframes
- CHoCH detection
- Internal CHoCH detection
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import yaml
import uuid

from src.engine.detector import MarketStructureDetector
from src.engine.rules import MarketStructureRules
from src.engine.state import MarketEvent, EventType, TrendState, SwingState


class TestMarketStructureDetector:
    """Test cases for MarketStructureDetector."""
    
    @pytest.fixture
    def config_dict(self):
        """Sample configuration for testing."""
        return {
            'rules': {
                'candle_color_rule': {
                    'red': 'close < open',
                    'green': 'close > open',
                    'doji': 'close == open'
                },
                'min_body_pct': 0.0,
                'two_candle_confirmation': True,
                'engulfing_4h_plus': True,
                'choch_consecutive_candles': 2,
                'internal_choch_threshold': 3
            },
            'detection': {
                'detect_hh': True,
                'detect_hl': True,
                'detect_ll': True,
                'detect_lh': True,
                'detect_choch': True,
                'detect_internal_choch': True
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, config_dict):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        import os
        os.unlink(temp_path)
    
    @pytest.fixture
    def simple_uptrend_data(self):
        """Create simple uptrending data for HH/HL testing."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=20, freq='15T')
        
        # Create clear uptrend with HH and HL patterns
        data = []
        base_price = 2000.0
        
        for i, dt in enumerate(dates):
            # Rising trend with volatility
            trend_price = base_price + (i * 2)  # +2 per candle
            
            if i < 5:  # Initial rise
                open_p = trend_price + np.random.uniform(-1, 1)
                close_p = open_p + np.random.uniform(0.5, 2)  # Green candles
                high = max(open_p, close_p) + np.random.uniform(0.5, 2)
                low = min(open_p, close_p) - np.random.uniform(0.2, 1)
            elif i == 5 or i == 6:  # Two red candles for HH confirmation
                open_p = trend_price + 3
                close_p = open_p - np.random.uniform(2, 4)  # Red candles
                high = open_p + np.random.uniform(0.5, 1)
                if i == 6:  # Second red candle closes below first's low
                    close_p = data[i-1]['low'] - 0.5
                low = min(open_p, close_p) - np.random.uniform(0.2, 1)
            elif 7 <= i <= 10:  # Pullback for HL formation
                open_p = trend_price - np.random.uniform(1, 3)
                close_p = open_p + np.random.uniform(-2, 1)
                high = max(open_p, close_p) + np.random.uniform(0.2, 1)
                low = min(open_p, close_p) - np.random.uniform(0.5, 2)
            elif i == 11:  # Close above HH to trigger HL
                open_p = trend_price
                close_p = data[4]['high'] + 1  # Close above the HH
                high = close_p + 1
                low = open_p - 1
            else:  # Continue trend
                open_p = trend_price + np.random.uniform(-1, 1)
                close_p = open_p + np.random.uniform(0, 2)
                high = max(open_p, close_p) + np.random.uniform(0.5, 1.5)
                low = min(open_p, close_p) - np.random.uniform(0.2, 1)
            
            data.append({
                'datetime': dt,
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_p, 2),
                'volume': 1000 + i * 50
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def simple_downtrend_data(self):
        """Create simple downtrending data for LL/LH testing."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=20, freq='15T')
        
        data = []
        base_price = 2100.0
        
        for i, dt in enumerate(dates):
            # Falling trend
            trend_price = base_price - (i * 2)  # -2 per candle
            
            if i < 5:  # Initial fall
                open_p = trend_price + np.random.uniform(-1, 1)
                close_p = open_p - np.random.uniform(0.5, 2)  # Red candles
                high = max(open_p, close_p) + np.random.uniform(0.2, 1)
                low = min(open_p, close_p) - np.random.uniform(0.5, 2)
            elif i == 5 or i == 6:  # Two green candles for LL confirmation
                open_p = trend_price - 3
                close_p = open_p + np.random.uniform(2, 4)  # Green candles
                low = open_p - np.random.uniform(0.5, 1)
                if i == 6:  # Second green candle closes above first's high
                    close_p = data[i-1]['high'] + 0.5
                high = max(open_p, close_p) + np.random.uniform(0.2, 1)
            elif 7 <= i <= 10:  # Bounce for LH formation
                open_p = trend_price + np.random.uniform(1, 3)
                close_p = open_p + np.random.uniform(-1, 2)
                high = max(open_p, close_p) + np.random.uniform(0.5, 2)
                low = min(open_p, close_p) - np.random.uniform(0.2, 1)
            elif i == 11:  # Close below LL to trigger LH
                open_p = trend_price
                close_p = data[4]['low'] - 1  # Close below the LL
                high = open_p + 1
                low = close_p - 1
            else:  # Continue trend
                open_p = trend_price + np.random.uniform(-1, 1)
                close_p = open_p - np.random.uniform(0, 2)
                high = max(open_p, close_p) + np.random.uniform(0.2, 1)
                low = min(open_p, close_p) - np.random.uniform(0.5, 1.5)
            
            data.append({
                'datetime': dt,
                'open': round(open_p, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_p, 2),
                'volume': 1000 + i * 50
            })
        
        return pd.DataFrame(data)
    
    def test_detector_initialization(self, temp_config_file):
        """Test detector initialization."""
        detector = MarketStructureDetector(temp_config_file)
        
        assert detector.config is not None
        assert isinstance(detector.rules, MarketStructureRules)
        assert detector.trend_state is not None
        assert detector.swing_state is not None
        assert detector.events == []
    
    def test_reset_state(self, temp_config_file):
        """Test state reset functionality."""
        detector = MarketStructureDetector(temp_config_file)
        
        # Add some dummy state
        detector.trend_state.current_trend = "UP"
        detector.swing_state.candidate_hh_price = 2000.0
        detector.events.append(MarketEvent(
            id=str(uuid.uuid4()),
            event_type=EventType.HH,
            datetime=datetime.now(),
            price=2000.0,
            candle_index=5,
            trigger_index=7,
            trigger_rule="test",
            timeframe="15min"
        ))
        
        # Reset and verify
        detector.reset_state()
        
        assert detector.trend_state.current_trend == "UNKNOWN"
        assert detector.swing_state.candidate_hh_price is None
        assert detector.events == []
        assert detector.processed_candles == 0
    
    def test_validate_input_data_valid(self, temp_config_file, simple_uptrend_data):
        """Test input data validation with valid data."""
        detector = MarketStructureDetector(temp_config_file)
        
        # Should not raise any exceptions
        detector._validate_input_data(simple_uptrend_data)
    
    def test_validate_input_data_missing_columns(self, temp_config_file):
        """Test input data validation with missing columns."""
        detector = MarketStructureDetector(temp_config_file)
        
        invalid_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5, freq='H'),
            'open': [100, 101, 102, 103, 104],
            # Missing 'high', 'low', 'close'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            detector._validate_input_data(invalid_df)
    
    def test_validate_input_data_insufficient_candles(self, temp_config_file):
        """Test input data validation with insufficient candles."""
        detector = MarketStructureDetector(temp_config_file)
        
        insufficient_df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0]
        })
        
        with pytest.raises(ValueError, match="Need at least 2 candles"):
            detector._validate_input_data(insufficient_df)
    
    def test_hh_detection_uptrend(self, temp_config_file, simple_uptrend_data):
        """Test HH detection in uptrend."""
        detector = MarketStructureDetector(temp_config_file)
        events = detector.detect_market_structure(simple_uptrend_data, "15min")
        
        # Should detect at least one HH
        hh_events = [e for e in events if e.event_type == EventType.HH]
        assert len(hh_events) > 0
        
        # Verify HH event properties
        for hh_event in hh_events:
            assert hh_event.price > 0
            assert hh_event.candle_index >= 0
            assert hh_event.trigger_index >= hh_event.candle_index
            assert hh_event.trigger_rule in ["two_red_close_below_low", "red_engulfing_green_4h"]
            assert hh_event.timeframe == "15min"
    
    def test_ll_detection_downtrend(self, temp_config_file, simple_downtrend_data):
        """Test LL detection in downtrend."""
        detector = MarketStructureDetector(temp_config_file)
        events = detector.detect_market_structure(simple_downtrend_data, "15min")
        
        # Should detect at least one LL
        ll_events = [e for e in events if e.event_type == EventType.LL]
        assert len(ll_events) > 0
        
        # Verify LL event properties
        for ll_event in ll_events:
            assert ll_event.price > 0
            assert ll_event.candle_index >= 0
            assert ll_event.trigger_index >= ll_event.candle_index
            assert ll_event.trigger_rule in ["two_green_close_above_high", "green_engulfing_red_4h"]
            assert ll_event.timeframe == "15min"
    
    def test_hl_detection_after_hh(self, temp_config_file, simple_uptrend_data):
        """Test HL detection after HH confirmation."""
        detector = MarketStructureDetector(temp_config_file)
        events = detector.detect_market_structure(simple_uptrend_data, "15min")
        
        # Should detect HH first, then HL
        hh_events = [e for e in events if e.event_type == EventType.HH]
        hl_events = [e for e in events if e.event_type == EventType.HL]
        
        assert len(hh_events) > 0
        assert len(hl_events) > 0
        
        # HL should come after HH
        first_hh = min(hh_events, key=lambda x: x.candle_index)
        first_hl = min(hl_events, key=lambda x: x.candle_index)
        assert first_hl.candle_index > first_hh.candle_index
    
    def test_lh_detection_after_ll(self, temp_config_file, simple_downtrend_data):
        """Test LH detection after LL confirmation."""
        detector = MarketStructureDetector(temp_config_file)
        events = detector.detect_market_structure(simple_downtrend_data, "15min")
        
        # Should detect LL first, then LH
        ll_events = [e for e in events if e.event_type == EventType.LL]
        lh_events = [e for e in events if e.event_type == EventType.LH]
        
        assert len(ll_events) > 0
        assert len(lh_events) > 0
        
        # LH should come after LL
        first_ll = min(ll_events, key=lambda x: x.candle_index)
        first_lh = min(lh_events, key=lambda x: x.candle_index)
        assert first_lh.candle_index > first_ll.candle_index
    
    def test_engulfing_confirmation_4h(self, temp_config_file):
        """Test engulfing pattern confirmation on 4H timeframe."""
        detector = MarketStructureDetector(temp_config_file)
        
        # Create data with clear engulfing pattern
        dates = pd.date_range('2023-01-01 00:00:00', periods=10, freq='4H')
        data = []
        
        for i, dt in enumerate(dates):
            if i == 0:  # Green candle
                data.append({
                    'datetime': dt, 'open': 2000.0, 'high': 2010.0,
                    'low': 1995.0, 'close': 2008.0, 'volume': 1000
                })
            elif i == 1:  # Red engulfing green
                data.append({
                    'datetime': dt, 'open': 2012.0, 'high': 2015.0,
                    'low': 1990.0, 'close': 1992.0, 'volume': 1500
                })
            else:  # Regular candles
                price = 2000 + i * 2
                data.append({
                    'datetime': dt, 'open': price, 'high': price + 5,
                    'low': price - 3, 'close': price + 1, 'volume': 1000
                })