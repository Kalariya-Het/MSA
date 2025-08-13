"""
Unit tests for the DataResampler class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import yaml

from src.data.resample import DataResampler


class TestDataResampler:
    """Test cases for DataResampler class."""
    
    @pytest.fixture
    def config_dict(self):
        """Sample configuration for testing."""
        return {
            'data': {
                'base_timeframe': '15min',
                'resampled_timeframes': ['1H', '4H', '1D'],
                'resampled_data_path': 'data/resampled/'
            },
            'output': {
                'logs_path': 'logs/'
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
        os.unlink(temp_path)
    
    @pytest.fixture
    def sample_15m_data(self):
        """Generate 15-minute sample data for testing."""
        # Generate 4 hours of 15-minute data (16 candles)
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        dates = pd.date_range(start_time, periods=16, freq='15T')
        
        data = []
        base_price = 2000.0
        
        for i, dt in enumerate(dates):
            # Create predictable price movements for testing
            open_price = base_price + i
            high = open_price + 5
            low = open_price - 3
            close = open_price + np.random.choice([-2, -1, 0, 1, 2])
            
            data.append({
                'datetime': dt,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000 + i * 100
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, temp_config_file):
        """Test DataResampler initialization."""
        resampler = DataResampler(temp_config_file)
        assert resampler.config is not None
        assert resampler.config['data']['base_timeframe'] == '15min'
        assert '1H' in resampler.config['data']['resampled_timeframes']
    
    def test_get_pandas_frequency(self, temp_config_file):
        """Test pandas frequency conversion."""
        resampler = DataResampler(temp_config_file)
        
        assert resampler._get_pandas_frequency('15min') == '15T'
        assert resampler._get_pandas_frequency('1H') == '1H'
        assert resampler._get_pandas_frequency('4H') == '4H'
        assert resampler._get_pandas_frequency('1D') == '1D'
        assert resampler._get_pandas_frequency('Daily') == '1D'
        
        with pytest.raises(ValueError):
            resampler._get_pandas_frequency('invalid_timeframe')
    
    def test_resample_ohlc_to_1h(self, temp_config_file, sample_15m_data):
        """Test resampling 15m data to 1H."""
        resampler = DataResampler(temp_config_file)
        
        resampled = resampler.resample_ohlc(sample_15m_data, '1H')
        
        # Should have 4 1H candles from 16 15m candles
        assert len(resampled) == 4
        
        # Check column structure
        expected_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert list(resampled.columns) == expected_columns
        
        # Verify OHLC logic for first hour
        first_hour_15m = sample_15m_data.iloc[0:4]  # First 4 15m candles
        first_1h = resampled.iloc[0]
        
        assert first_1h['open'] == first_hour_15m.iloc[0]['open']  # First open
        assert first_1h['high'] == first_hour_15m['high'].max()    # Highest high
        assert first_1h['low'] == first_hour_15m['low'].min()      # Lowest low
        assert first_1h['close'] == first_hour_15m.iloc[-1]['close']  # Last close
        assert first_1h['volume'] == first_hour_15m['volume'].sum()   # Sum volume
    
    def test_resample_ohlc_to_4h(self, temp_config_file, sample_15m_data):
        """Test resampling 15m data to 4H."""
        resampler = DataResampler(temp_config_file)
        
        resampled = resampler.resample_ohlc(sample_15m_data, '4H')
        
        # Should have 1 4H candle from 16 15m candles (4 hours of data)
        assert len(resampled) == 1
        
        # Verify OHLC logic for the 4H candle
        first_4h = resampled.iloc[0]
        
        assert first_4h['open'] == sample_15m_data.iloc[0]['open']    # First open
        assert first_4h['high'] == sample_15m_data['high'].max()      # Highest high
        assert first_4h['low'] == sample_15m_data['low'].min()        # Lowest low
        assert first_4h['close'] == sample_15m_data.iloc[-1]['close'] # Last close
        assert first_4h['volume'] == sample_15m_data['volume'].sum()  # Sum volume
    
    def test_resample_ohlc_missing_datetime_column(self, temp_config_file):
        """Test error handling when datetime column is missing."""
        resampler = DataResampler(temp_config_file)
        
        df_no_datetime = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104]
        })
        
        with pytest.raises(ValueError, match="DataFrame must have 'datetime' column"):
            resampler.resample_ohlc(df_no_datetime, '1H')
    
    def test_validate_resampled_data(self, temp_config_file, sample_15m_data):
        """Test validation of resampled data."""
        resampler = DataResampler(temp_config_file)
        
        resampled = resampler.resample_ohlc(sample_15m_data, '1H')
        validation_report = resampler.validate_resampled_data(
            sample_15m_data, resampled, '1H'
        )
        
        # Check report structure
        assert 'timeframe' in validation_report
        assert 'validation_timestamp' in validation_report
        assert 'original_candles' in validation_report
        assert 'resampled_candles' in validation_report
        assert 'checks' in validation_report
        
        # Check validation results
        assert validation_report['timeframe'] == '1H'
        assert validation_report['original_candles'] == len(sample_15m_data)
        assert validation_report['resampled_candles'] == len(resampled)
        assert validation_report['checks']['ohlc_relationships_valid'] is True
    
    def test_get_expected_frequency_minutes(self, temp_config_file):
        """Test expected frequency calculation."""
        resampler = DataResampler(temp_config_file)
        
        assert resampler._get_expected_frequency_minutes('15min') == 15
        assert resampler._get_expected_frequency_minutes('1H') == 60
        assert resampler._get_expected_frequency_minutes('4H') == 240
        assert resampler._get_expected_frequency_minutes('1D') == 1440
        assert resampler._get_expected_frequency_minutes('Daily') == 1440
        assert resampler._get_expected_frequency_minutes('unknown') == 60  # default
    
    def test_resample_all_timeframes(self, temp_config_file, sample_15m_data):
        """Test resampling to all configured timeframes."""
        resampler = DataResampler(temp_config_file)
        
        resampled_data = resampler.resample_all_timeframes(sample_15m_data)
        
        # Should include base timeframe plus all resampled timeframes
        expected_timeframes = ['15min', '1H', '4H', '1D']
        assert set(resampled_data.keys()) == set(expected_timeframes)
        
        # Check that original data is preserved
        pd.testing.assert_frame_equal(resampled_data['15min'], sample_15m_data)
        
        # Check resampled data exists and has correct structure
        for tf in ['1H', '4H', '1D']:
            assert tf in resampled_data
            assert isinstance(resampled_data[tf], pd.DataFrame)
            assert len(resampled_data[tf]) > 0
            assert 'datetime' in resampled_data[tf].columns
    
    def test_save_resampled_data(self, temp_config_file, sample_15m_data):
        """Test saving resampled data to files."""
        resampler = DataResampler(temp_config_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            resampler.config['data']['resampled_data_path'] = temp_dir
            
            # Create resampled data
            resampled_data = {
                '15min': sample_15m_data,
                '1H': resampler.resample_ohlc(sample_15m_data, '1H')
            }
            
            saved_files = resampler.save_resampled_data(resampled_data, 'test_xauusd')
            
            # Check files were created
            assert '15min' in saved_files
            assert '1H' in saved_files
            
            for tf, filepath in saved_files.items():
                assert os.path.exists(filepath)
                assert filepath.endswith(f'test_xauusd_{tf}.csv')
                
                # Verify file content
                loaded_df = pd.read_csv(filepath)
                assert len(loaded_df) == len(resampled_data[tf])
    
    def test_create_resampling_report(self, temp_config_file, sample_15m_data):
        """Test creation of resampling report."""
        resampler = DataResampler(temp_config_file)
        
        # Create sample resampled data
        resampled_data = {
            '15min': sample_15m_data,
            '1H': resampler.resample_ohlc(sample_15m_data, '1H')
        }
        
        # Create sample validation reports
        validation_reports = [
            {'timeframe': '1H', 'checks': {'ohlc_relationships_valid': True}}
        ]
        
        report = resampler.create_resampling_report(resampled_data, validation_reports)
        
        # Check report structure
        assert 'resampling_timestamp' in report
        assert 'base_timeframe' in report
        assert 'target_timeframes' in report
        assert 'summary' in report
        assert 'validation_results' in report
        
        # Check summary contains data for each timeframe
        assert '15min' in report['summary']
        assert '1H' in report['summary']
        
        for tf_summary in report['summary'].values():
            assert 'total_candles' in tf_summary
            assert 'date_range' in tf_summary
            assert 'price_range' in tf_summary
    
    def test_resample_with_gaps(self, temp_config_file):
        """Test resampling with gaps in data."""
        resampler = DataResampler(temp_config_file)
        
        # Create data with a gap
        dates1 = pd.date_range('2023-01-01 00:00:00', periods=4, freq='15T')
        dates2 = pd.date_range('2023-01-01 02:00:00', periods=4, freq='15T')  # 1 hour gap
        dates = dates1.union(dates2)
        
        data = []
        for i, dt in enumerate(dates):
            data.append({
                'datetime': dt,
                'open': 2000 + i,
                'high': 2005 + i,
                'low': 1995 + i,
                'close': 2002 + i,
                'volume': 1000
            })
        
        df_with_gaps = pd.DataFrame(data)
        resampled = resampler.resample_ohlc(df_with_gaps, '1H')
        
        # Should handle gaps appropriately
        assert len(resampled) >= 1  # At least one complete hour
        
        # Validate OHLC relationships still hold
        validation = resampler.validate_resampled_data(df_with_gaps, resampled, '1H')
        assert validation['checks']['ohlc_relationships_valid'] is True
    
    def test_resample_empty_dataframe(self, temp_config_file):
        """Test resampling empty DataFrame."""
        resampler = DataResampler(temp_config_file)
        
        empty_df = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close'])
        resampled = resampler.resample_ohlc(empty_df, '1H')
        
        # Should return empty DataFrame with correct columns
        assert len(resampled) == 0
        expected_columns = ['datetime', 'open', 'high', 'low', 'close']
        assert list(resampled.columns) == expected_columns
    
    def test_resample_single_candle(self, temp_config_file):
        """Test resampling with single candle."""
        resampler = DataResampler(temp_config_file)
        
        single_candle = pd.DataFrame({
            'datetime': [datetime(2023, 1, 1, 0, 0, 0)],
            'open': [2000.0],
            'high': [2005.0],
            'low': [1995.0],
            'close': [2002.0],
            'volume': [1000]
        })
        
        resampled = resampler.resample_ohlc(single_candle, '1H')
        
        # Should return the same candle
        assert len(resampled) == 1
        assert resampled.iloc[0]['open'] == 2000.0
        assert resampled.iloc[0]['high'] == 2005.0
        assert resampled.iloc[0]['low'] == 1995.0
        assert resampled.iloc[0]['close'] == 2002.0
    
    def test_resample_without_volume(self, temp_config_file):
        """Test resampling data without volume column."""
        resampler = DataResampler(temp_config_file)
        
        # Create data without volume
        dates = pd.date_range('2023-01-01 00:00:00', periods=8, freq='15T')
        data = []
        
        for i, dt in enumerate(dates):
            data.append({
                'datetime': dt,
                'open': 2000 + i,
                'high': 2005 + i,
                'low': 1995 + i,
                'close': 2002 + i
            })
        
        df_no_volume = pd.DataFrame(data)
        resampled = resampler.resample_ohlc(df_no_volume, '1H')
        
        # Should work without volume column
        assert len(resampled) == 2  # 2 hours of data
        assert 'volume' not in resampled.columns
        expected_columns = ['datetime', 'open', 'high', 'low', 'close']
        assert list(resampled.columns) == expected_columns


if __name__ == "_main_":
    pytest.main([__file__, "-v"])