"""
Unit tests for the DataLoader class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import yaml

from src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def config_dict(self):
        """Sample configuration for testing."""
        return {
            'data': {
                'timezone': 'UTC',
                'drop_duplicates': True,
                'fill_missing': 'forward',
                'validate_ohlc': True,
                'clean_data_path': 'data/clean/',
                'raw_data_path': 'data/raw/'
            },
            'output': {
                'logs_path': 'logs/'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    def sample_ohlc_data(self):
        """Generate sample OHLC data for testing."""
        dates = pd.date_range('2023-01-01 00:00:00', periods=100, freq='15T')
        
        data = []
        price = 2000.0
        
        for dt in dates:
            # Simple price evolution
            price += np.random.normal(0, 5)
            open_price = price + np.random.normal(0, 1)
            close_price = price + np.random.normal(0, 1)
            high = max(open_price, close_price) + abs(np.random.normal(0, 2))
            low = min(open_price, close_price) - abs(np.random.normal(0, 2))
            
            data.append({
                'datetime': dt,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(1000, 5000)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_ohlc_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlc_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_initialization(self, temp_config_file):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_config_file)
        assert loader.config is not None
        assert loader.config['data']['timezone'] == 'UTC'
    
    def test_load_csv_basic(self, temp_config_file, temp_csv_file):
        """Test basic CSV loading functionality."""
        loader = DataLoader(temp_config_file)
        df = loader.load_csv(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'datetime' in df.columns
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close'])
    
    def test_standardize_columns(self, temp_config_file):
        """Test column name standardization."""
        loader = DataLoader(temp_config_file)
        
        # Create DataFrame with mixed case columns
        df = pd.DataFrame({
            'DateTime': pd.date_range('2023-01-01', periods=3, freq='H'),
            'OPEN': [100, 101, 102],
            'High': [105, 106, 107],
            'low': [95, 96, 97],
            'Close': [102, 103, 104]
        })
        
        standardized = loader._standardize_columns(df, 'DateTime')
        
        assert 'datetime' in standardized.columns
        assert 'open' in standardized.columns
        assert 'high' in standardized.columns
        assert 'low' in standardized.columns
        assert 'close' in standardized.columns
    
    def test_remove_duplicates(self, temp_config_file):
        """Test duplicate removal functionality."""
        loader = DataLoader(temp_config_file)
        
        # Create DataFrame with duplicates
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:00', '2023-01-01 01:00:00']),
            'open': [100, 100, 101],
            'high': [105, 105, 106],
            'low': [95, 95, 96],
            'close': [102, 102, 103]
        })
        
        cleaned = loader._remove_duplicates(df)
        
        # Should keep only unique datetimes (last occurrence)
        assert len(cleaned) == 2
        assert cleaned['datetime'].nunique() == 2
    
    def test_validate_ohlc_valid_data(self, temp_config_file):
        """Test OHLC validation with valid data."""
        loader = DataLoader(temp_config_file)
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=3, freq='H'),
            'open': [100.0, 102.0, 104.0],
            'high': [105.0, 107.0, 109.0],
            'low': [95.0, 97.0, 99.0],
            'close': [102.0, 104.0, 106.0]
        })
        
        validated = loader._validate_ohlc(df)
        
        # All rows should remain as they're valid
        assert len(validated) == 3
    
    def test_validate_ohlc_invalid_data(self, temp_config_file):
        """Test OHLC validation with invalid data."""
        loader = DataLoader(temp_config_file)
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=4, freq='H'),
            'open': [100.0, 102.0, 104.0, 106.0],
            'high': [105.0, 101.0, 109.0, 111.0],  # Second row: high < open (invalid)
            'low': [95.0, 97.0, 110.0, 99.0],     # Third row: low > high (invalid)
            'close': [102.0, 104.0, 106.0, 108.0]
        })
        
        validated = loader._validate_ohlc(df)
        
        # Should remove invalid rows
        assert len(validated) == 2  # Only first and last rows should remain
    
    def test_handle_missing_data_forward_fill(self, temp_config_file):
        """Test missing data handling with forward fill."""
        loader = DataLoader(temp_config_file)
        loader.config['data']['fill_missing'] = 'forward'
        
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=4, freq='H'),
            'open': [100.0, np.nan, 104.0, 106.0],
            'high': [105.0, np.nan, 109.0, 111.0],
            'low': [95.0, np.nan, 99.0, 101.0],
            'close': [102.0, np.nan, 106.0, 108.0]
        })
        
        filled = loader._handle_missing_data(df)
        
        # Should have no missing values
        assert not filled.isnull().any().any()
        # Second row should have values from first row
        assert filled.iloc[1]['open'] == 100.0
    
    def test_generate_data_report(self, temp_config_file, sample_ohlc_data):
        """Test data report generation."""
        loader = DataLoader(temp_config_file)
        
        report = loader.generate_data_report(sample_ohlc_data, "test_file.csv")
        
        assert 'source_file' in report
        assert 'processing_timestamp' in report
        assert 'total_rows' in report
        assert 'date_range' in report
        assert 'data_quality' in report
        assert 'candle_analysis' in report
        
        assert report['total_rows'] == len(sample_ohlc_data)
        assert report['source_file'] == "test_file.csv"
    
    def test_normalize_datetime_utc(self, temp_config_file):
        """Test datetime normalization to UTC."""
        loader = DataLoader(temp_config_file)
        
        df = pd.DataFrame({
            'datetime': ['2023-01-01 12:00:00', '2023-01-01 13:00:00'],
            'open': [100.0, 102.0],
            'high': [105.0, 107.0],
            'low': [95.0, 97.0],
            'close': [102.0, 104.0]
        })
        
        normalized = loader._normalize_datetime(df)
        
        # Should have datetime column with timezone info
        assert pd.api.types.is_datetime64_any_dtype(normalized['datetime'])
        # Should be timezone aware
        assert normalized['datetime'].dt.tz is not None
    
    def test_invalid_csv_file(self, temp_config_file):
        """Test handling of invalid CSV file."""
        loader = DataLoader(temp_config_file)
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv("non_existent_file.csv")
    
    def test_missing_required_columns(self, temp_config_file):
        """Test handling of CSV with missing required columns."""
        loader = DataLoader(temp_config_file)
        
        # Create CSV with missing columns
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=3, freq='H'),
            'open': [100, 101, 102],
            # Missing 'high', 'low', 'close'
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                loader.load_csv(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_clean_data(self, temp_config_file, sample_ohlc_data):
        """Test saving cleaned data."""
        loader = DataLoader(temp_config_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            loader.config['data']['clean_data_path'] = temp_dir
            
            saved_path = loader.save_clean_data(sample_ohlc_data, "test_clean.csv")
            
            # Check file was created
            assert os.path.exists(saved_path)
            
            # Load and verify content
            loaded_df = pd.read_csv(saved_path)
            assert len(loaded_df) == len(sample_ohlc_data)
            assert list(loaded_df.columns) == list(sample_ohlc_data.columns)


if __name__ == "_main_":
    pytest.main([__file__, "-v"])