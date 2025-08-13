"""
Data loading and preprocessing module for XAUUSD market structure detection.

This module handles:
- CSV data loading and validation
- Timezone normalization
- Duplicate removal
- OHLC data validation
- Missing data handling
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of OHLC data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataLoader with configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def load_csv(self, file_path: str, datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Load CSV data with proper datetime parsing and validation.
        
        Args:
            file_path: Path to CSV file
            datetime_column: Name of datetime column
            
        Returns:
            Cleaned DataFrame with datetime index
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col.lower() not in df.columns.str.lower()]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Standardize column names
            df = self._standardize_columns(df, datetime_column)
            
            # Parse datetime and set timezone
            df = self._normalize_datetime(df)
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Remove duplicates
            if self.config['data']['drop_duplicates']:
                df = self._remove_duplicates(df)
            
            # Validate OHLC data
            if self.config['data']['validate_ohlc']:
                df = self._validate_ohlc(df)
            
            # Handle missing data
            df = self._handle_missing_data(df)
            
            logger.info(f"Cleaned data: {len(df)} rows remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        # Create mapping for case-insensitive column matching
        column_mapping = {}
        for col in df.columns:
            if col.lower() == datetime_column.lower():
                column_mapping[col] = 'datetime'
            elif col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                column_mapping[col] = col.lower()
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _normalize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime column to specified timezone."""
        target_tz = self.config['data']['timezone']
        
        try:
            # Try to parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            
            # Convert to target timezone
            if target_tz.upper() != 'UTC':
                target_timezone = pytz.timezone(target_tz)
                df['datetime'] = df['datetime'].dt.tz_convert(target_timezone)
            
            logger.info(f"Normalized datetime to {target_tz} timezone")
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing datetime: {e}")
            # Try alternative parsing methods
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
                df['datetime'] = df['datetime'].dt.tz_localize('UTC')
                logger.warning("Used fallback datetime parsing")
                return df
            except Exception as e2:
                logger.error(f"Fallback datetime parsing failed: {e2}")
                raise
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on datetime."""
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on datetime (keep last)
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        
        return df
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data integrity."""
        initial_count = len(df)
        
        # Check for invalid OHLC relationships
        valid_mask = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['high'] >= df['low']) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0)
        )
        
        invalid_rows = ~valid_mask
        if invalid_rows.sum() > 0:
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC data")
            # Log some examples
            invalid_examples = df[invalid_rows].head(3)
            for idx, row in invalid_examples.iterrows():
                logger.warning(f"Invalid OHLC at {row['datetime']}: "
                             f"O={row['open']}, H={row['high']}, "
                             f"L={row['low']}, C={row['close']}")
        
        # Remove invalid rows
        df = df[valid_mask].copy()
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid OHLC rows")
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data according to configuration."""
        fill_method = self.config['data']['fill_missing']
        
        if df.isnull().any().any():
            logger.info("Handling missing data")
            
            if fill_method == 'forward':
                df = df.fillna(method='ffill')
            elif fill_method == 'backward':
                df = df.fillna(method='bfill')
            elif fill_method == 'interpolate':
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].interpolate()
            elif fill_method == 'drop':
                df = df.dropna()
            
            # Drop any remaining NaNs
            df = df.dropna()
            
        return df
    
    def save_clean_data(self, df: pd.DataFrame, filename: str) -> str:
        """Save cleaned data to CSV."""
        clean_path = Path(self.config['data']['clean_data_path'])
        clean_path.mkdir(parents=True, exist_ok=True)
        
        output_file = clean_path / filename
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved cleaned data to {output_file}")
        return str(output_file)
    
    def generate_data_report(self, df: pd.DataFrame, source_file: str) -> Dict:
        """Generate data integrity report."""
        report = {
            'source_file': source_file,
            'processing_timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat(),
                'duration_days': (df['datetime'].max() - df['datetime'].min()).days
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_datetimes': df.duplicated(subset=['datetime']).sum(),
                'price_statistics': {
                    'min_price': df[['open', 'high', 'low', 'close']].min().min(),
                    'max_price': df[['open', 'high', 'low', 'close']].max().max(),
                    'avg_high_low_spread': (df['high'] - df['low']).mean()
                }
            },
            'candle_analysis': {
                'green_candles': (df['close'] > df['open']).sum(),
                'red_candles': (df['close'] < df['open']).sum(),
                'doji_candles': (df['close'] == df['open']).sum()
            }
        }
        
        return report
    
    def save_report(self, report: Dict, filename: str = "data_integrity_report.json"):
        """Save data integrity report."""
        import json
        
        logs_path = Path(self.config['output']['logs_path'])
        logs_path.mkdir(parents=True, exist_ok=True)
        
        report_file = logs_path / filename
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved data report to {report_file}")
        return str(report_file)


def main():
    """Example usage of DataLoader."""
    loader = DataLoader()
    
    # Example: Load and process data
    # df = loader.load_csv("data/raw/xauusd_15m.csv")
    # clean_file = loader.save_clean_data(df, "xauusd_15m_clean.csv")
    # report = loader.generate_data_report(df, "xauusd_15m.csv")
    # loader.save_report(report)
    
    print("DataLoader initialized successfully")


if __name__ == "__main__":
    main()