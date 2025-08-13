"""
Data resampling module for XAUUSD market structure detection.

This module handles:
- Resampling 15m data to higher timeframes (1H, 4H, Daily)
- OHLC aggregation with proper candle formation
- Volume aggregation (if available)
- Timezone-aware resampling
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class DataResampler:
    """Handles resampling of OHLC data to higher timeframes."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataResampler with configuration."""
        self.config = self._load_config(config_path)
        self.timeframe_mapping = {
            '15min': '15T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D',
            'Daily': '1D'
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def resample_ohlc(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLC data to target timeframe.
        
        Args:
            df: DataFrame with datetime index and OHLC columns
            target_timeframe: Target timeframe (1H, 4H, 1D)
            
        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {target_timeframe}")
        
        if 'datetime' not in df.columns:
            raise ValueError("DataFrame must have 'datetime' column")
        
        # Set datetime as index for resampling
        df_indexed = df.set_index('datetime')
        
        # Get pandas frequency string
        freq = self._get_pandas_frequency(target_timeframe)
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        
        # Add volume if present
        if 'volume' in df.columns:
            agg_rules['volume'] = 'sum'
        
        try:
            # Perform resampling
            resampled = df_indexed.resample(freq).agg(agg_rules)
            
            # Remove rows with NaN values (incomplete periods)
            resampled = resampled.dropna()
            
            # Reset index to make datetime a column again
            resampled = resampled.reset_index()
            
            # Ensure proper column order
            column_order = ['datetime', 'open', 'high', 'low', 'close']
            if 'volume' in resampled.columns:
                column_order.append('volume')
            
            resampled = resampled[column_order]
            
            logger.info(f"Resampled from {len(df)} to {len(resampled)} candles")
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            raise
    
    def _get_pandas_frequency(self, timeframe: str) -> str:
        """Convert timeframe string to pandas frequency."""
        if timeframe in self.timeframe_mapping:
            return self.timeframe_mapping[timeframe]
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def validate_resampled_data(self, original_df: pd.DataFrame, 
                              resampled_df: pd.DataFrame, 
                              timeframe: str) -> Dict:
        """
        Validate resampled data integrity.
        
        Args:
            original_df: Original 15m data
            resampled_df: Resampled data
            timeframe: Target timeframe
            
        Returns:
            Validation report
        """
        logger.info(f"Validating resampled {timeframe} data")
        
        validation_report = {
            'timeframe': timeframe,
            'validation_timestamp': datetime.now().isoformat(),
            'original_candles': len(original_df),
            'resampled_candles': len(resampled_df),
            'checks': {}
        }
        
        # Check OHLC relationships
        ohlc_valid = (
            (resampled_df['high'] >= resampled_df['open']) &
            (resampled_df['high'] >= resampled_df['close']) &
            (resampled_df['low'] <= resampled_df['open']) &
            (resampled_df['low'] <= resampled_df['close']) &
            (resampled_df['high'] >= resampled_df['low'])
        ).all()
        
        validation_report['checks']['ohlc_relationships_valid'] = bool(ohlc_valid)
        
        # Check for missing values
        missing_values = resampled_df.isnull().sum()
        validation_report['checks']['missing_values'] = missing_values.to_dict()
        
        # Check datetime continuity
        if len(resampled_df) > 1:
            # Expected frequency based on timeframe
            expected_freq = self._get_expected_frequency_minutes(timeframe)
            time_diffs = resampled_df['datetime'].diff().dt.total_seconds() / 60
            
            # Allow for some variation due to market hours/weekends
            irregular_intervals = (time_diffs > expected_freq * 1.5).sum()
            validation_report['checks']['irregular_intervals'] = int(irregular_intervals)
        
        # Price range checks
        price_stats = {
            'min_price': float(resampled_df[['open', 'high', 'low', 'close']].min().min()),
            'max_price': float(resampled_df[['open', 'high', 'low', 'close']].max().max()),
            'price_range_original': float(original_df[['open', 'high', 'low', 'close']].max().max() - 
                                        original_df[['open', 'high', 'low', 'close']].min().min()),
            'price_range_resampled': float(resampled_df[['open', 'high', 'low', 'close']].max().max() - 
                                         resampled_df[['open', 'high', 'low', 'close']].min().min())
        }
        validation_report['checks']['price_statistics'] = price_stats
        
        # Log validation results
        if ohlc_valid:
            logger.info(f"✓ {timeframe} OHLC relationships valid")
        else:
            logger.warning(f"✗ {timeframe} OHLC relationships invalid")
        
        if missing_values.sum() == 0:
            logger.info(f"✓ {timeframe} No missing values")
        else:
            logger.warning(f"✗ {timeframe} Missing values found: {missing_values.to_dict()}")
        
        return validation_report
    
    def _get_expected_frequency_minutes(self, timeframe: str) -> int:
        """Get expected frequency in minutes for timeframe."""
        frequency_map = {
            '15min': 15,
            '1H': 60,
            '4H': 240,
            '1D': 1440,
            'Daily': 1440
        }
        return frequency_map.get(timeframe, 60)
    
    def resample_all_timeframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample data to all configured timeframes.
        
        Args:
            df: Source DataFrame (typically 15m data)
            
        Returns:
            Dictionary of timeframe -> DataFrame
        """
        resampled_data = {}
        base_timeframe = self.config['data']['base_timeframe']
        target_timeframes = self.config['data']['resampled_timeframes']
        
        logger.info(f"Resampling {base_timeframe} data to {target_timeframes}")
        
        # Keep original data
        resampled_data[base_timeframe] = df.copy()
        
        # Resample to each target timeframe
        for timeframe in target_timeframes:
            try:
                resampled_df = self.resample_ohlc(df, timeframe)
                resampled_data[timeframe] = resampled_df
                
                # Validate resampled data
                validation_report = self.validate_resampled_data(df, resampled_df, timeframe)
                logger.info(f"Resampling to {timeframe} completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to resample to {timeframe}: {e}")
                continue
        
        return resampled_data
    
    def save_resampled_data(self, resampled_data: Dict[str, pd.DataFrame], 
                          base_filename: str) -> Dict[str, str]:
        """
        Save resampled data to CSV files.
        
        Args:
            resampled_data: Dictionary of timeframe -> DataFrame
            base_filename: Base filename (without extension)
            
        Returns:
            Dictionary of timeframe -> saved file path
        """
        resampled_path = Path(self.config['data']['resampled_data_path'])
        resampled_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for timeframe, df in resampled_data.items():
            # Create filename with timeframe suffix
            filename = f"{base_filename}_{timeframe}.csv"
            file_path = resampled_path / filename
            
            try:
                df.to_csv(file_path, index=False)
                saved_files[timeframe] = str(file_path)
                logger.info(f"Saved {timeframe} data to {file_path}")
                
            except Exception as e:
                logger.error(f"Error saving {timeframe} data: {e}")
                continue
        
        return saved_files
    
    def create_resampling_report(self, resampled_data: Dict[str, pd.DataFrame], 
                               validation_reports: List[Dict]) -> Dict:
        """Create comprehensive resampling report."""
        report = {
            'resampling_timestamp': datetime.now().isoformat(),
            'base_timeframe': self.config['data']['base_timeframe'],
            'target_timeframes': self.config['data']['resampled_timeframes'],
            'summary': {},
            'validation_results': validation_reports
        }
        
        # Add summary statistics for each timeframe
        for timeframe, df in resampled_data.items():
            report['summary'][timeframe] = {
                'total_candles': len(df),
                'date_range': {
                    'start': df['datetime'].min().isoformat(),
                    'end': df['datetime'].max().isoformat()
                },
                'price_range': {
                    'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                    'max': float(df[['open', 'high', 'low', 'close']].max().max())
                }
            }
        
        return report


def main():
    """Example usage of DataResampler."""
    resampler = DataResampler()
    
    # Example usage:
    # from src.data.loader import DataLoader
    # loader = DataLoader()
    # df = loader.load_csv("data/clean/xauusd_15m_clean.csv")
    # resampled_data = resampler.resample_all_timeframes(df)
    # saved_files = resampler.save_resampled_data(resampled_data, "xauusd")
    
    print("DataResampler initialized successfully")


if __name__ == "_main_":
    main()