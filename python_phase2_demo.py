#!/usr/bin/env python3
"""
Phase 2 Demo Script - XAUUSD Market Structure Detection Core Engine

This script demonstrates the complete Phase 2 implementation:
1. Market structure detection (HH/HL/LL/LH)
2. Two-candle confirmation patterns
3. CHoCH detection (Change of Character)
4. Internal CHoCH detection
5. Engulfing patterns for 4H+ timeframes
6. Event export and reporting

Usage:
    python phase2_demo.py [--sample] [--input-file INPUT_FILE] [--timeframe TIMEFRAME]
"""

import argparse
import logging
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.engine.detector import MarketStructureDetector
from src.engine.state import EventType
from src.data.loader import DataLoader
from src.utils.helpers import create_sample_data, export_to_json, calculate_price_statistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_complex_test_scenario(n_candles: int = 200) -> pd.DataFrame:
    """
    Create complex test scenario with multiple market structure patterns.
    
    Args:
        n_candles: Number of candles to generate
        
    Returns:
        DataFrame with realistic market structure scenarios
    """
    logger.info(f"Creating complex test scenario with {n_candles} candles")
    
    dates = pd.date_range('2023-01-01 00:00:00', periods=n_candles, freq='15T')
    data = []
    
    base_price = 2000.0
    phase = "uptrend"  # uptrend, downtrend, consolidation
    phase_candles = 0
    phase_duration = 40  # Candles per phase
    
    for i, dt in enumerate(dates):
        phase_candles += 1
        
        # Switch phases periodically
        if phase_candles > phase_duration:
            phases = ["uptrend", "downtrend", "consolidation"]
            current_idx = phases.index(phase)
            phase = phases[(current_idx + 1) % len(phases)]
            phase_candles = 0
            phase_duration = np.random.randint(30, 60)  # Vary phase duration
            logger.info(f"Switching to {phase} phase at candle {i}")
        
        # Generate candle based on current phase
        if phase == "uptrend":
            trend_bias = 0.6  # Bullish bias
            volatility = 3.0
        elif phase == "downtrend":
            trend_bias = -0.6  # Bearish bias
            volatility = 3.5
        else:  # consolidation
            trend_bias = 0.0
            volatility = 2.0
        
        # Add some randomness and trend
        price_change = np.random.normal(trend_bias, 1.0)
        base_price = max(base_price + price_change, 10.0)  # Keep prices positive
        
        # Create realistic OHLC
        open_price = base_price + np.random.normal(0, 0.5)
        close_price = open_price + np.random.normal(trend_bias * 0.5, volatility)
        
        # Ensure proper OHLC relationships
        high = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.3))
        low = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.3))
        
        # Add specific patterns at key points
        if phase_candles == 10 and phase in ["uptrend", "downtrend"]:
            # Create clear confirmation patterns
            if phase == "uptrend":
                # Two red candles for HH confirmation
                if i > 0:  # Second red candle
                    close_price = data[i-1]['low'] - 1  # Close below previous low
                    open_price = close_price + 3  # Red candle
                    high = open_price + 1
                    low = close_price - 1
            else:  # downtrend
                # Two green candles for LL confirmation
                if i > 0:  # Second green candle
                    close_price = data[i-1]['high'] + 1  # Close above previous high
                    open_price = close_price - 3  # Green candle
                    high = close_price + 1
                    low = open_price - 1
        
        data.append({
            'datetime': dt,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': np.random.randint(1000, 10000),
            'phase': phase  # For analysis
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created complex scenario: {len(df)} candles across multiple phases")
    return df


def analyze_detection_performance(detector: MarketStructureDetector, 
                                df: pd.DataFrame, 
                                timeframe: str) -> dict:
    """
    Analyze detection performance and generate metrics.
    
    Args:
        detector: MarketStructureDetector instance
        df: Input OHLC data
        timeframe: Timeframe being analyzed
        
    Returns:
        Performance analysis dictionary
    """
    logger.info("Analyzing detection performance...")
    
    start_time = time.time()
    events = detector.detect_market_structure(df, timeframe)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Event statistics
    event_counts = {}
    for event_type in EventType:
        count = sum(1 for event in events if event.event_type == event_type)
        event_counts[event_type.value] = count
    
    # Price analysis
    if events:
        prices = [event.price for event in events]
        price_stats = {
            'min_event_price': min(prices),
            'max_event_price': max(prices),
            'price_range': max(prices) - min(prices),
            'avg_event_price': sum(prices) / len(prices)
        }
    else:
        price_stats = {'min_event_price': 0, 'max_event_price': 0, 'price_range': 0, 'avg_event_price': 0}
    
    # Sequence analysis
    sequence_analysis = analyze_event_sequence(events)
    
    # Performance metrics
    performance = {
        'processing_time_seconds': processing_time,
        'candles_processed': len(df),
        'candles_per_second': len(df) / processing_time if processing_time > 0 else 0,
        'events_detected': len(events),
        'events_per_1000_candles': (len(events) / len(df) * 1000) if len(df) > 0 else 0,
        'detection_efficiency': len(events) / processing_time if processing_time > 0 else 0
    }
    
    return {
        'performance_metrics': performance,
        'event_statistics': event_counts,
        'price_analysis': price_stats,
        'sequence_analysis': sequence_analysis,
        'events': events
    }


def analyze_event_sequence(events: list) -> dict:
    """
    Analyze the sequence of detected events for patterns.
    
    Args:
        events: List of MarketEvent objects
        
    Returns:
        Sequence analysis dictionary
    """
    if not events:
        return {'total_sequences': 0, 'patterns': {}}
    
    # Sort events by candle index
    sorted_events = sorted(events, key=lambda x: x.candle_index)
    
    # Analyze patterns
    patterns = {
        'hh_hl_sequences': 0,
        'll_lh_sequences': 0,
        'choch_after_swings': 0,
        'internal_choch_count': 0
    }
    
    # Look for HH→HL sequences
    for i in range(len(sorted_events) - 1):
        current = sorted_events[i]
        next_event = sorted_events[i + 1]
        
        if (current.event_type == EventType.HH and 
            next_event.event_type == EventType.HL):
            patterns['hh_hl_sequences'] += 1
        
        elif (current.event_type == EventType.LL and 
              next_event.event_type == EventType.LH):
            patterns['ll_lh_sequences'] += 1
    
    # Count CHoCH and Internal CHoCH
    for event in sorted_events:
        if 'CHOCH' in event.event_type.value:
            if 'INTERNAL' in event.event_type.value:
                patterns['internal_choch_count'] += 1
            else:
                patterns['choch_after_swings'] += 1
    
    # Time intervals between events
    if len(sorted_events) > 1:
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = sorted_events[i].candle_index - sorted_events[i-1].candle_index
            intervals.append(interval)
        
        interval_stats = {
            'avg_interval': sum(intervals) / len(intervals),
            'min_interval': min(intervals),
            'max_interval': max(intervals)
        }
    else:
        interval_stats = {'avg_interval': 0, 'min_interval': 0, 'max_interval': 0}
    
    return {
        'total_events': len(sorted_events),
        'patterns': patterns,
        'interval_statistics': interval_stats,
        'first_event_candle': sorted_events[0].candle_index if sorted_events else 0,
        'last_event_candle': sorted_events[-1].candle_index if sorted_events else 0
    }


def display_event_summary(events: list, timeframe: str):
    """
    Display a formatted summary of detected events.
    
    Args:
        events: List of MarketEvent objects
        timeframe: Timeframe analyzed
    """
    print(f"\n{'='*80}")
    print(f"MARKET STRUCTURE EVENTS DETECTED - {timeframe}")
    print(f"{'='*80}")
    
    if not events:
        print("No events detected in the data.")
        return
    
    # Group events by type
    event_groups = {}
    for event in events:
        event_type = event.event_type.value
        if event_type not in event_groups:
            event_groups[event_type] = []
        event_groups[event_type].append(event)
    
    # Display each group
    for event_type, group_events in event_groups.items():
        print(f"\n{event_type} Events ({len(group_events)}):")
        print("-" * 60)
        
        for event in sorted(group_events, key=lambda x: x.candle_index):
            datetime_str = event.datetime.strftime("%Y-%m-%d %H:%M")
            print(f"  Candle {event.candle_index:3d} | {datetime_str} | "
                  f"Price: ${event.price:7.2f} | Rule: {event.trigger_rule}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("DETECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Events: {len(events)}")
    
    for event_type in EventType:
        count = sum(1 for e in events if e.event_type == event_type)
        if count > 0:
            print(f"{event_type.value:15s}: {count:3d}")
    
    if events:
        first_event = min(events, key=lambda x: x.candle_index)
        last_event = max(events, key=lambda x: x.candle_index)
        print(f"\nFirst Event: Candle {first_event.candle_index} ({first_event.event_type.value})")
        print(f"Last Event:  Candle {last_event.candle_index} ({last_event.event_type.value})")


def export_detection_results(analysis: dict, output_dir: str, session_id: str):
    """
    Export detection results to files.
    
    Args:
        analysis: Analysis results dictionary
        output_dir: Output directory path
        session_id: Unique session identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export events to CSV
    if analysis['events']:
        events_data = []
        for event in analysis['events']:
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
        
        events_df = pd.DataFrame(events_data)
        events_file = output_path / f"events_{session_id}_{timestamp}.csv"
        events_df.to_csv(events_file, index=False)
        logger.info(f"Exported {len(events_data)} events to {events_file}")
    
    # Export analysis report
    report = {
        'session_info': {
            'session_id': session_id,
            'timestamp': timestamp,
            'phase': 'Phase 2 - Core Detection Engine'
        },
        'performance_metrics': analysis['performance_metrics'],
        'event_statistics': analysis['event_statistics'],
        'price_analysis': analysis['price_analysis'],
        'sequence_analysis': analysis['sequence_analysis']
    }
    
    report_file = output_path / f"detection_report_{session_id}_{timestamp}.json"
    export_to_json(report, str(report_file))
    logger.info(f"Exported analysis report to {report_file}")
    
    return {
        'events_file': str(events_file) if analysis['events'] else None,
        'report_file': str(report_file)
    }


def main():
    """Main demo script."""
    parser = argparse.ArgumentParser(description='XAUUSD Market Structure Detection - Phase 2 Demo')
    parser.add_argument('--sample', action='store_true', 
                       help='Generate and use sample data')
    parser.add_argument('--input-file', type=str, 
                       help='Path to input CSV file (if not using sample data)')
    parser.add_argument('--timeframe', type=str, default='15min',
                       choices=['15min', '1H', '4H', '1D'],
                       help='Timeframe for detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--complex-scenario', action='store_true',
                       help='Use complex test scenario instead of simple sample')
    parser.add_argument('--output-dir', type=str, default='outputs/phase2',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("XAUUSD MARKET STRUCTURE DETECTION - PHASE 2 DEMO")
        print("Core Detection Engine")
        print("=" * 80)
        
        # Step 1: Initialize detector
        print("\n1. Initializing Market Structure Detector...")
        detector = MarketStructureDetector(args.config)
        print("✓ MarketStructureDetector initialized")
        print(f"✓ Configuration loaded: {args.config}")
        
        # Step 2: Load or generate data
        print("\n2. Loading/generating data...")
        if args.sample:
            if args.complex_scenario:
                print("   Generating complex test scenario...")
                df = create_complex_test_scenario(n_candles=500)
                data_source = "complex_scenario"
            else:
                print("   Generating simple sample data...")
                df = create_sample_data(n_candles=200, start_price=2000.0)
                data_source = "simple_sample"
            
            print(f"✓ Generated {len(df)} sample candles")
            
        elif args.input_file:
            print(f"   Loading data from: {args.input_file}")
            loader = DataLoader(args.config)
            df = loader.load_csv(args.input_file)
            data_source = args.input_file
            print(f"✓ Loaded {len(df)} candles from file")
            
        else:
            print("   Error: Please specify --sample or --input-file")
            return 1
        
        # Step 3: Display data overview
        print("\n3. Data Overview...")
        stats = calculate_price_statistics(df)
        print(f"   Candles: {len(df)}")
        print(f"   Date range: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
        print(f"   Price range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
        print(f"   Green: {stats['candle_analysis']['green_pct']:.1f}% | Red: {stats['candle_analysis']['red_pct']:.1f}% | Doji: {stats['candle_analysis']['doji_pct']:.1f}%")
        
        # Step 4: Run market structure detection
        print(f"\n4. Running market structure detection ({args.timeframe})...")
        print("   Detecting: HH, HL, LL, LH, CHoCH, Internal CHoCH")
        
        analysis = analyze_detection_performance(detector, df, args.timeframe)
        
        performance = analysis['performance_metrics']
        print(f"✓ Detection completed in {performance['processing_time_seconds']:.3f} seconds")
        print(f"✓ Processing speed: {performance['candles_per_second']:.0f} candles/second")
        print(f"✓ Events detected: {performance['events_detected']}")
        print(f"✓ Event density: {performance['events_per_1000_candles']:.1f} events per 1000 candles")
        
        # Step 5: Display results
        print("\n5. Detection Results...")
        display_event_summary(analysis['events'], args.timeframe)
        
        # Step 6: Sequence analysis
        print(f"\n{'='*80}")
        print("SEQUENCE ANALYSIS")
        print(f"{'='*80}")
        
        seq_analysis = analysis['sequence_analysis']
        patterns = seq_analysis['patterns']
        
        print(f"HH→HL sequences: {patterns['hh_hl_sequences']}")
        print(f"LL→LH sequences: {patterns['ll_lh_sequences']}")
        print(f"CHoCH events: {patterns['choch_after_swings']}")
        print(f"Internal CHoCH events: {patterns['internal_choch_count']}")
        
        if seq_analysis['interval_statistics']['avg_interval'] > 0:
            print(f"Average interval between events: {seq_analysis['interval_statistics']['avg_interval']:.1f} candles")
            print(f"Min/Max intervals: {seq_analysis['interval_statistics']['min_interval']} - {seq_analysis['interval_statistics']['max_interval']} candles")
        
        # Step 7: Export results
        print(f"\n6. Exporting results...")
        session_id = f"phase2_{args.timeframe}_{datetime.now().strftime('%H%M%S')}"
        export_files = export_detection_results(analysis, args.output_dir, session_id)
        
        if export_files['events_file']:
            print(f"✓ Events exported to: {export_files['events_file']}")
        print(f"✓ Analysis report exported to: {export_files['report_file']}")
        
        # Step 8: Validation and quality checks
        print(f"\n7. Quality Assessment...")
        
        # Basic validation
        events = analysis['events']
        if events:
            # Check for logical sequence
            sorted_events = sorted(events, key=lambda x: x.candle_index)
            
            # Validate event indices
            invalid_indices = [e for e in events if e.candle_index < 0 or e.candle_index >= len(df)]
            if not invalid_indices:
                print("✓ All event indices are valid")
            else:
                print(f"⚠  Found {len(invalid_indices)} events with invalid indices")
            
            # Check price validity
            invalid_prices = [e for e in events if e.price <= 0]
            if not invalid_prices:
                print("✓ All event prices are positive")
            else:
                print(f"⚠  Found {len(invalid_prices)} events with invalid prices")
            
            # Check for duplicate events
            event_keys = set()
            duplicates = 0
            for event in events:
                key = (event.event_type, event.candle_index, event.price)
                if key in event_keys:
                    duplicates += 1
                else:
                    event_keys.add(key)
            
            if duplicates == 0:
                print("✓ No duplicate events detected")
            else:
                print(f"⚠  Found {duplicates} potential duplicate events")
        
        # Performance assessment
        if performance['candles_per_second'] > 1000:
            print("✓ Excellent processing speed")
        elif performance['candles_per_second'] > 500:
            print("✓ Good processing speed")
        else:
            print("⚠  Processing speed could be optimized")
        
        # Step 9: Success summary
        print(f"\n{'='*80}")
        print("PHASE 2 COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"✓ Processed {len(df)} {args.timeframe} candles")
        print(f"✓ Detected {len(analysis['events'])} market structure events")
        print(f"✓ Processing time: {performance['processing_time_seconds']:.3f} seconds")
        print(f"✓ Detection rules: All core patterns implemented")
        print(f"✓ Results exported to: {args.output_dir}")
        
        print(f"\nNext Steps:")
        print("- Review detected events for accuracy")
        print("- Proceed to Phase 3: Interactive Visualization")
        print("- Use exported data for further analysis")
        print("- Test with different timeframes and datasets")
        
        # Detailed breakdown
        print(f"\nDetection Breakdown:")
        for event_type, count in analysis['event_statistics'].items():
            if count > 0:
                print(f"  {event_type}: {count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np  # Import numpy for the demo
    exit_code = main()
    sys.exit(exit_code)