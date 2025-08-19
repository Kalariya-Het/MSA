"""
State management for XAUUSD market structure detection.

This module defines:
- Data structures for tracking trend state
- Swing state management (candidate HH/LL)
- Market events representation
- Event types enumeration

Maintains all state in deterministic structures for single-pass algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import pandas as pd


class EventType(Enum):
    """Market structure event types."""
    HH = "HH"                           # Higher High
    HL = "HL"                           # Higher Low  
    LL = "LL"                           # Lower Low
    LH = "LH"                           # Lower High
    CHOCH_UP_DOWN = "CHOCH_UP_DOWN"     # Change of Character Up to Down
    CHOCH_DOWN_UP = "CHOCH_DOWN_UP"     # Change of Character Down to Up
    INTERNAL_CHOCH_UP_DOWN = "INTERNAL_CHOCH_UP_DOWN"     # Internal CHoCH Up to Down
    INTERNAL_CHOCH_DOWN_UP = "INTERNAL_CHOCH_DOWN_UP"     # Internal CHoCH Down to Up


@dataclass
class MarketEvent:
    """Represents a detected market structure event."""
    id: str                                    # Unique identifier
    event_type: EventType                      # Type of event
    datetime: datetime                         # Event datetime (candle timestamp)
    price: float                              # Event price
    candle_index: int                         # Index of event candle in DataFrame
    trigger_index: int                        # Index of candle that triggered confirmation
    trigger_rule: str                         # Rule that triggered the event
    timeframe: str                            # Timeframe (15min, 1H, 4H, etc.)
    notes: str = ""                           # Additional notes
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        
        if self.candle_index < 0:
            raise ValueError("Candle index must be non-negative")
        
        if self.trigger_index < self.candle_index:
            raise ValueError("Trigger index must be >= candle index")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'datetime': self.datetime.isoformat() if isinstance(self.datetime, datetime) else str(self.datetime),
            'price': self.price,
            'candle_index': self.candle_index,
            'trigger_index': self.trigger_index,
            'trigger_rule': self.trigger_rule,
            'timeframe': self.timeframe,
            'notes': self.notes,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketEvent':
        """Create event from dictionary."""
        # Parse datetime if it's a string
        dt = data['datetime']
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        return cls(
            id=data['id'],
            event_type=EventType(data['event_type']),
            datetime=dt,
            price=float(data['price']),
            candle_index=int(data['candle_index']),
            trigger_index=int(data['trigger_index']),
            trigger_rule=data['trigger_rule'],
            timeframe=data['timeframe'],
            notes=data.get('notes', ''),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation of event."""
        return (f"{self.event_type.value} at {self.price:.2f} "
                f"(candle {self.candle_index}, {self.datetime.strftime('%Y-%m-%d %H:%M')})")
    
    def __repr__(self) -> str:
        """Detailed representation of event."""
        return (f"MarketEvent(type={self.event_type.value}, price={self.price:.2f}, "
                f"candle_idx={self.candle_index}, trigger_idx={self.trigger_index})")


@dataclass
class TrendState:
    """Tracks current trend state and confirmed swings."""
    
    # Current trend direction
    current_trend: str = "UNKNOWN"  # UP, DOWN, or UNKNOWN
    
    # Last confirmed Higher High
    last_hh_idx: Optional[int] = None
    last_hh_price: Optional[float] = None
    last_hh_datetime: Optional[datetime] = None
    
    # Last confirmed Higher Low
    last_hl_idx: Optional[int] = None
    last_hl_price: Optional[float] = None
    last_hl_datetime: Optional[datetime] = None
    
    # Last confirmed Lower Low
    last_ll_idx: Optional[int] = None
    last_ll_price: Optional[float] = None
    last_ll_datetime: Optional[datetime] = None
    
    # Last confirmed Lower High
    last_lh_idx: Optional[int] = None
    last_lh_price: Optional[float] = None
    last_lh_datetime: Optional[datetime] = None
    
    def get_last_major_swing(self) -> Optional[Dict[str, Any]]:
        """Get the most recent major swing (HH or LL)."""
        last_hh_idx = self.last_hh_idx or -1
        last_ll_idx = self.last_ll_idx or -1
        
        if last_hh_idx > last_ll_idx and self.last_hh_price:
            return {
                'type': 'HH',
                'index': self.last_hh_idx,
                'price': self.last_hh_price,
                'datetime': self.last_hh_datetime
            }
        elif last_ll_idx > last_hh_idx and self.last_ll_price:
            return {
                'type': 'LL',
                'index': self.last_ll_idx,
                'price': self.last_ll_price,
                'datetime': self.last_ll_datetime
            }
        
        return None
    
    def get_last_minor_swing(self) -> Optional[Dict[str, Any]]:
        """Get the most recent minor swing (HL or LH)."""
        last_hl_idx = self.last_hl_idx or -1
        last_lh_idx = self.last_lh_idx or -1
        
        if last_hl_idx > last_lh_idx and self.last_hl_price:
            return {
                'type': 'HL',
                'index': self.last_hl_idx,
                'price': self.last_hl_price,
                'datetime': self.last_hl_datetime
            }
        elif last_lh_idx > last_hl_idx and self.last_lh_price:
            return {
                'type': 'LH',
                'index': self.last_lh_idx,
                'price': self.last_lh_price,
                'datetime': self.last_lh_datetime
            }
        
        return None
    
    def reset(self):
        """Reset trend state."""
        self.current_trend = "UNKNOWN"
        self.last_hh_idx = None
        self.last_hh_price = None
        self.last_hh_datetime = None
        self.last_hl_idx = None
        self.last_hl_price = None
        self.last_hl_datetime = None
        self.last_ll_idx = None
        self.last_ll_price = None
        self.last_ll_datetime = None
        self.last_lh_idx = None
        self.last_lh_price = None
        self.last_lh_datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'current_trend': self.current_trend,
            'last_hh': {
                'index': self.last_hh_idx,
                'price': self.last_hh_price,
                'datetime': self.last_hh_datetime.isoformat() if self.last_hh_datetime else None
            } if self.last_hh_idx is not None else None,
            'last_hl': {
                'index': self.last_hl_idx,
                'price': self.last_hl_price,
                'datetime': self.last_hl_datetime.isoformat() if self.last_hl_datetime else None
            } if self.last_hl_idx is not None else None,
            'last_ll': {
                'index': self.last_ll_idx,
                'price': self.last_ll_price,
                'datetime': self.last_ll_datetime.isoformat() if self.last_ll_datetime else None
            } if self.last_ll_idx is not None else None,
            'last_lh': {
                'index': self.last_lh_idx,
                'price': self.last_lh_price,
                'datetime': self.last_lh_datetime.isoformat() if self.last_lh_datetime else None
            } if self.last_lh_idx is not None else None
        }


@dataclass 
class SwingState:
    """Tracks candidate swing highs and lows during detection."""
    
    # Candidate Higher High (highest high since last HL)
    candidate_hh_idx: Optional[int] = None
    candidate_hh_price: Optional[float] = None
    candidate_hh_datetime: Optional[datetime] = None
    
    # Candidate Lower Low (lowest low since last LH)
    candidate_ll_idx: Optional[int] = None
    candidate_ll_price: Optional[float] = None
    candidate_ll_datetime: Optional[datetime] = None
    
    def reset_candidate_hh(self):
        """Reset candidate HH after confirmation or trend change."""
        self.candidate_hh_idx = None
        self.candidate_hh_price = None
        self.candidate_hh_datetime = None
    
    def reset_candidate_ll(self):
        """Reset candidate LL after confirmation or trend change."""
        self.candidate_ll_idx = None
        self.candidate_ll_price = None
        self.candidate_ll_datetime = None
    
    def reset_all_candidates(self):
        """Reset all candidate swings."""
        self.reset_candidate_hh()
        self.reset_candidate_ll()
    
    def has_candidate_hh(self) -> bool:
        """Check if there's a valid candidate HH."""
        return all(x is not None for x in [self.candidate_hh_idx, self.candidate_hh_price, self.candidate_hh_datetime])
    
    def has_candidate_ll(self) -> bool:
        """Check if there's a valid candidate LL."""
        return all(x is not None for x in [self.candidate_ll_idx, self.candidate_ll_price, self.candidate_ll_datetime])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'candidate_hh': {
                'index': self.candidate_hh_idx,
                'price': self.candidate_hh_price,
                'datetime': self.candidate_hh_datetime.isoformat() if self.candidate_hh_datetime else None
            } if self.candidate_hh_idx is not None else None,
            'candidate_ll': {
                'index': self.candidate_ll_idx,
                'price': self.candidate_ll_price,
                'datetime': self.candidate_ll_datetime.isoformat() if self.candidate_ll_datetime else None
            } if self.candidate_ll_idx is not None else None
        }


@dataclass
class DetectionSession:
    """Represents a complete detection session with all state and results."""
    
    session_id: str
    timeframe: str
    start_datetime: datetime
    end_datetime: Optional[datetime] = None
    
    # State objects
    trend_state: TrendState = field(default_factory=TrendState)
    swing_state: SwingState = field(default_factory=SwingState)
    
    # Results
    events: List[MarketEvent] = field(default_factory=list)
    processed_candles: int = 0
    
    # Session metadata
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: MarketEvent):
        """Add event to session and maintain sorted order."""
        self.events.append(event)
        # Keep events sorted by candle index
        self.events.sort(key=lambda x: x.candle_index)
    
    def get_events_by_type(self, event_type: EventType) -> List[MarketEvent]:
        """Get all events of specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_in_range(self, start_idx: int, end_idx: int) -> List[MarketEvent]:
        """Get events within candle index range."""
        return [event for event in self.events 
                if start_idx <= event.candle_index <= end_idx]
    
    def get_recent_events(self, count: int = 10) -> List[MarketEvent]:
        """Get most recent events by candle index."""
        return sorted(self.events, key=lambda x: x.candle_index)[-count:]
    
    def finalize_session(self, processing_time: Optional[float] = None):
        """Finalize detection session."""
        self.end_datetime = datetime.now()
        
        if processing_time:
            self.performance_metrics['processing_time_seconds'] = processing_time
            self.performance_metrics['candles_per_second'] = (
                self.processed_candles / processing_time if processing_time > 0 else 0
            )
        
        # Calculate event statistics
        event_counts = {}
        for event_type in EventType:
            count = len(self.get_events_by_type(event_type))
            event_counts[event_type.value] = count
        
        self.performance_metrics['event_counts'] = event_counts
        self.performance_metrics['total_events'] = len(self.events)
        self.performance_metrics['events_per_1000_candles'] = (
            (len(self.events) / self.processed_candles * 1000) if self.processed_candles > 0 else 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'timeframe': self.timeframe,
            'start_datetime': self.start_datetime.isoformat(),
            'end_datetime': self.end_datetime.isoformat() if self.end_datetime else None,
            'processed_candles': self.processed_candles,
            'trend_state': self.trend_state.to_dict(),
            'swing_state': self.swing_state.to_dict(),
            'events': [event.to_dict() for event in self.events],
            'config_snapshot': self.config_snapshot,
            'performance_metrics': self.performance_metrics
        }
    
    def export_events_summary(self) -> Dict[str, Any]:
        """Export summary of detected events."""
        if not self.events:
            return {'total_events': 0, 'event_breakdown': {}, 'date_range': None}
        
        # Event breakdown
        event_breakdown = {}
        for event_type in EventType:
            events = self.get_events_by_type(event_type)
            event_breakdown[event_type.value] = {
                'count': len(events),
                'prices': [e.price for e in events],
                'datetimes': [e.datetime.isoformat() for e in events]
            }
        
        # Date range
        sorted_events = sorted(self.events, key=lambda x: x.datetime)
        date_range = {
            'start': sorted_events[0].datetime.isoformat(),
            'end': sorted_events[-1].datetime.isoformat(),
            'duration_hours': (sorted_events[-1].datetime - sorted_events[0].datetime).total_seconds() / 3600
        }
        
        return {
            'session_id': self.session_id,
            'timeframe': self.timeframe,
            'total_events': len(self.events),
            'processed_candles': self.processed_candles,
            'event_breakdown': event_breakdown,
            'date_range': date_range,
            'current_trend': self.trend_state.current_trend,
            'performance_metrics': self.performance_metrics
        }


class StateManager:
    """Manages state persistence and recovery for detection sessions."""
    
    def __init__(self):
        self.current_session: Optional[DetectionSession] = None
        self.session_history: List[str] = []  # Session IDs
    
    def start_new_session(self, session_id: str, timeframe: str, config: Dict[str, Any]) -> DetectionSession:
        """Start a new detection session."""
        session = DetectionSession(
            session_id=session_id,
            timeframe=timeframe,
            start_datetime=datetime.now(),
            config_snapshot=config.copy()
        )
        
        self.current_session = session
        self.session_history.append(session_id)
        
        return session
    
    def save_session_state(self, filepath: str):
        """Save current session state to file."""
        if not self.current_session:
            raise ValueError("No active session to save")
        
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2, default=str)
    
    def load_session_state(self, filepath: str) -> DetectionSession:
        """Load session state from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct session
        session = DetectionSession(
            session_id=data['session_id'],
            timeframe=data['timeframe'],
            start_datetime=pd.to_datetime(data['start_datetime']),
            end_datetime=pd.to_datetime(data['end_datetime']) if data['end_datetime'] else None,
            processed_candles=data['processed_candles'],
            config_snapshot=data['config_snapshot'],
            performance_metrics=data['performance_metrics']
        )
        
        # Reconstruct events
        session.events = [MarketEvent.from_dict(event_data) for event_data in data['events']]
        
        # Reconstruct trend state
        trend_data = data['trend_state']
        session.trend_state.current_trend = trend_data['current_trend']
        
        if trend_data['last_hh']:
            hh = trend_data['last_hh']
            session.trend_state.last_hh_idx = hh['index']
            session.trend_state.last_hh_price = hh['price']
            session.trend_state.last_hh_datetime = pd.to_datetime(hh['datetime']) if hh['datetime'] else None
        
        if trend_data['last_hl']:
            hl = trend_data['last_hl']
            session.trend_state.last_hl_idx = hl['index']
            session.trend_state.last_hl_price = hl['price']
            session.trend_state.last_hl_datetime = pd.to_datetime(hl['datetime']) if hl['datetime'] else None
        
        if trend_data['last_ll']:
            ll = trend_data['last_ll']
            session.trend_state.last_ll_idx = ll['index']
            session.trend_state.last_ll_price = ll['price']
            session.trend_state.last_ll_datetime = pd.to_datetime(ll['datetime']) if ll['datetime'] else None
        
        if trend_data['last_lh']:
            lh = trend_data['last_lh']
            session.trend_state.last_lh_idx = lh['index']
            session.trend_state.last_lh_price = lh['price']
            session.trend_state.last_lh_datetime = pd.to_datetime(lh['datetime']) if lh['datetime'] else None
        
        # Reconstruct swing state
        swing_data = data['swing_state']
        if swing_data['candidate_hh']:
            hh = swing_data['candidate_hh']
            session.swing_state.candidate_hh_idx = hh['index']
            session.swing_state.candidate_hh_price = hh['price']
            session.swing_state.candidate_hh_datetime = pd.to_datetime(hh['datetime']) if hh['datetime'] else None
        
        if swing_data['candidate_ll']:
            ll = swing_data['candidate_ll']
            session.swing_state.candidate_ll_idx = ll['index']
            session.swing_state.candidate_ll_price = ll['price']
            session.swing_state.candidate_ll_datetime = pd.to_datetime(ll['datetime']) if ll['datetime'] else None
        
        self.current_session = session
        return session
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current session."""
        if not self.current_session:
            return None
        
        return self.current_session.export_events_summary()


# Utility functions for state management
def create_event_id() -> str:
    """Create unique event ID."""
    import uuid
    return str(uuid.uuid4())


def create_session_id(timeframe: str) -> str:
    """Create unique session ID."""
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"session_{timeframe}_{timestamp}_{short_uuid}"


def validate_event_sequence(events: List[MarketEvent]) -> Dict[str, Any]:
    """Validate sequence of market structure events for logical consistency."""
    validation_report = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    if not events:
        return validation_report
    
    # Sort events by candle index
    sorted_events = sorted(events, key=lambda x: x.candle_index)
    
    # Check for basic patterns
    hh_count = sum(1 for e in events if e.event_type == EventType.HH)
    hl_count = sum(1 for e in events if e.event_type == EventType.HL)
    ll_count = sum(1 for e in events if e.event_type == EventType.LL)
    lh_count = sum(1 for e in events if e.event_type == EventType.LH)
    
    validation_report['statistics'] = {
        'total_events': len(events),
        'hh_count': hh_count,
        'hl_count': hl_count,
        'll_count': ll_count,
        'lh_count': lh_count,
        'choch_count': sum(1 for e in events if 'CHOCH' in e.event_type.value),
        'internal_choch_count': sum(1 for e in events if 'INTERNAL_CHOCH' in e.event_type.value)
    }
    
    # Basic logical checks
    if hh_count == 0 and hl_count > 0:
        validation_report['warnings'].append("HL events without any HH events")
    
    if ll_count == 0 and lh_count > 0:
        validation_report['warnings'].append("LH events without any LL events")
    
    # Check price progression for HH/LL
    hh_events = [e for e in sorted_events if e.event_type == EventType.HH]
    for i in range(1, len(hh_events)):
        if hh_events[i].price <= hh_events[i-1].price:
            validation_report['warnings'].append(
                f"HH at index {hh_events[i].candle_index} price {hh_events[i].price} "
                f"not higher than previous HH {hh_events[i-1].price}"
            )
    
    ll_events = [e for e in sorted_events if e.event_type == EventType.LL]
    for i in range(1, len(ll_events)):
        if ll_events[i].price >= ll_events[i-1].price:
            validation_report['warnings'].append(
                f"LL at index {ll_events[i].candle_index} price {ll_events[i].price} "
                f"not lower than previous LL {ll_events[i-1].price}"
            )
    
    # Set overall validity
    validation_report['is_valid'] = len(validation_report['errors']) == 0
    
    return validation_report


def main():
    """Example usage and testing of state management."""
    import uuid
    
    # Test event creation
    test_event = MarketEvent(
        id=str(uuid.uuid4()),
        event_type=EventType.HH,
        datetime=datetime.now(),
        price=2000.50,
        candle_index=100,
        trigger_index=102,
        trigger_rule="two_red_close_below_low",
        timeframe="15min",
        notes="Test HH event"
    )
    
    print("Created test event:", test_event)
    
    # Test state objects
    trend_state = TrendState()
    swing_state = SwingState()
    
    print("Initialized trend state:", trend_state.current_trend)
    print("Has candidate HH:", swing_state.has_candidate_hh())
    
    # Test session management
    state_manager = StateManager()
    session = state_manager.start_new_session("test_session", "15min", {})
    session.add_event(test_event)
    
    print("Session summary:", state_manager.get_session_summary())
    
    print("State management testing complete!")


if __name__ == "__main__":
    main()