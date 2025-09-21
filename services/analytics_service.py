"""
Analytics Service for DurgasAI.

Provides comprehensive analytics and metrics tracking for the application.
This service collects, processes, and provides insights into:
- User behavior and interaction patterns
- Model performance and usage statistics
- System performance and resource utilization
- Error patterns and debugging insights
- Feature adoption and usage analytics
- Session analytics and user engagement

Architecture:
- Centralized analytics collection and processing
- Integration with logging system for data collection
- Real-time metrics calculation and aggregation
- Historical data analysis and trending
- Export capabilities for external analysis
- Privacy-conscious data collection

Key Features:
- User interaction tracking
- Model performance analytics
- System resource monitoring
- Error pattern analysis
- Feature usage statistics
- Session analytics and engagement metrics
- Customizable reporting and dashboards
- Data export and visualization support
"""

import json
import sqlite3
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import threading
import time
from collections import defaultdict, Counter
import statistics

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation
from .config_service import ConfigService


@dataclass
class UserInteraction:
    """User interaction event."""
    event_id: str
    session_id: str
    user_id: Optional[str]
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    page: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    metric_id: str
    metric_type: str  # 'response_time', 'memory_usage', 'cpu_usage', etc.
    value: float
    unit: str
    timestamp: datetime
    component: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ErrorEvent:
    """Error event tracking."""
    error_id: str
    error_type: str
    error_message: str
    component: str
    timestamp: datetime
    session_id: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalyticsReport:
    """Analytics report structure."""
    report_id: str
    report_type: str
    generated_at: datetime
    date_range: Tuple[datetime, datetime]
    data: Dict[str, Any]
    summary: Dict[str, Any]


class AnalyticsService:
    """
    Service for analytics and metrics tracking.
    
    This service provides comprehensive analytics capabilities for monitoring
    user behavior, system performance, and application usage patterns.
    
    Key Responsibilities:
    - Event tracking and collection
    - Performance metrics monitoring
    - Error pattern analysis
    - User behavior analytics
    - Report generation and insights
    - Data persistence and retrieval
    """
    
    def __init__(self, config_service: ConfigService):
        """
        Initialize the analytics service.
        
        Args:
            config_service: Configuration service instance
        """
        debug("Initializing AnalyticsService", "analytics_service")
        
        self.config_service = config_service
        
        # Database setup
        self.db_path = Path("output/analytics/analytics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches for real-time analytics
        self.user_interactions: List[UserInteraction] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.error_events: List[ErrorEvent] = []
        
        # Analytics configuration
        self.max_cache_size = 10000
        self.batch_size = 100
        self.auto_flush_interval = 300  # 5 minutes
        
        # Threading for background operations
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Initialize database and load configuration
        self._initialize_database()
        self._load_analytics_configuration()
        self._start_background_tasks()
        
        info("AnalyticsService initialized successfully", "analytics_service")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for analytics storage."""
        debug("Initializing analytics database", "analytics_service")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # User interactions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        user_id TEXT,
                        event_type TEXT,
                        event_data TEXT,
                        timestamp TEXT,
                        page TEXT,
                        model_id TEXT
                    )
                """)
                
                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id TEXT PRIMARY KEY,
                        metric_type TEXT,
                        value REAL,
                        unit TEXT,
                        timestamp TEXT,
                        component TEXT,
                        metadata TEXT
                    )
                """)
                
                # Error events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS error_events (
                        error_id TEXT PRIMARY KEY,
                        error_type TEXT,
                        error_message TEXT,
                        component TEXT,
                        timestamp TEXT,
                        session_id TEXT,
                        stack_trace TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON user_interactions(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON user_interactions(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON error_events(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_errors_component ON error_events(component)")
                
                conn.commit()
                
            debug("Analytics database initialized successfully", "analytics_service")
            
        except Exception as e:
            error("Failed to initialize analytics database", "analytics_service", error_obj=e)
    
    def _load_analytics_configuration(self) -> None:
        """Load analytics configuration."""
        debug("Loading analytics configuration", "analytics_service")
        
        try:
            app_config = self.config_service.get_app_config()
            analytics_config = app_config.get('analytics', {})
            
            self.max_cache_size = analytics_config.get('max_cache_size', 10000)
            self.batch_size = analytics_config.get('batch_size', 100)
            self.auto_flush_interval = analytics_config.get('auto_flush_interval', 300)
            
            debug("Analytics configuration loaded", "analytics_service",
                  max_cache_size=self.max_cache_size,
                  batch_size=self.batch_size,
                  auto_flush_interval=self.auto_flush_interval)
            
        except Exception as e:
            warning("Failed to load analytics configuration, using defaults", "analytics_service", error_obj=e)
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for analytics processing."""
        debug("Starting analytics background tasks", "analytics_service")
        
        def background_flush():
            while not self._shutdown:
                try:
                    time.sleep(self.auto_flush_interval)
                    if not self._shutdown:
                        self.flush_to_database()
                except Exception as e:
                    error("Error in background flush task", "analytics_service", error_obj=e)
        
        flush_thread = threading.Thread(target=background_flush, daemon=True)
        flush_thread.start()
        
        debug("Background tasks started", "analytics_service")
    
    def track_user_interaction(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        page: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Track a user interaction event.
        
        Args:
            event_type: Type of interaction
            event_data: Event-specific data
            session_id: Optional session ID
            user_id: Optional user ID
            page: Optional page name
            model_id: Optional model ID
            
        Returns:
            str: Event ID
        """
        debug(f"Tracking user interaction: {event_type}", "analytics_service")
        
        try:
            import uuid
            event_id = str(uuid.uuid4())
            
            interaction = UserInteraction(
                event_id=event_id,
                session_id=session_id or "unknown",
                user_id=user_id,
                event_type=event_type,
                event_data=event_data,
                timestamp=datetime.now(),
                page=page,
                model_id=model_id
            )
            
            with self._lock:
                self.user_interactions.append(interaction)
                
                # Flush if cache is getting full
                if len(self.user_interactions) >= self.max_cache_size:
                    self._flush_user_interactions()
            
            log_user_action("interaction_tracked", 
                event_type=event_type,
                event_id=event_id,
                session_id=session_id
            )
            
            debug(f"User interaction tracked: {event_id}", "analytics_service")
            return event_id
            
        except Exception as e:
            error(f"Failed to track user interaction: {event_type}", "analytics_service", error_obj=e)
            return ""
    
    def track_performance_metric(
        self,
        metric_type: str,
        value: float,
        unit: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track a performance metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            unit: Unit of measurement
            component: Component name
            metadata: Optional metadata
            
        Returns:
            str: Metric ID
        """
        debug(f"Tracking performance metric: {metric_type}", "analytics_service")
        
        try:
            import uuid
            metric_id = str(uuid.uuid4())
            
            metric = PerformanceMetric(
                metric_id=metric_id,
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                component=component,
                metadata=metadata
            )
            
            with self._lock:
                self.performance_metrics.append(metric)
                
                # Flush if cache is getting full
                if len(self.performance_metrics) >= self.max_cache_size:
                    self._flush_performance_metrics()
            
            debug(f"Performance metric tracked: {metric_id}", "analytics_service")
            return metric_id
            
        except Exception as e:
            error(f"Failed to track performance metric: {metric_type}", "analytics_service", error_obj=e)
            return ""
    
    def track_error_event(
        self,
        error_type: str,
        error_message: str,
        component: str,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            component: Component where error occurred
            session_id: Optional session ID
            stack_trace: Optional stack trace
            metadata: Optional metadata
            
        Returns:
            str: Error ID
        """
        debug(f"Tracking error event: {error_type}", "analytics_service")
        
        try:
            import uuid
            error_id = str(uuid.uuid4())
            
            error_event = ErrorEvent(
                error_id=error_id,
                error_type=error_type,
                error_message=error_message,
                component=component,
                timestamp=datetime.now(),
                session_id=session_id,
                stack_trace=stack_trace,
                metadata=metadata
            )
            
            with self._lock:
                self.error_events.append(error_event)
                
                # Flush if cache is getting full
                if len(self.error_events) >= self.max_cache_size:
                    self._flush_error_events()
            
            debug(f"Error event tracked: {error_id}", "analytics_service")
            return error_id
            
        except Exception as e:
            error(f"Failed to track error event: {error_type}", "analytics_service", error_obj=e)
            return ""
    
    def flush_to_database(self) -> None:
        """Flush all cached data to database."""
        debug("Flushing analytics data to database", "analytics_service")
        
        with LoggedOperation("analytics_flush", "analytics_service"):
            try:
                with self._lock:
                    self._flush_user_interactions()
                    self._flush_performance_metrics()
                    self._flush_error_events()
                
                info("Analytics data flushed to database", "analytics_service")
                
            except Exception as e:
                error("Failed to flush analytics data", "analytics_service", error_obj=e)
    
    def _flush_user_interactions(self) -> None:
        """Flush user interactions to database."""
        if not self.user_interactions:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for interaction in self.user_interactions:
                    cursor.execute("""
                        INSERT OR REPLACE INTO user_interactions
                        (event_id, session_id, user_id, event_type, event_data, timestamp, page, model_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interaction.event_id,
                        interaction.session_id,
                        interaction.user_id,
                        interaction.event_type,
                        json.dumps(interaction.event_data),
                        interaction.timestamp.isoformat(),
                        interaction.page,
                        interaction.model_id
                    ))
                
                conn.commit()
                
            debug(f"Flushed {len(self.user_interactions)} user interactions", "analytics_service")
            self.user_interactions.clear()
            
        except Exception as e:
            error("Failed to flush user interactions", "analytics_service", error_obj=e)
    
    def _flush_performance_metrics(self) -> None:
        """Flush performance metrics to database."""
        if not self.performance_metrics:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric in self.performance_metrics:
                    cursor.execute("""
                        INSERT OR REPLACE INTO performance_metrics
                        (metric_id, metric_type, value, unit, timestamp, component, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metric.metric_id,
                        metric.metric_type,
                        metric.value,
                        metric.unit,
                        metric.timestamp.isoformat(),
                        metric.component,
                        json.dumps(metric.metadata) if metric.metadata else None
                    ))
                
                conn.commit()
                
            debug(f"Flushed {len(self.performance_metrics)} performance metrics", "analytics_service")
            self.performance_metrics.clear()
            
        except Exception as e:
            error("Failed to flush performance metrics", "analytics_service", error_obj=e)
    
    def _flush_error_events(self) -> None:
        """Flush error events to database."""
        if not self.error_events:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for error_event in self.error_events:
                    cursor.execute("""
                        INSERT OR REPLACE INTO error_events
                        (error_id, error_type, error_message, component, timestamp, session_id, stack_trace, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        error_event.error_id,
                        error_event.error_type,
                        error_event.error_message,
                        error_event.component,
                        error_event.timestamp.isoformat(),
                        error_event.session_id,
                        error_event.stack_trace,
                        json.dumps(error_event.metadata) if error_event.metadata else None
                    ))
                
                conn.commit()
                
            debug(f"Flushed {len(self.error_events)} error events", "analytics_service")
            self.error_events.clear()
            
        except Exception as e:
            error("Failed to flush error events", "analytics_service", error_obj=e)
    
    def generate_user_analytics_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AnalyticsReport:
        """Generate user analytics report."""
        debug("Generating user analytics report", "analytics_service")
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get user interactions in date range
                cursor.execute("""
                    SELECT * FROM user_interactions
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start_date.isoformat(), end_date.isoformat()))
                
                interactions = cursor.fetchall()
                
                # Process analytics
                total_interactions = len(interactions)
                unique_sessions = len(set(row[1] for row in interactions))
                
                # Event type distribution
                event_types = Counter(row[3] for row in interactions)
                
                # Page usage
                pages = Counter(row[6] for row in interactions if row[6])
                
                # Model usage
                models = Counter(row[7] for row in interactions if row[7])
                
                # Daily activity
                daily_activity = defaultdict(int)
                for row in interactions:
                    date = datetime.fromisoformat(row[5]).date()
                    daily_activity[date.isoformat()] += 1
                
                report_data = {
                    'total_interactions': total_interactions,
                    'unique_sessions': unique_sessions,
                    'event_types': dict(event_types),
                    'page_usage': dict(pages),
                    'model_usage': dict(models),
                    'daily_activity': dict(daily_activity)
                }
                
                summary = {
                    'most_popular_event': event_types.most_common(1)[0] if event_types else None,
                    'most_popular_page': pages.most_common(1)[0] if pages else None,
                    'most_used_model': models.most_common(1)[0] if models else None,
                    'avg_interactions_per_session': total_interactions / unique_sessions if unique_sessions > 0 else 0
                }
                
                import uuid
                report = AnalyticsReport(
                    report_id=str(uuid.uuid4()),
                    report_type='user_analytics',
                    generated_at=datetime.now(),
                    date_range=(start_date, end_date),
                    data=report_data,
                    summary=summary
                )
                
                info("User analytics report generated", "analytics_service")
                return report
                
        except Exception as e:
            error("Failed to generate user analytics report", "analytics_service", error_obj=e)
            
            import uuid
            return AnalyticsReport(
                report_id=str(uuid.uuid4()),
                report_type='user_analytics',
                generated_at=datetime.now(),
                date_range=(start_date, end_date),
                data={},
                summary={'error': str(e)}
            )
    
    def generate_performance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AnalyticsReport:
        """Generate performance analytics report."""
        debug("Generating performance analytics report", "analytics_service")
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get performance metrics in date range
                cursor.execute("""
                    SELECT * FROM performance_metrics
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start_date.isoformat(), end_date.isoformat()))
                
                metrics = cursor.fetchall()
                
                # Process performance analytics
                total_metrics = len(metrics)
                
                # Group by metric type
                metrics_by_type = defaultdict(list)
                for row in metrics:
                    metrics_by_type[row[1]].append(row[2])  # metric_type -> values
                
                # Calculate statistics for each metric type
                metric_stats = {}
                for metric_type, values in metrics_by_type.items():
                    if values:
                        metric_stats[metric_type] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'avg': statistics.mean(values),
                            'median': statistics.median(values)
                        }
                        if len(values) > 1:
                            metric_stats[metric_type]['stdev'] = statistics.stdev(values)
                
                # Component performance
                components = Counter(row[5] for row in metrics)
                
                report_data = {
                    'total_metrics': total_metrics,
                    'metric_statistics': metric_stats,
                    'component_distribution': dict(components),
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    }
                }
                
                # Generate summary insights
                summary = {
                    'total_metric_types': len(metrics_by_type),
                    'most_monitored_component': components.most_common(1)[0] if components else None
                }
                
                # Add performance insights
                if 'response_time' in metric_stats:
                    summary['avg_response_time'] = metric_stats['response_time']['avg']
                
                import uuid
                report = AnalyticsReport(
                    report_id=str(uuid.uuid4()),
                    report_type='performance_analytics',
                    generated_at=datetime.now(),
                    date_range=(start_date, end_date),
                    data=report_data,
                    summary=summary
                )
                
                info("Performance analytics report generated", "analytics_service")
                return report
                
        except Exception as e:
            error("Failed to generate performance analytics report", "analytics_service", error_obj=e)
            
            import uuid
            return AnalyticsReport(
                report_id=str(uuid.uuid4()),
                report_type='performance_analytics',
                generated_at=datetime.now(),
                date_range=(start_date, end_date),
                data={},
                summary={'error': str(e)}
            )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics metrics."""
        debug("Retrieving real-time analytics metrics", "analytics_service")
        
        with self._lock:
            current_time = datetime.now()
            
            # Recent activity (last hour)
            hour_ago = current_time - timedelta(hours=1)
            recent_interactions = [
                interaction for interaction in self.user_interactions
                if interaction.timestamp >= hour_ago
            ]
            recent_metrics = [
                metric for metric in self.performance_metrics
                if metric.timestamp >= hour_ago
            ]
            recent_errors = [
                error_event for error_event in self.error_events
                if error_event.timestamp >= hour_ago
            ]
            
            # Active sessions (last 30 minutes)
            thirty_min_ago = current_time - timedelta(minutes=30)
            active_sessions = set(
                interaction.session_id for interaction in self.user_interactions
                if interaction.timestamp >= thirty_min_ago
            )
            
            metrics = {
                'cache_status': {
                    'user_interactions': len(self.user_interactions),
                    'performance_metrics': len(self.performance_metrics),
                    'error_events': len(self.error_events)
                },
                'recent_activity': {
                    'interactions_last_hour': len(recent_interactions),
                    'metrics_last_hour': len(recent_metrics),
                    'errors_last_hour': len(recent_errors),
                    'active_sessions': len(active_sessions)
                },
                'system_status': {
                    'database_path': str(self.db_path),
                    'last_flush': current_time.isoformat(),
                    'auto_flush_interval': self.auto_flush_interval
                }
            }
            
            debug("Real-time metrics retrieved", "analytics_service")
            return metrics
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old analytics data."""
        debug(f"Cleaning up analytics data older than {days_to_keep} days", "analytics_service")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old user interactions
                cursor.execute("DELETE FROM user_interactions WHERE timestamp < ?", (cutoff_date.isoformat(),))
                deleted_counts['user_interactions'] = cursor.rowcount
                
                # Delete old performance metrics
                cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date.isoformat(),))
                deleted_counts['performance_metrics'] = cursor.rowcount
                
                # Delete old error events
                cursor.execute("DELETE FROM error_events WHERE timestamp < ?", (cutoff_date.isoformat(),))
                deleted_counts['error_events'] = cursor.rowcount
                
                conn.commit()
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                
            info(f"Analytics data cleanup completed", "analytics_service", deleted_counts=deleted_counts)
            return deleted_counts
            
        except Exception as e:
            error("Failed to cleanup old analytics data", "analytics_service", error_obj=e)
            return {}
    
    def shutdown(self) -> None:
        """Shutdown the analytics service."""
        debug("Shutting down AnalyticsService", "analytics_service")
        
        self._shutdown = True
        
        # Flush remaining data
        self.flush_to_database()
        
        info("AnalyticsService shutdown completed", "analytics_service")
