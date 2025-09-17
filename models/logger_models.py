"""
Logger data models and structures for the AI Agent Dashboard

This module defines the data models and structures used by the logging system,
including log entries, configuration, and analysis results.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import re
import json


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    """Individual log entry structure"""
    timestamp: str = Field(description="Log timestamp")
    level: LogLevel = Field(description="Log level")
    message: str = Field(description="Log message")
    logger: str = Field(description="Logger name")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    thread_id: Optional[str] = Field(default=None, description="Thread identifier")
    process_id: Optional[int] = Field(default=None, description="Process ID")
    module: Optional[str] = Field(default=None, description="Module name")
    function: Optional[str] = Field(default=None, description="Function name")
    line_number: Optional[int] = Field(default=None, description="Line number")
    exception_info: Optional[str] = Field(default=None, description="Exception information")
    stack_trace: Optional[List[str]] = Field(default=None, description="Stack trace")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if not v:
            raise ValueError("Timestamp cannot be empty")
        try:
            # Try to parse the timestamp
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError("Invalid timestamp format")
        return v
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()
    
    @validator('logger')
    def validate_logger(cls, v):
        if not v or not v.strip():
            raise ValueError("Logger name cannot be empty")
        return v.strip()
    
    @validator('level')
    def validate_level(cls, v):
        if isinstance(v, str):
            try:
                return LogLevel(v.upper())
            except ValueError:
                raise ValueError(f"Invalid log level: {v}")
        return v


class LogAnalysisResult(BaseModel):
    """Log analysis result structure"""
    total_entries: int = Field(description="Total log entries analyzed")
    level_distribution: Dict[str, int] = Field(description="Distribution of log levels")
    hourly_activity: Dict[str, int] = Field(description="Activity by hour")
    error_count: int = Field(description="Total error count")
    warning_count: int = Field(description="Total warning count")
    common_errors: List[Dict[str, Any]] = Field(description="Most common error patterns")
    performance_entries: int = Field(description="Performance-related entries count")
    analysis_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class LogExportData(BaseModel):
    """Log export data structure"""
    export_timestamp: str = Field(description="Export timestamp")
    entry_count: int = Field(description="Number of entries exported")
    log_entries: List[LogEntry] = Field(description="Exported log entries")
    format: str = Field(description="Export format (txt/json/csv)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Export metadata")


class LogConfiguration(BaseModel):
    """Logger configuration settings"""
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Current log level")
    max_entries: int = Field(default=1000, description="Maximum entries to display")
    auto_refresh: bool = Field(default=False, description="Enable auto-refresh")
    refresh_interval: int = Field(default=30, description="Auto-refresh interval in seconds")
    log_file_path: str = Field(default="output/logs/genai_agent.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, description="Maximum log file size in MB")
    enable_archiving: bool = Field(default=True, description="Enable log archiving")
    archive_retention_days: int = Field(default=30, description="Archive retention period")


class LogStatistics(BaseModel):
    """Log statistics and metrics"""
    total_logs: int = Field(description="Total number of log entries")
    errors: int = Field(description="Number of error entries")
    warnings: int = Field(description="Number of warning entries")
    info: int = Field(description="Number of info entries")
    debug: int = Field(description="Number of debug entries")
    file_size_mb: float = Field(description="Log file size in MB")
    last_updated: str = Field(description="Last update timestamp")
    uptime_hours: float = Field(description="Logger uptime in hours")


class LogFilter(BaseModel):
    """Log filtering criteria"""
    level_filter: Optional[LogLevel] = Field(default=None, description="Filter by log level")
    max_entries: int = Field(default=100, description="Maximum entries to return")
    start_time: Optional[str] = Field(default=None, description="Start time filter")
    end_time: Optional[str] = Field(default=None, description="End time filter")
    message_contains: Optional[str] = Field(default=None, description="Message content filter")
    logger_name: Optional[str] = Field(default=None, description="Logger name filter")


class LogArchiveInfo(BaseModel):
    """Log archive information"""
    archive_name: str = Field(description="Archive file name")
    archive_path: str = Field(description="Archive file path")
    original_size_mb: float = Field(description="Original log file size")
    archive_size_mb: float = Field(description="Archive file size")
    created_at: str = Field(description="Archive creation timestamp")
    entry_count: int = Field(description="Number of entries archived")


def create_log_entry(timestamp: str, level: LogLevel, message: str, 
                    logger: str = "genai_agent", context: Optional[Dict[str, Any]] = None) -> LogEntry:
    """Create a new log entry"""
    return LogEntry(
        timestamp=timestamp,
        level=level,
        message=message,
        logger=logger,
        context=context
    )


def create_log_filter(level_filter: Optional[LogLevel] = None, 
                     max_entries: int = 100,
                     start_time: Optional[str] = None,
                     end_time: Optional[str] = None,
                     message_contains: Optional[str] = None,
                     logger_name: Optional[str] = None) -> LogFilter:
    """Create a log filter"""
    return LogFilter(
        level_filter=level_filter,
        max_entries=max_entries,
        start_time=start_time,
        end_time=end_time,
        message_contains=message_contains,
        logger_name=logger_name
    )


def create_default_log_config() -> LogConfiguration:
    """Create default log configuration"""
    return LogConfiguration()


def validate_log_entry(entry_data: Dict[str, Any]) -> bool:
    """Validate log entry data"""
    try:
        LogEntry(**entry_data)
        return True
    except Exception:
        return False


def parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a log line into a LogEntry with enhanced parsing"""
    try:
        line = line.strip()
        if not line:
            return None
        
        # Try different log formats
        # Format 1: timestamp - logger - level - message
        if ' - ' in line:
            parts = line.split(' - ', 3)
            if len(parts) >= 4:
                timestamp = parts[0]
                logger_name = parts[1]
                level_str = parts[2]
                message = parts[3]
                
                # Extract additional info from message
                context = {}
                thread_id = None
                process_id = None
                module = None
                function = None
                line_number = None
                exception_info = None
                stack_trace = None
                
                # Look for thread info
                thread_match = re.search(r'\[Thread-(\d+)\]', message)
                if thread_match:
                    thread_id = thread_match.group(1)
                    message = re.sub(r'\[Thread-\d+\]', '', message).strip()
                
                # Look for process info
                process_match = re.search(r'\[PID:(\d+)\]', message)
                if process_match:
                    process_id = int(process_match.group(1))
                    message = re.sub(r'\[PID:\d+\]', '', message).strip()
                
                # Look for module/function info
                module_match = re.search(r'\[(\w+\.py):(\d+):(\w+)\]', message)
                if module_match:
                    module = module_match.group(1)
                    line_number = int(module_match.group(2))
                    function = module_match.group(3)
                    message = re.sub(r'\[\w+\.py:\d+:\w+\]', '', message).strip()
                
                # Look for exception info
                if 'Traceback' in message or 'Exception' in message:
                    exception_info = message
                    stack_trace = message.split('\n')
                
                # Validate and convert level
                try:
                    level = LogLevel(level_str.upper())
                except ValueError:
                    level = LogLevel.INFO
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    logger=logger_name,
                    context=context if context else None,
                    thread_id=thread_id,
                    process_id=process_id,
                    module=module,
                    function=function,
                    line_number=line_number,
                    exception_info=exception_info,
                    stack_trace=stack_trace
                )
        
        # Format 2: JSON log format
        if line.startswith('{') and line.endswith('}'):
            try:
                log_data = json.loads(line)
                return LogEntry(
                    timestamp=log_data.get('timestamp', datetime.now().isoformat()),
                    level=LogLevel(log_data.get('level', 'INFO').upper()),
                    message=log_data.get('message', ''),
                    logger=log_data.get('logger', 'unknown'),
                    context=log_data.get('context'),
                    thread_id=log_data.get('thread_id'),
                    process_id=log_data.get('process_id'),
                    module=log_data.get('module'),
                    function=log_data.get('function'),
                    line_number=log_data.get('line_number'),
                    exception_info=log_data.get('exception_info'),
                    stack_trace=log_data.get('stack_trace')
                )
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Format 3: Simple format - just message
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message=line,
            logger="unknown"
        )
        
    except Exception:
        return None


def parse_log_file(file_path: str, max_lines: int = 1000) -> List[LogEntry]:
    """Parse a log file and return list of LogEntry objects"""
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                entry = parse_log_line(line)
                if entry:
                    entries.append(entry)
    except Exception as e:
        print(f"Error parsing log file {file_path}: {e}")
    return entries


def filter_logs_by_level(entries: List[LogEntry], level: LogLevel) -> List[LogEntry]:
    """Filter log entries by level"""
    return [entry for entry in entries if entry.level == level]


def filter_logs_by_time_range(entries: List[LogEntry], start_time: datetime, end_time: datetime) -> List[LogEntry]:
    """Filter log entries by time range"""
    filtered = []
    for entry in entries:
        try:
            entry_time = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
            if start_time <= entry_time <= end_time:
                filtered.append(entry)
        except ValueError:
            continue
    return filtered


def filter_logs_by_logger(entries: List[LogEntry], logger_name: str) -> List[LogEntry]:
    """Filter log entries by logger name"""
    return [entry for entry in entries if entry.logger == logger_name]


def filter_logs_by_message_pattern(entries: List[LogEntry], pattern: str) -> List[LogEntry]:
    """Filter log entries by message pattern (regex)"""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        return [entry for entry in entries if regex.search(entry.message)]
    except re.error:
        return []


def get_log_statistics(entries: List[LogEntry]) -> Dict[str, Any]:
    """Get comprehensive statistics from log entries"""
    if not entries:
        return {}
    
    stats = {
        'total_entries': len(entries),
        'level_distribution': {},
        'logger_distribution': {},
        'time_range': {
            'earliest': None,
            'latest': None
        },
        'error_count': 0,
        'warning_count': 0,
        'info_count': 0,
        'debug_count': 0,
        'critical_count': 0,
        'unique_loggers': set(),
        'unique_modules': set(),
        'exception_count': 0
    }
    
    timestamps = []
    
    for entry in entries:
        # Level distribution
        level_name = entry.level.value
        stats['level_distribution'][level_name] = stats['level_distribution'].get(level_name, 0) + 1
        
        # Logger distribution
        stats['logger_distribution'][entry.logger] = stats['logger_distribution'].get(entry.logger, 0) + 1
        
        # Count by level
        if entry.level == LogLevel.ERROR:
            stats['error_count'] += 1
        elif entry.level == LogLevel.WARNING:
            stats['warning_count'] += 1
        elif entry.level == LogLevel.INFO:
            stats['info_count'] += 1
        elif entry.level == LogLevel.DEBUG:
            stats['debug_count'] += 1
        elif entry.level == LogLevel.CRITICAL:
            stats['critical_count'] += 1
        
        # Collect unique values
        stats['unique_loggers'].add(entry.logger)
        if entry.module:
            stats['unique_modules'].add(entry.module)
        
        # Exception count
        if entry.exception_info:
            stats['exception_count'] += 1
        
        # Collect timestamps
        try:
            entry_time = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
            timestamps.append(entry_time)
        except ValueError:
            continue
    
    # Time range
    if timestamps:
        stats['time_range']['earliest'] = min(timestamps).isoformat()
        stats['time_range']['latest'] = max(timestamps).isoformat()
    
    # Convert sets to lists for JSON serialization
    stats['unique_loggers'] = list(stats['unique_loggers'])
    stats['unique_modules'] = list(stats['unique_modules'])
    
    return stats


def export_logs_to_json(entries: List[LogEntry], file_path: str) -> bool:
    """Export log entries to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([entry.dict() for entry in entries], f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error exporting logs to JSON: {e}")
        return False


def export_logs_to_csv(entries: List[LogEntry], file_path: str) -> bool:
    """Export log entries to CSV file"""
    try:
        import csv
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if entries:
                writer = csv.DictWriter(f, fieldnames=entries[0].dict().keys())
                writer.writeheader()
                for entry in entries:
                    writer.writerow(entry.dict())
        return True
    except Exception as e:
        print(f"Error exporting logs to CSV: {e}")
        return False
