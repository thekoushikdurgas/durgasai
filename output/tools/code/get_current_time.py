"""
Enhanced Get Current Time Tool
Supports 70+ time formats and standards based on comprehensive time format guide
"""

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

def get_current_time(
    format_type: str = "iso_8601_extended",
    timezone_offset: Optional[str] = None,
    include_timezone: bool = True,
    precision: str = "seconds",
    include_weekday: bool = False,
    include_week_number: bool = False,
    include_julian_date: bool = False,
    include_epoch_variants: bool = False,
    include_historical_formats: bool = False,
    custom_format_pattern: Optional[str] = None,
    include_common_strings: bool = False
) -> str:
    """
    Get the current date and time in various formats and standards
    
    Args:
        format_type (str): Type of time format to return
        timezone_offset (str): Timezone offset (e.g., "+05:30", "-08:00", "UTC")
        include_timezone (bool): Whether to include timezone information
        precision (str): Time precision (seconds, milliseconds, microseconds, nanoseconds)
        include_weekday (bool): Whether to include weekday information
        include_week_number (bool): Whether to include week number
        include_julian_date (bool): Whether to include Julian date
        include_epoch_variants (bool): Whether to include various epoch time formats
        include_historical_formats (bool): Whether to include historical time formats
        custom_format_pattern (str): Custom strftime pattern for custom formatting
        include_common_strings (bool): Whether to include common time string formats (mm/dd, dd/mm, etc.)
    
    Returns:
        str: JSON string containing formatted time data
    """
    try:
        # Get current time
        now_utc = datetime.now(timezone.utc)
        
        # Apply timezone offset if specified
        if timezone_offset and timezone_offset != "UTC":
            if timezone_offset.startswith(('+', '-')):
                # Parse offset like "+05:30" or "-08:00"
                sign = 1 if timezone_offset.startswith('+') else -1
                offset_str = timezone_offset[1:]
                hours, minutes = map(int, offset_str.split(':'))
                offset = timedelta(hours=sign * hours, minutes=sign * minutes)
                now = now_utc + offset
            else:
                # Handle named timezones (simplified)
                now = now_utc
        else:
            now = now_utc
        
        # Initialize result dictionary
        result = {
            "timestamp": now.timestamp(),
            "unix_timestamp_seconds": int(now.timestamp()),
            "unix_timestamp_milliseconds": int(now.timestamp() * 1000),
            "unix_timestamp_microseconds": int(now.timestamp() * 1000000),
            "unix_timestamp_nanoseconds": int(now.timestamp() * 1000000000),
            "format_type": format_type,
            "timezone": str(now.tzinfo) if now.tzinfo else "UTC"
        }
        
        # Add precision-based timestamps
        if precision == "milliseconds":
            result["timestamp"] = now.timestamp() * 1000
        elif precision == "microseconds":
            result["timestamp"] = now.timestamp() * 1000000
        elif precision == "nanoseconds":
            result["timestamp"] = now.timestamp() * 1000000000
        
        # Handle custom format pattern
        if custom_format_pattern:
            try:
                result["formatted_time"] = now.strftime(custom_format_pattern)
                result["custom_format_pattern"] = custom_format_pattern
                result["custom_format_result"] = now.strftime(custom_format_pattern)
            except ValueError as e:
                result["formatted_time"] = now.isoformat()
                result["custom_format_error"] = f"Invalid format pattern: {str(e)}"
        # Format based on requested format type
        elif format_type == "iso_8601_extended":
            result["formatted_time"] = now.isoformat()
            result["iso_8601_date"] = now.strftime("%Y-%m-%d")
            result["iso_8601_time"] = now.strftime("%H:%M:%S")
            if include_timezone:
                result["iso_8601_utc"] = now_utc.isoformat() + "Z"
        
        elif format_type == "iso_8601_basic":
            result["formatted_time"] = now.strftime("%Y%m%dT%H%M%S")
            result["iso_8601_date_basic"] = now.strftime("%Y%m%d")
            result["iso_8601_time_basic"] = now.strftime("%H%M%S")
        
        elif format_type == "rfc_3339":
            result["formatted_time"] = now.isoformat()
            result["rfc_3339_utc"] = now_utc.isoformat() + "Z"
            result["rfc_3339_with_offset"] = now.isoformat()
        
        elif format_type == "unix_epoch":
            result["formatted_time"] = str(int(now.timestamp()))
            result["unix_seconds"] = int(now.timestamp())
            result["unix_milliseconds"] = int(now.timestamp() * 1000)
            result["unix_microseconds"] = int(now.timestamp() * 1000000)
            result["unix_nanoseconds"] = int(now.timestamp() * 1000000000)
        
        elif format_type == "12_hour_clock":
            result["formatted_time"] = now.strftime("%I:%M:%S %p")
            result["12_hour_time"] = now.strftime("%I:%M:%S %p")
            result["am_pm_designation"] = now.strftime("%p")
            result["12_hour_hour"] = now.strftime("%I")
        
        elif format_type == "24_hour_clock":
            result["formatted_time"] = now.strftime("%H:%M:%S")
            result["24_hour_time"] = now.strftime("%H:%M:%S")
            result["24_hour_hour"] = now.strftime("%H")
        
        elif format_type == "military_time":
            result["formatted_time"] = now.strftime("%H%M")
            result["military_time"] = now.strftime("%H%M")
            result["military_time_with_seconds"] = now.strftime("%H%M%S")
        
        elif format_type == "human_readable":
            result["formatted_time"] = now.strftime("%B %d, %Y at %I:%M:%S %p")
            result["long_date"] = now.strftime("%A, %B %d, %Y")
            result["short_date"] = now.strftime("%m/%d/%Y")
            result["time_only"] = now.strftime("%I:%M:%S %p")
        
        elif format_type == "system_formats":
            result["formatted_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            result["system_timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
            result["log_format"] = now.strftime("%Y-%m-%d %H:%M:%S")
            result["database_format"] = now.strftime("%Y-%m-%d %H:%M:%S")
        
        else:  # Default to ISO 8601 extended
            result["formatted_time"] = now.isoformat()
        
        # Add additional information based on flags
        if include_weekday:
            result["weekday"] = now.strftime("%A")
            result["weekday_short"] = now.strftime("%a")
            result["weekday_number"] = now.weekday()
        
        if include_week_number:
            result["week_number"] = now.isocalendar()[1]
            result["year_week"] = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
            result["weekday_in_week"] = now.isocalendar()[2]
        
        if include_julian_date:
            # Calculate Julian Day Number (simplified)
            jd = now.toordinal() + 1721424.5
            result["julian_day_number"] = jd
            result["modified_julian_day"] = jd - 2400000.5
            result["julian_date"] = f"{int(jd)}.{int((jd % 1) * 86400)}"
        
        if include_epoch_variants:
            result["epoch_variants"] = {
                "unix_seconds": int(now.timestamp()),
                "unix_milliseconds": int(now.timestamp() * 1000),
                "unix_microseconds": int(now.timestamp() * 1000000),
                "unix_nanoseconds": int(now.timestamp() * 1000000000),
                "javascript_timestamp": int(now.timestamp() * 1000),
                "java_system_time": int(now.timestamp() * 1000)
            }
        
        if include_historical_formats:
            result["historical_formats"] = {
                "gmt_time": now_utc.strftime("%H:%M:%S GMT"),
                "utc_time": now_utc.strftime("%H:%M:%S UTC"),
                "railway_time": now_utc.strftime("%H:%M:%S"),
                "sidereal_time": "Not calculated",  # Would require astronomical calculations
                "apparent_solar_time": "Not calculated"  # Would require solar position calculations
            }
        
        # Add timezone information
        if include_timezone:
            result["timezone_info"] = {
                "utc_offset": now.strftime("%z"),
                "timezone_name": str(now.tzinfo),
                "dst_active": now.dst() != timedelta(0) if now.dst() else False,
                "utc_time": now_utc.isoformat(),
                "local_time": now.isoformat()
            }
        
        # Add common string formats if requested
        if include_common_strings:
            result["common_strings"] = {
                "mm_dd": now.strftime("%m/%d"),
                "dd_mm": now.strftime("%d/%m"),
                "mm_dd_yyyy": now.strftime("%m/%d/%Y"),
                "dd_mm_yyyy": now.strftime("%d/%m/%Y"),
                "yyyy_mm_dd": now.strftime("%Y/%m/%d"),
                "mm_dd_yy": now.strftime("%m/%d/%y"),
                "dd_mm_yy": now.strftime("%d/%m/%y"),
                "yy_mm_dd": now.strftime("%y/%m/%d"),
                "mm_dd_hh_mm": now.strftime("%m/%d %H:%M"),
                "dd_mm_hh_mm": now.strftime("%d/%m %H:%M"),
                "hh_mm": now.strftime("%H:%M"),
                "hh_mm_ss": now.strftime("%H:%M:%S"),
                "hh_mm_am_pm": now.strftime("%I:%M %p"),
                "month_name_dd": now.strftime("%B %d"),
                "month_abbr_dd": now.strftime("%b %d"),
                "day_month_year": now.strftime("%d %B %Y"),
                "day_month_abbr_year": now.strftime("%d %b %Y"),
                "weekday_month_dd": now.strftime("%A, %B %d"),
                "weekday_abbr_month_dd": now.strftime("%a, %b %d"),
                "mm_dd_yyyy_hh_mm": now.strftime("%m/%d/%Y %H:%M"),
                "dd_mm_yyyy_hh_mm": now.strftime("%d/%m/%Y %H:%M"),
                "yyyy_mm_dd_hh_mm": now.strftime("%Y/%m/%d %H:%M"),
                "mm_dd_yyyy_hh_mm_ss": now.strftime("%m/%d/%Y %H:%M:%S"),
                "dd_mm_yyyy_hh_mm_ss": now.strftime("%d/%m/%Y %H:%M:%S"),
                "yyyy_mm_dd_hh_mm_ss": now.strftime("%Y/%m/%d %H:%M:%S")
            }
        
        # Add comprehensive format collection
        result["all_formats"] = {
            "iso_8601_extended": now.isoformat(),
            "iso_8601_basic": now.strftime("%Y%m%dT%H%M%S"),
            "rfc_3339": now.isoformat(),
            "unix_timestamp": int(now.timestamp()),
            "12_hour": now.strftime("%I:%M:%S %p"),
            "24_hour": now.strftime("%H:%M:%S"),
            "military": now.strftime("%H%M"),
            "human_readable": now.strftime("%B %d, %Y at %I:%M:%S %p"),
            "system_format": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date_only": now.strftime("%Y-%m-%d"),
            "time_only": now.strftime("%H:%M:%S"),
            "long_date": now.strftime("%A, %B %d, %Y"),
            "short_date": now.strftime("%m/%d/%Y")
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "error": f"Error getting current time: {str(e)}",
            "timestamp": time.time(),
            "format_type": format_type
        }
        return json.dumps(error_result, indent=2)