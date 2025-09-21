"""
Get comprehensive weather information for a location using Open-Meteo API.
Supports current weather, historical data, and forecasts with detailed parameters.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time

def get_current_weather(
    location: str, 
    unit: str = "celsius", 
    date: Optional[str] = None, 
    time: Optional[str] = None,
    forecast_days: int = 1,
    include_daily: bool = False,
    include_hourly: bool = True
) -> str:
    """
    Get comprehensive weather information for a location using Open-Meteo API.
    
    Args:
        location: The city and state/country, e.g. "San Francisco, CA" or "London, UK"
        unit: Temperature unit, either "celsius" or "fahrenheit"
        date: Date in YYYY-MM-DD format for historical/forecast data. Leave empty for current weather
        time: Time in HH:MM format (24-hour) for specific hourly data. Leave empty for current time
        forecast_days: Number of forecast days (1-16). Only used when date is not specified
        include_daily: Include daily weather summary (max/min temps, precipitation, etc.)
        include_hourly: Include detailed hourly weather data
    
    Returns:
        JSON string containing comprehensive weather data including:
        - weather_report: Formatted weather information summary
        - current_weather: Current weather conditions
        - location_info: Location details including coordinates and timezone
        - hourly_weather: Hourly weather data (if requested)
        - daily_weather: Daily weather summary (if requested)
        - api_info: API response metadata and data source information
    """
    try:
        # Step 1: Geocode location to get coordinates
        coordinates = _geocode_location(location)
        if not coordinates:
            return _create_error_response(f"Location '{location}' not found")
        
        lat, lon, location_details = coordinates
        
        # Step 2: Determine API endpoint and date range
        api_endpoint, date_range = _determine_api_endpoint(date, forecast_days)
        if not api_endpoint:
            return _create_error_response(f"Invalid date range: {date}")
        
        # Step 3: Build API parameters
        params = _build_api_parameters(
            lat, lon, unit, date_range, include_daily, include_hourly
        )
        
        # Step 4: Make API request
        response = requests.get(api_endpoint, params=params, timeout=30)
        
        if response.status_code != 200:
            return _create_error_response(
                f"API request failed (status {response.status_code}): {response.text}"
            )
        
        data = response.json()
        
        if 'error' in data:
            return _create_error_response(f"API error: {data.get('reason', 'Unknown error')}")
        
        # Step 5: Process and format response
        result = _process_weather_data(
            data, location_details, unit, date, time, include_daily, include_hourly
        )
        
        return json.dumps(result, indent=2)
        
    except requests.exceptions.RequestException as e:
        return _create_error_response(f"Network error: {str(e)}")
    except Exception as e:
        return _create_error_response(f"Unexpected error: {str(e)}")

def _geocode_location(location: str) -> Optional[tuple]:
    """Geocode location using Open-Meteo Geocoding API"""
    try:
        # Try multiple location formats for better geocoding success
        location_variants = [
            location,  # Original location
            location.split(',')[0].strip(),  # Just the city name
            location.replace(',', ' ').strip()  # Remove commas
        ]
        
        for loc_variant in location_variants:
            geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": loc_variant,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            response = requests.get(geocoding_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return (
                        result["latitude"],
                        result["longitude"],
                        {
                            "name": result["name"],
                            "country": result.get("country", ""),
                            "admin1": result.get("admin1", ""),
                            "timezone": result.get("timezone", "UTC"),
                            "elevation": result.get("elevation", 0)
                        }
                    )
        
        return None
    except Exception:
        return None

def _determine_api_endpoint(date: Optional[str], forecast_days: int) -> tuple:
    """Determine which API endpoint to use based on date"""
    if not date:
        return "https://api.open-meteo.com/v1/forecast", None
    
    try:
        requested_date = datetime.strptime(date, "%Y-%m-%d").date()
        current_date = datetime.now().date()
        
        # Historical data (more than 5 days ago)
        if requested_date < current_date - timedelta(days=5):
            return "https://archive-api.open-meteo.com/v1/archive", (date, date)
        
        # Forecast data (future dates up to 16 days)
        elif requested_date > current_date + timedelta(days=16):
            return None, None  # Invalid date range
        
        # Recent historical or near-future forecast
        else:
            return "https://api.open-meteo.com/v1/forecast", (date, date)
    
    except ValueError:
        return None, None

def _build_api_parameters(
    lat: float, 
    lon: float, 
    unit: str, 
    date_range: Optional[tuple], 
    include_daily: bool, 
    include_hourly: bool
) -> Dict[str, Any]:
    """Build API parameters based on requirements"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto"
    }
    
    # Add date range if specified
    if date_range:
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]
    
    # Set temperature unit
    if unit == "fahrenheit":
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
        params["precipitation_unit"] = "inch"
    else:
        params["temperature_unit"] = "celsius"
        params["wind_speed_unit"] = "kmh"
        params["precipitation_unit"] = "mm"
    
    # Add hourly parameters
    if include_hourly:
        hourly_vars = [
            "temperature_2m", "relative_humidity_2m", "precipitation", 
            "weather_code", "wind_speed_10m", "wind_direction_10m",
            "pressure_msl", "cloud_cover", "visibility"
        ]
        params["hourly"] = ",".join(hourly_vars)
    
    # Add daily parameters
    if include_daily:
        daily_vars = [
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max"
        ]
        params["daily"] = ",".join(daily_vars)
    
    # Add current weather parameters
    params["current"] = "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
    
    return params

def _process_weather_data(
    data: Dict[str, Any], 
    location_details: Dict[str, Any], 
    unit: str, 
    date: Optional[str], 
    time: Optional[str],
    include_daily: bool, 
    include_hourly: bool
) -> Dict[str, Any]:
    """Process and format weather data from API response"""
    
    # Extract current weather
    current_weather = _extract_current_weather(data, unit)
    
    # Extract hourly weather if requested
    hourly_weather = []
    if include_hourly and "hourly" in data:
        hourly_weather = _extract_hourly_weather(data, unit, date, time)
    
    # Extract daily weather if requested
    daily_weather = []
    if include_daily and "daily" in data:
        daily_weather = _extract_daily_weather(data, unit)
    
    # Create weather report
    weather_report = _create_weather_report(
        current_weather, location_details, unit, date, time
    )
    
    # Extract API info
    api_info = {
        "generation_time_ms": data.get("generationtime_ms", 0),
        "timezone": data.get("timezone", "UTC"),
        "utc_offset_seconds": data.get("utc_offset_seconds", 0),
        "elevation": data.get("elevation", 0),
        "data_source": "Open-Meteo API"
    }
    
    return {
        "weather_report": weather_report,
        "current_weather": current_weather,
        "location_info": {
            "name": location_details["name"],
            "country": location_details["country"],
            "admin1": location_details["admin1"],
            "timezone": location_details["timezone"],
            "elevation": location_details["elevation"],
            "latitude": data.get("latitude", 0),
            "longitude": data.get("longitude", 0)
        },
        "hourly_weather": hourly_weather,
        "daily_weather": daily_weather,
        "api_info": api_info
    }

def _extract_current_weather(data: Dict[str, Any], unit: str) -> Dict[str, Any]:
    """Extract current weather information"""
    current = data.get("current", {})
    units = data.get("current_units", {})
    
    temp_unit = "°C" if unit == "celsius" else "°F"
    
    return {
        "temperature": current.get("temperature_2m", 0),
        "temperature_unit": temp_unit,
        "humidity": current.get("relative_humidity_2m", 0),
        "humidity_unit": "%",
        "weather_code": current.get("weather_code", 0),
        "weather_description": _get_weather_description(current.get("weather_code", 0)),
        "wind_speed": current.get("wind_speed_10m", 0),
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
        "timestamp": current.get("time", "")
    }

def _extract_hourly_weather(
    data: Dict[str, Any], 
    unit: str, 
    date: Optional[str], 
    time: Optional[str]
) -> List[Dict[str, Any]]:
    """Extract hourly weather data"""
    hourly = data.get("hourly", {})
    units = data.get("hourly_units", {})
    
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    humidities = hourly.get("relative_humidity_2m", [])
    weather_codes = hourly.get("weather_code", [])
    wind_speeds = hourly.get("wind_speed_10m", [])
    precipitations = hourly.get("precipitation", [])
    
    hourly_data = []
    
    for i, timestamp in enumerate(times):
        # Filter by specific time if requested
        if time and not timestamp.endswith(f"T{time}:00"):
            continue
        
        # Filter by specific date if requested
        if date and not timestamp.startswith(date):
            continue
        
        hourly_data.append({
            "timestamp": timestamp,
            "temperature": temps[i] if i < len(temps) else None,
            "temperature_unit": units.get("temperature_2m", "°C"),
            "humidity": humidities[i] if i < len(humidities) else None,
            "humidity_unit": units.get("relative_humidity_2m", "%"),
            "weather_code": weather_codes[i] if i < len(weather_codes) else None,
            "weather_description": _get_weather_description(weather_codes[i] if i < len(weather_codes) else 0),
            "wind_speed": wind_speeds[i] if i < len(wind_speeds) else None,
            "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
            "precipitation": precipitations[i] if i < len(precipitations) else None,
            "precipitation_unit": units.get("precipitation", "mm")
        })
    
    return hourly_data

def _extract_daily_weather(data: Dict[str, Any], unit: str) -> List[Dict[str, Any]]:
    """Extract daily weather summary"""
    daily = data.get("daily", {})
    units = data.get("daily_units", {})
    
    times = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precipitations = daily.get("precipitation_sum", [])
    weather_codes = daily.get("weather_code", [])
    
    daily_data = []
    
    for i, date in enumerate(times):
        daily_data.append({
            "date": date,
            "max_temperature": max_temps[i] if i < len(max_temps) else None,
            "min_temperature": min_temps[i] if i < len(min_temps) else None,
            "temperature_unit": units.get("temperature_2m_max", "°C"),
            "precipitation": precipitations[i] if i < len(precipitations) else None,
            "precipitation_unit": units.get("precipitation_sum", "mm"),
            "weather_code": weather_codes[i] if i < len(weather_codes) else None,
            "weather_description": _get_weather_description(weather_codes[i] if i < len(weather_codes) else 0)
        })
    
    return daily_data

def _create_weather_report(
    current_weather: Dict[str, Any], 
    location_details: Dict[str, Any], 
    unit: str, 
    date: Optional[str], 
    time: Optional[str]
) -> str:
    """Create a formatted weather report"""
    location_name = location_details["name"]
    country = location_details["country"]
    admin1 = location_details["admin1"]
    
    location_str = f"{location_name}"
    if admin1:
        location_str += f", {admin1}"
    if country:
        location_str += f", {country}"
    
    temp = current_weather.get("temperature", 0)
    temp_unit = current_weather.get("temperature_unit", "°C")
    humidity = current_weather.get("humidity", 0)
    weather_desc = current_weather.get("weather_description", "Unknown")
    wind_speed = current_weather.get("wind_speed", 0)
    wind_unit = current_weather.get("wind_speed_unit", "km/h")
    
    time_info = ""
    if date:
        time_info = f" for {date}"
        if time:
            time_info += f" at {time}"
    else:
        time_info = " (current conditions)"
    
    report = f"""🌤️ Weather Report for {location_str}{time_info}

📍 Location: {location_str}
🌡️ Temperature: {temp}{temp_unit}
💧 Humidity: {humidity}%
🌤️ Conditions: {weather_desc}
💨 Wind Speed: {wind_speed} {wind_unit}

Data provided by Open-Meteo API"""
    
    return report

def _get_weather_description(weather_code: int) -> str:
    """Convert WMO weather code to description"""
    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_descriptions.get(weather_code, "Unknown")

def _create_error_response(error_message: str) -> str:
    """Create error response"""
    error_result = {
        "weather_report": f"❌ Error: {error_message}",
        "current_weather": None,
        "location_info": None,
        "hourly_weather": [],
        "daily_weather": [],
        "api_info": {
            "error": True,
            "message": error_message,
            "data_source": "Open-Meteo API"
        }
    }
    return json.dumps(error_result, indent=2)
