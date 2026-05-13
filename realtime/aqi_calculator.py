"""
Module for calculating Air Quality Index (AQI) based on US EPA standards.
"""

import pandas as pd

# US EPA AQI Breakpoints
AQI_BREAKPOINTS = {
    "pm25": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "o3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
    "so2": [
        (0.000, 0.035, 0, 50),
        (0.036, 0.075, 51, 100),
        (0.076, 0.185, 101, 150),
        (0.186, 0.304, 151, 200),
        (0.305, 0.604, 201, 300),
        (0.605, 1.004, 301, 500),
    ],
    "no2": [
        (0.000, 0.053, 0, 50),
        (0.054, 0.100, 51, 100),
        (0.101, 0.360, 101, 150),
        (0.361, 0.649, 151, 200),
        (0.650, 1.249, 201, 300),
        (1.250, 2.049, 301, 500),
    ],
}


def calculate_aqi_for_pollutant(concentration, pollutant):
    """
    Calculate AQI for a single pollutant using linear interpolation.
    
    Args:
        concentration: Pollutant concentration
        pollutant: Pollutant name (pm25, pm10, o3, co, so2, no2)
    
    Returns:
        AQI value (0-500) or None if concentration is invalid
    """
    if pd.isna(concentration) or concentration < 0:
        return None
    
    breakpoints = AQI_BREAKPOINTS.get(pollutant)
    if not breakpoints:
        return None
    
    # Find the appropriate breakpoint
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            # Linear interpolation formula
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)
    
    # If concentration exceeds all breakpoints, return max AQI
    if concentration > breakpoints[-1][1]:
        return 500
    
    return None


def convert_unit_to_standard(value, parameter, unit):
    """
    Convert pollutant concentration to US EPA standard units.
    
    PM2.5, PM10: µg/m³ (no conversion needed if already in µg/m³)
    O3, CO, SO2, NO2: convert to ppm if in other units
    """
    if unit == "ppm":
        return value
    
    # Conversion factors for µg/m³ to ppm
    conversion_factors = {
        "o3": 24.45 / 48.0 / 1000,
        "co": 24.45 / 28.0 / 1000,
        "so2": 24.45 / 64.0 / 1000,
        "no2": 24.45 / 46.0 / 1000,
    }
    
    if parameter in conversion_factors and unit == "µg/m³":
        return value * conversion_factors[parameter]
    
    return value


def get_aqi_category(aqi):
    """Get AQI category based on AQI value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

