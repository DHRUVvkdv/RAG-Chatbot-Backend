import boto3
from boto3.dynamodb.conditions import Key, Attr
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import json


class LiveDataService:
    def __init__(self):
        # Initialize DynamoDB connection
        self.dynamodb = boto3.resource("dynamodb")
        self.table_name = os.environ.get("DYNAMODB_TABLE_NAME", "lewas-observations")
        self.table = self.dynamodb.Table(self.table_name)

    def get_latest_reading(
        self, medium: str = "air", metric: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the single most recent reading for a specific medium/metric.

        Args:
            medium: The medium type (air, water, etc.)
            metric: The specific metric (temperature, humidity, etc.)

        Returns:
            Latest reading or None if not found
        """
        try:
            logging.info(f"Fetching latest {medium}/{metric} reading")

            # Get instrument ID for weather station
            instrument_id = "3"  # This is for weather station

            # Query the table for the latest entry for this instrument
            response = self.table.query(
                KeyConditionExpression=Key("instrument_id").eq(instrument_id),
                ScanIndexForward=False,  # Sort in descending order (newest first)
                Limit=10,  # Get a few in case we need to filter
            )

            # Filter for the specific medium and metric
            filtered_items = []
            for item in response.get("Items", []):
                if "sample" in item and item["sample"].get("medium") == medium:
                    if metric is None or item["sample"].get("metric") == metric:
                        filtered_items.append(item)

            # Return the first (most recent) matching item
            if filtered_items:
                return filtered_items[0]
            return None

        except Exception as e:
            logging.error(f"Error fetching latest reading: {str(e)}")
            return None

    def get_recent_readings(
        self, medium: str = "air", metric: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent readings for a specific medium/metric.

        Args:
            medium: The medium type (air, water, etc.)
            metric: The specific metric (temperature, humidity, etc.)
            limit: Maximum number of readings to return

        Returns:
            List of recent readings
        """
        try:
            logging.info(f"Fetching recent {medium}/{metric} readings, limit: {limit}")

            # Get instrument ID for weather station

            instrument_id = "3"  # This is for weather station

            # Query the table for recent entries for this instrument
            response = self.table.query(
                KeyConditionExpression=Key("instrument_id").eq(instrument_id),
                ScanIndexForward=False,  # Sort in descending order (newest first)
                Limit=50,  # Get enough to filter and still have 'limit' results
            )

            # Filter for the specific medium and metric
            filtered_items = []
            for item in response.get("Items", []):
                if "sample" in item and item["sample"].get("medium") == medium:
                    if metric is None or item["sample"].get("metric") == metric:
                        filtered_items.append(item)
                        if len(filtered_items) >= limit:
                            break

            return filtered_items

        except Exception as e:
            logging.error(f"Error fetching recent readings: {str(e)}")
            return []

    def format_reading(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """Format a reading for output"""
        if not reading:
            return {}

        try:
            return {
                "timestamp": reading.get("datetime", ""),
                "value": reading.get("value", ""),
                "unit": reading.get("unit", ""),
                "medium": reading.get("sample", {}).get("medium", ""),
                "metric": reading.get("sample", {}).get("metric", ""),
            }
        except Exception as e:
            logging.error(f"Error formatting reading: {str(e)}")
            return {}
