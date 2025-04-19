import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import boto3
import io
import os
import uuid
import logging
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any, Optional, Tuple
import traceback
import sys

# Configure logging to show more details
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("visualization_service")


class VisualizationService:
    def __init__(self):
        # Log environment variables (without showing actual secrets)
        logger.info(f"S3_BUCKET from env: {os.environ.get('S3_BUCKET', 'Not Set')}")
        logger.info(f"AWS_REGION from env: {os.environ.get('AWS_REGION', 'Not Set')}")
        logger.info(
            f"AWS_ACCESS_KEY_ID set: {'Yes' if os.environ.get('AWS_ACCESS_KEY_ID') else 'No'}"
        )
        logger.info(
            f"AWS_SECRET_ACCESS_KEY set: {'Yes' if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'No'}"
        )

        try:
            # Try to initialize the S3 client with profile if needed
            if os.environ.get("AWS_PROFILE"):
                logger.info(f"Using AWS profile: {os.environ.get('AWS_PROFILE')}")
                session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
                self.s3_client = session.client("s3")
            else:
                logger.info("Using default credentials")
                self.s3_client = boto3.client("s3")

            # Test S3 connection
            logger.info("Testing S3 connection...")
            buckets = self.s3_client.list_buckets()
            logger.info(
                f"S3 connection successful! Found {len(buckets['Buckets'])} buckets"
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a dummy client anyway to avoid further errors
            self.s3_client = boto3.client("s3")

        self.bucket_name = os.environ.get("S3_BUCKET", "lewas-chatbot-v2")
        logger.info(f"Using bucket: {self.bucket_name}")
        self.graph_prefix = "data/v1/visualizations/"
        logger.info(f"Using prefix: {self.graph_prefix}")

        # Configure matplotlib/seaborn
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)

    def create_visualization(
        self, medium: str, metric: str, data: List[Dict[str, Any]], query_id: str
    ) -> Optional[str]:
        """
        Create a visualization and upload it to S3.
        """
        try:
            logger.info(
                f"Creating visualization for {medium}/{metric} with {len(data)} data points"
            )

            if not data:
                logger.warning(
                    f"No data provided for visualization of {medium}/{metric}"
                )
                return None

            # Extract data for plotting
            timestamps = []
            values = []
            unit = ""

            # Sort data by timestamp
            sorted_data = sorted(data, key=lambda x: x.get("timestamp", ""))
            logger.info(f"Sorted {len(sorted_data)} data points by timestamp")

            eastern_tz = pytz.timezone("US/Eastern")

            for point in sorted_data:
                timestamp_str = point.get("timestamp", "")
                if timestamp_str:
                    # Convert to datetime and localize to EST
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    timestamp_est = timestamp.astimezone(eastern_tz)
                    timestamps.append(timestamp_est)

                value = point.get("value", 0)
                try:
                    # Convert string values to float
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0
                values.append(value)

                # Get the unit from the first point
                if not unit and "unit" in point:
                    unit = point.get("unit", "")

            logger.info(
                f"Processed {len(timestamps)} timestamps and {len(values)} values"
            )
            logger.info(f"Unit: {unit}")

            # Create the plot
            logger.info("Creating matplotlib figure")
            plt.figure(figsize=(12, 6))

            # Plot the data
            plt.plot(timestamps, values, "b-o", linewidth=2, markersize=6)

            # Set labels and title
            plt.title(f"{medium.capitalize()} {metric.capitalize()} Trend")
            plt.xlabel("Time (EST)")
            plt.ylabel(f"{metric.capitalize()} ({unit})")

            # Format x-axis to show time properly
            plt.gcf().autofmt_xdate()

            # Add grid
            plt.grid(True, alpha=0.7)

            # Ensure the plot is tight
            plt.tight_layout()
            logger.info("Plot creation complete")

            # Generate a unique filename
            filename = f"{query_id}_{medium}_{metric}_{uuid.uuid4().hex[:8]}.png"
            s3_key = f"{self.graph_prefix}{filename}"
            logger.info(f"Generated filename: {filename}")
            logger.info(f"S3 key: {s3_key}")

            # Save to buffer
            logger.info("Saving plot to memory buffer")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300)
            buf.seek(0)
            logger.info("Plot saved to buffer successfully")

            # Try different upload methods
            try:
                # Method 1: Using upload_fileobj
                logger.info(
                    f"Attempting to upload to S3 with upload_fileobj: bucket={self.bucket_name}, key={s3_key}"
                )
                self.s3_client.upload_fileobj(
                    buf,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        "ContentType": "image/png",
                        "Expires": datetime.now() + timedelta(hours=24),
                    },
                )
                logger.info("Successfully uploaded to S3 using upload_fileobj")
            except Exception as upload_error:
                logger.error(f"Error with upload_fileobj: {str(upload_error)}")

                try:
                    # Method 2: Using put_object
                    logger.info("Trying alternative upload method with put_object")
                    buf.seek(0)  # Reset buffer position
                    response = self.s3_client.put_object(
                        Body=buf.getvalue(),
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        ContentType="image/png",
                    )
                    logger.info(f"Successfully uploaded using put_object: {response}")
                except Exception as put_error:
                    logger.error(f"Error with put_object: {str(put_error)}")

                    # Fallback to local save for testing
                    local_path = f"/tmp/{filename}"
                    logger.info(f"Falling back to local save: {local_path}")
                    plt.savefig(local_path, format="png", dpi=300)
                    plt.close()
                    return f"Local file saved (testing only): {local_path}"

            # Generate a signed URL that expires in 24 hours
            logger.info("Generating presigned URL")
            try:
                direct_url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            except Exception as url_error:
                logger.error(f"Error generating direct URL: {str(url_error)}")

            plt.close()
            logger.info("Visualization process complete")

            return direct_url

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            plt.close()
            return None

    def get_data_for_timeframe(
        self,
        dynamodb_table,
        instrument_id: str,
        medium: str,
        metric: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get data from DynamoDB for the specified timeframe.
        """
        try:
            logger.info(
                f"Getting data for: instrument_id={instrument_id}, medium={medium}, metric={metric}, hours={hours}"
            )

            # Calculate the start time (24 hours ago)
            start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            logger.info(f"Start time: {start_time}")

            # Query the table for recent entries
            logger.info(
                f"Querying DynamoDB table: {dynamodb_table.table_name if hasattr(dynamodb_table, 'table_name') else 'unknown'}"
            )
            response = dynamodb_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key(
                    "instrument_id"
                ).eq(instrument_id)
                & boto3.dynamodb.conditions.Key("datetime").gte(start_time),
                ScanIndexForward=True,  # Sort in ascending order (oldest first)
            )
            logger.info(f"Query returned {len(response.get('Items', []))} items")

            # Filter for the specific medium and metric
            filtered_data = []
            for item in response.get("Items", []):
                if "sample" in item and item["sample"].get("medium") == medium:
                    if item["sample"].get("metric") == metric:
                        filtered_data.append(
                            {
                                "timestamp": item.get("datetime", ""),
                                "value": item.get("value", ""),
                                "unit": item.get("unit", ""),
                                "medium": item["sample"].get("medium", ""),
                                "metric": item["sample"].get("metric", ""),
                            }
                        )

            logger.info(
                f"Filtered down to {len(filtered_data)} items for {medium}/{metric}"
            )

            # Log a sample item for debugging
            if filtered_data:
                logger.info(f"Sample data point: {filtered_data[0]}")

            return filtered_data

        except Exception as e:
            logger.error(f"Error retrieving data for visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return []
