import logging
from langchain.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock
from typing import Dict, Any
from models.query import QueryModel
from services.pinecone_service import query_pinecone
from services.live_data_service import LiveDataService
from datetime import datetime
import pytz

# Import model ID from pinecone service
from services.pinecone_service import BEDROCK_MODEL_ID


def classify_query(query_text: str) -> str:
    """
    Use LLM to classify the query and determine which handler to use.
    Returns:
    - "LIVE_DATA:medium:metric:mode" for live data queries
      (mode can be 'latest' or 'recent')
    - "VISUALIZATION:medium:metric" for visualization queries
    - "RAG" for document-based queries
    """
    CLASSIFICATION_PROMPT = """
    You are an AI assistant for the LEWAS lab at Virginia Tech. 
    Determine if this query requires:
    
    1. Live weather station data (asking about current temperature, humidity, pressure, etc.)
       - If asking for the "current", "latest", or "now" value, use "latest" mode
       - If asking for "recent" values or a plural form like "readings", use "recent" mode
    2. Visualization of weather data (asking for graphs, charts, trends)
    3. Document-based information (asking about LEWAS lab, research, operations, etc.)
    
    Weather station data includes:
    - Air: temperature, humidity, pressure
    - Rain: accumulation, duration, intensity
    - Wind: speed, direction
    - Battery: voltage
    
    Query: {question}
    
    Respond ONLY with one of these formats:
    - LIVE_DATA:medium:metric:mode (e.g., LIVE_DATA:air:temperature:latest or LIVE_DATA:air:humidity:recent)
    - VISUALIZATION:medium:metric (e.g., VISUALIZATION:air:humidity)
    - RAG
    """

    try:
        prompt_template = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        prompt = prompt_template.format(question=query_text)
        model = ChatBedrock(model_id=BEDROCK_MODEL_ID)
        response = model.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error classifying query: {str(e)}")
        return "RAG"  # Default to RAG on error


def extract_query_params(query_type: str) -> Dict[str, str]:
    """Extract parameters from the classified query type."""
    parts = query_type.split(":")
    params = {}

    if len(parts) >= 3:
        params["medium"] = parts[1]
        params["metric"] = parts[2]

    # Extract mode (latest or recent)
    if len(parts) >= 4:
        params["mode"] = parts[3]
    else:
        params["mode"] = "latest"  # Default to latest

    return params


def handle_live_data_query(query_text: str, params: Dict[str, str]) -> QueryModel:
    """Handle live data queries by fetching sensor readings."""
    query_model = QueryModel(query_text=query_text)

    try:
        service = LiveDataService()
        medium = params.get("medium", "air")
        metric = params.get("metric", "temperature")
        mode = params.get("mode", "latest")

        if mode == "latest":
            # Get single latest reading
            reading = service.get_latest_reading(medium=medium, metric=metric)

            if not reading:
                query_model.answer_text = f"Sorry, I couldn't retrieve the latest {medium} {metric} data from the LEWAS weather station."
                query_model.is_complete = True
                return query_model

            # Format for output
            formatted_reading = service.format_reading(reading)
            eastern_tz = pytz.timezone("US/Eastern")
            timestamp_utc = datetime.fromisoformat(
                formatted_reading["timestamp"].replace("Z", "+00:00")
            )
            timestamp_est = timestamp_utc.astimezone(eastern_tz)
            formatted_time = timestamp_est.strftime("%Y-%m-%d %H:%M:%S EST")

            response_text = f"The latest {medium} {metric} reading from the LEWAS weather station is:\n\n"
            response_text += f"{formatted_reading['value']} {formatted_reading['unit']} (recorded at {formatted_time})"

            query_model.answer_text = response_text

        else:  # "recent" mode
            # Get multiple recent readings
            readings = service.get_recent_readings(
                medium=medium, metric=metric, limit=10
            )

            if not readings:
                query_model.answer_text = f"Sorry, I couldn't retrieve any recent {medium} {metric} data from the LEWAS weather station."
                query_model.is_complete = True
                return query_model

            # Format for output
            response_text = f"Here are the recent {medium} {metric} readings from the LEWAS weather station:\n\n"

            for reading in readings:
                formatted_reading = service.format_reading(reading)
                eastern_tz = pytz.timezone("US/Eastern")
                timestamp_utc = datetime.fromisoformat(
                    formatted_reading["timestamp"].replace("Z", "+00:00")
                )
                timestamp_est = timestamp_utc.astimezone(eastern_tz)
                formatted_time = timestamp_est.strftime("%Y-%m-%d %H:%M:%S EST")

                response_text += f"- {formatted_time}: {formatted_reading['value']} {formatted_reading['unit']}\n"

            query_model.answer_text = response_text

        query_model.sources = [
            f"Live data from LEWAS weather station ({medium}/{metric})"
        ]
        query_model.is_complete = True

    except Exception as e:
        logging.error(f"Error handling live data query: {str(e)}")
        query_model.answer_text = (
            f"An error occurred while fetching weather data: {str(e)}"
        )
        query_model.is_complete = False

    return query_model


def handle_visualization_query(query_text: str, params: Dict[str, str]) -> QueryModel:
    """Handle visualization queries (placeholder for now)."""
    query_model = QueryModel(query_text=query_text)

    try:
        medium = params.get("medium", "air")
        metric = params.get("metric", "temperature")

        # Placeholder for now
        query_model.answer_text = (
            f"You requested a visualization of {medium} {metric} data. "
            f"This feature is coming soon! For now, I can provide you with the latest readings."
        )

        # Fall back to live data
        service = LiveDataService()
        readings = service.get_latest_readings(medium=medium, metric=metric, limit=10)

        if readings:
            response_text = (
                query_model.answer_text + "\n\nHere are the latest readings:\n\n"
            )

            for reading in readings:
                eastern_tz = pytz.timezone("US/Eastern")
                timestamp_utc = datetime.fromisoformat(
                    formatted_reading["timestamp"].replace("Z", "+00:00")
                )
                timestamp_est = timestamp_utc.astimezone(eastern_tz)
                formatted_time = timestamp_est.strftime("%Y-%m-%d %H:%M:%S EST")
                value = reading["value"]
                unit = reading["unit"]

                response_text += f"- {formatted_time}: {value} {unit}\n"

            query_model.answer_text = response_text

        query_model.sources = [
            f"Live data from LEWAS weather station ({medium}/{metric})"
        ]
        query_model.is_complete = True

    except Exception as e:
        logging.error(f"Error handling visualization query: {str(e)}")
        query_model.answer_text = (
            f"An error occurred while processing your visualization request: {str(e)}"
        )
        query_model.is_complete = False

    return query_model
